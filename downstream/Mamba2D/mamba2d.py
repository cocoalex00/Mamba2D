import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.layers import DropPath
from collections import OrderedDict

from mmengine.model import BaseModule
from mmengine.registry import MODELS

from .kernels.wavefront_cuda import wavefront_scan_cuda
from .utils import LayerNorm2D, MLP, StarReLU, SepConv, Downsampling, Scale, RoPEAttention2D

# Default cache size of 8 is too low for training runs with multiple compiled graphs.
torch._dynamo.config.cache_size_limit = 64

class M2DBlock(nn.Module):
    '''
        Minimal version of Mamba 2D block, adapted from base 1D implementation:
        https://github.com/alxndrTL/mamba.py
    '''
    def __init__(self,
                 d_model: int, # D
                 d_state: int = 16, # N in paper/comments
                 expand_factor: int = 1, # E in paper/comments
                 dt_rank: int|str = 'auto',
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dt_init: str = "random", # "random" or "constant"
                 dt_scale: float = 1.0,
                 dt_init_floor = 1e-4,
                 double_scans: bool = False, # Enable/Disable 2 scans
                 local_path: bool = False,   # Enable/Disable local path
                 scan_start: str = "TL",     # Set scan start direction
                 recomp: str = "partial"     # Set recomputation extent
                ):
        super().__init__()

        self.double_scans = double_scans
        self.d_model = d_model
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments
        self.dt_rank = dt_rank
        self.recomp = recomp

        # Parse scan start direction
        match scan_start:
            case "TL"|"BR":
                if not double_scans:
                    self.scan_start = scan_start
                else:
                    raise ValueError("Cannot choose scan direction if double scans are enabled!")
            case _:
                raise ValueError(f"Invalid Scan Direction, expected TL or BR and got {scan_start}")

        # Add SepConv local path
        self.local_norm = nn.LayerNorm(self.d_model) if local_path else None
        self.local_path = SepConv(self.d_model, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=StarReLU,
        bias=False, kernel_size=3, padding=1, residual = False) if local_path else None

        # projects block input from D to ED
        self.in_proj = nn.Linear(self.d_model, self.d_inner)

        self.act1 = nn.GELU()

        # delta bias
        if self.dt_rank == 'auto': self.dt_rank = math.ceil(self.d_model / 16)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        dt_init_std = self.dt_rank**-0.5 * dt_scale

        # projects x to input-dependent deltaT, deltaL, BT, BL, C
        self.x_proj = nn.Linear(self.d_inner, 2 * self.dt_rank + 3 * self.d_state, bias=False)

        # projects deltaT from dt_rank to d_inner
        self.dt_projT = nn.Linear(self.dt_rank, self.d_inner)

        # projects deltaL from dt_rank to d_inner
        self.dt_projL = nn.Linear(self.dt_rank, self.d_inner)

        # Init dt_projs
        if dt_init == "constant":
            nn.init.constant_(self.dt_projT.weight, dt_init_std)
            nn.init.constant_(self.dt_projL.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_projT.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.dt_projL.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        with torch.no_grad():
            self.dt_projT.bias.copy_(inv_dt)
            self.dt_projL.bias.copy_(inv_dt)

        # S4D real initialization
        AT = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        AL = AT.detach().clone()

        self.AT_log = nn.Parameter(torch.log(AT))
        self.AT_log._no_weight_decay = True

        self.AL_log = nn.Parameter(torch.log(AL))
        self.AL_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Init second set of params for bwd scan direction
        if self.double_scans:
            # S4D real initialization
            ATb = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            ALb = ATb.detach().clone()

            self.AT_log_b = nn.Parameter(torch.log(ATb))
            self.AT_log_b._no_weight_decay = True

            self.AL_log_b = nn.Parameter(torch.log(ALb))
            self.AL_log_b._no_weight_decay = True

            self.D_b = nn.Parameter(torch.ones(self.d_inner))
            self.D_b._no_weight_decay = True

            # projects x to input-dependent deltaT, deltaL, BT, BL, C
            self.x_proj_b = nn.Linear(self.d_inner, 2 * self.dt_rank + 3 * self.d_state, bias=False)

            # projects deltaT from dt_rank to d_inner
            self.dt_projT_b = nn.Linear(self.dt_rank, self.d_inner)

            # projects deltaL from dt_rank to d_inner
            self.dt_projL_b = nn.Linear(self.dt_rank, self.d_inner)

            # Init dt_projs
            if dt_init == "constant":
                nn.init.constant_(self.dt_projT_b.weight, dt_init_std)
                nn.init.constant_(self.dt_projL_b.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(self.dt_projT_b.weight, -dt_init_std, dt_init_std)
                nn.init.uniform_(self.dt_projL_b.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            with torch.no_grad():
                self.dt_projT_b.bias.copy_(inv_dt)
                self.dt_projL_b.bias.copy_(inv_dt)

        self.act2 = nn.GELU()

        # projects block output from ED back to D
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, x):
        # x : (B, H, W, ED)

        if self.local_path:
            local_features = self.local_path(self.local_norm(x))

        if self.scan_start == "BR":
            x = torch.flip(x, dims=(1,2))

        x = self.in_proj(x)
        x = self.act1(x)
        x = self.ssm(x)
        x = self.act2(x)
        x = self.out_proj(x)

        if self.scan_start == "BR":
            x = torch.flip(x, dims=(1,2))

        if self.local_path:
            x = x + local_features

        return x

    def ssm(self, x):
        # x : (B, H, W, ED)
        # y : (B, H, W, ED)

        y = wavefront_scan_cuda(x, self.AT_log, self.AL_log,
                                self.x_proj.weight,
                                self.dt_projT.weight, self.dt_projT.bias,
                                self.dt_projL.weight, self.dt_projL.bias,
                                self.D, self.recomp)

        if self.double_scans:
            # Implement bwd scan as flipped input
            x_b = torch.flip(x, dims=[1,2])

            y_b = wavefront_scan_cuda(x_b, self.AT_log_b, self.AL_log_b,
                                      self.x_proj_b.weight,
                                      self.dt_projT_b.weight, self.dt_projT_b.bias,
                                      self.dt_projL_b.weight, self.dt_projL_b.bias,
                                      self.D_b, self.recomp)

            y = (y + y_b)/2 # average output of both scans

        return y

# Adapted from: https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py#L479
class ResBlock(nn.Module):
    def __init__(self,
                 # Block Params
                 embed_dim: int = 64,
                 token_mixer: str = "2D",
                 mlp_expand_factor: int = 4,
                 drop_path: float = 0., # Drop path regularization
                 res_scale_init_value=None, # Init residual scaling factor
                 scan_start: str = None,    # Select Mamba2D scan direction start
                 recomp: str = "partial"    # Set recomputation extent for Mamba2D
                ):
        super().__init__()

        self.embed_dim = embed_dim

        self.norm1 = nn.LayerNorm(embed_dim)

        # NOTE: All token mixers should expect inputs as B,H,W,C!

        match token_mixer:
            case "2D":
                self.token_mixer = M2DBlock(embed_dim,
                                            scan_start=scan_start,
                                            recomp=recomp)
            case "2D_dbl_scan":
                self.token_mixer = M2DBlock(embed_dim,
                                            scan_start=scan_start,
                                            recomp=recomp,
                                            double_scans=True)
            case "2D_local":
                self.token_mixer = M2DBlock(embed_dim,
                                            scan_start=scan_start,
                                            recomp=recomp,
                                            local_path=True)
            case "RopeAttention":
                self.token_mixer = RoPEAttention2D(embed_dim)
            case _:
                raise ValueError(f"Invalid token mixer type: {token_mixer}")

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_scale1 = Scale(embed_dim, res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = nn.LayerNorm(embed_dim)

        self.MLP = MLP(embed_dim, mlp_expand_factor)

        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_scale2 = Scale(embed_dim, res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x):

        x = self.res_scale1(x) + self.drop_path1(self.token_mixer(self.norm1(x)))
        x = self.res_scale2(x) + self.drop_path2(self.MLP(self.norm2(x)))

        return x

@MODELS.register_module()
class Mamba2D(BaseModule):
    def __init__(self,

                 # Input Params
                 in_channels: int = 3,

                 # Model Params
                 channel_last: bool = True,
                 n_blocks: list|tuple = [3,3,9,3],
                 ds_stages: list|tuple = ["mf_stem","mf_2","mf_2","mf_2"],
                 embed_dim: list|tuple = [64,128,320,512],
                 token_mixer: list|tuple = ["2D_local", "2D_local", "RopeAttention", "RopeAttention"],
                 featmaps_out: bool = False,
                 drop_path_rate: float = 0.1,
                 res_scale_init_values: list|tuple = [None, None, 1.0, 1.0],
                 inter_flip: bool = False,      # Enable interleaved scan dirs
                 recomp: str = "partial",       # Set M2D recomputation extent
                 pt_ckpt=None,
                 pt_layers=('stages', 'ds', 'norm'),
                 init_cfg=None,
                 ):
        super().__init__(init_cfg=init_cfg)

        # Create model
        self.in_channels = in_channels
        self.channel_last = channel_last
        self.n_blocks = n_blocks
        self.ds_stages = ds_stages
        self.embed_dim = embed_dim
        self.token_mixer = token_mixer
        self.featmaps_out = featmaps_out
        self.inter_flip = inter_flip
        self.pt_ckpt = pt_ckpt
        self.pt_layers = pt_layers

        if not(len(n_blocks) ==
               len(embed_dim) ==
               len(token_mixer) ==
               len(res_scale_init_values)):
            raise ValueError("Check backbone init_args are defined for each stage!")

        # Create stem/downsample blocks
        self.ds = nn.ModuleDict()
        for d in range(len(self.ds_stages)):
            if (d == 0):

                ds_params = {
                    "name" : "stem",
                    "in_channels" : in_channels,
                    "out_channels" : embed_dim[0]
                }

            else:
                ds_params = {
                    "name" : f"ds{d}",
                    "in_channels" : embed_dim[d-1],
                    "out_channels" : embed_dim[d]
                }

            match self.ds_stages[d]:
                case 1:
                    ds_params["kernel_size"] = 1
                case 2:
                    ds_params["kernel_size"] = 2
                    ds_params["stride"] = 2
                case "mf_2": # Metaformer downsample + padding
                    ds_params["pre_norm"] = None
                    ds_params["kernel_size"] = 3
                    ds_params["stride"] = 2
                    ds_params["padding"] = 1
                case 4:
                    ds_params["kernel_size"] = 4
                    ds_params["stride"] = 4
                    ds_params["padding"] = 1
                case "mf_stem":
                    ds_params["kernel_size"] = 7
                    ds_params["stride"] = 4
                    ds_params["padding"] = 2
                    ds_params["post_norm"] = LayerNorm2D
                case _:
                    raise ValueError(f"Downsample scale {self.ds_stages[d]} for layer {ds_params['name']} not implemented!")

            self.ds.add_module(ds_params.pop("name"), Downsampling(**ds_params))

        # Create stages
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(n_blocks))]

        dp_count = 0
        self.stages = nn.ModuleDict()
        for i in range(len(n_blocks)):
            blocks_list = []
            for n in range(n_blocks[i]):
                scan_start = "BR" if self.inter_flip and (n % 2 == 1) else "TL" # Reverse scan dir for odd blocks

                blocks_list.append(
                    ResBlock(embed_dim[i],
                             token_mixer=token_mixer[i],
                             drop_path=dp_rates[dp_count+n],
                             res_scale_init_value=res_scale_init_values[i],
                             scan_start=scan_start,
                             recomp=recomp
                    )
                )

            # Add post-norm to stage
            blocks_list.append(nn.LayerNorm(embed_dim[i]))

            self.stages.add_module(str(i), nn.Sequential(*blocks_list))
            dp_count += n_blocks[i]

        # Init output feature map dict
        if (self.featmaps_out):
            self.featmaps = OrderedDict()

    def forward(self, x):
        for ds,(feat_name,stage) in zip(self.ds.values(), self.stages.items()):
            x = ds(x)

            x = rearrange(x, "b c h w -> b h w c")
            x = stage(x)
            x = rearrange(x, "b h w c -> b c h w")

            if (self.featmaps_out):
                self.featmaps[feat_name] = x

        if (self.featmaps_out):
            return tuple(self.featmaps.values())   # tuple for FPN
        else:
            return x

    def init_weights(self):
        super().init_weights()
        if self.pt_ckpt is None:
            raise ValueError(
                "No pretrained checkpoint specified (set pt_ckpt in backbone config).")
        print(f"Loading pretrained backbone weights from {self.pt_ckpt}...")
        try:
            pt_ckpt = torch.load(self.pt_ckpt, weights_only=True, mmap=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Pretrained checkpoint not found: {self.pt_ckpt}")
        pt_sd = {n.replace("backbone.", ""): p
                 for n, p in pt_ckpt["state_dict"].items()
                 if any(l in n for l in self.pt_layers)}
        missing, unexpected = self.load_state_dict(pt_sd, strict=False)
        missing   = [k for k in missing   if any(l in k for l in self.pt_layers)]
        unexpected = [k for k in unexpected if any(l in k for l in self.pt_layers)]
        if missing or unexpected:
            raise RuntimeError(
                f"Backbone/checkpoint key mismatch.\n"
                f"Missing:    {missing}\nUnexpected: {unexpected}")
        print("Pretrained backbone weights loaded successfully.")
