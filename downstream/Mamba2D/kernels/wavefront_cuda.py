import os
import torch
import torch.library
import torch.nn.functional as F

from torch.utils import cpp_extension

# Load the JIT extension
current_dir = os.path.dirname(os.path.abspath(__file__))
wf_cuda = cpp_extension.load(
    name="wavefront_cuda",
    sources=[
        os.path.join(current_dir, "wf_cuda/wf_cuda_bind.cpp"),
        os.path.join(current_dir, "wf_cuda/wf_cuda.cu"),
    ],
    extra_cflags=["-O2", "-fvisibility=hidden"],
    extra_cuda_cflags=["-Xptxas", "-O3"],
    verbose=False
)

# -----------------------------------------------------------------------------
# 1. Register Custom Operators for torch.compile Support
# -----------------------------------------------------------------------------

@torch.library.custom_op("mamba2d::wf_fwd", mutates_args={"hs"})
def wf_fwd_op(hs: torch.Tensor,
              x: torch.Tensor,
              deltaT: torch.Tensor,
              deltaL: torch.Tensor,
              BT: torch.Tensor,
              BL: torch.Tensor,
              AT: torch.Tensor,
              AL: torch.Tensor) -> None:
    wf_cuda.wf_fwd(hs, x, deltaT, deltaL, BT, BL, AT, AL)

@wf_fwd_op.register_fake
def wf_fwd_fake(hs, x, deltaT, deltaL, BT, BL, AT, AL):
    return

@torch.library.custom_op("mamba2d::wf_bwd", mutates_args={"grad_output", "dDAT", "dDAL", "omega"})
def wf_bwd_op(
    hs: torch.Tensor,
    x: torch.Tensor,
    deltaT: torch.Tensor,
    deltaL: torch.Tensor,
    AT: torch.Tensor,
    AL: torch.Tensor,
    grad_output: torch.Tensor, # Input & Output (In-place dBX)
    dDAT: torch.Tensor,        # Output (Write)
    dDAL: torch.Tensor,        # Output (Write)
    omega: torch.Tensor        # Scratch (Write)
) -> None:
    wf_cuda.wf_bwd(hs, x, deltaT, deltaL, AT, AL, grad_output, dDAT, dDAL, omega)

@wf_bwd_op.register_fake
def wf_bwd_fake(hs, x, deltaT, deltaL, AT, AL, grad_output, dDAT, dDAL, omega):
    return

# -----------------------------------------------------------------------------
# 2. Helper Functions (Projection & Fused Math)
# -----------------------------------------------------------------------------

@torch.compile
def proj_params(x, AT_log, AL_log, x_proj_w, dt_projT_w, dt_projT_b, dt_projL_w, dt_projL_b):
    """
    Compute SSM parameters from x. Called in forward and recomputed in backward.
    """
    # x : (B, H, W, ED)
    d_state = AT_log.shape[1]
    dt_rank = dt_projT_w.shape[1]

    AT = -torch.exp(AT_log.float())  # (ED, N)
    AL = -torch.exp(AL_log.float())  # (ED, N)

    delta2BC = F.linear(x, x_proj_w)  # (B, H, W, 2*dt_rank + 3*d_state)

    # Splits proj: (B, H, W, ...) -> deltaT, deltaL, BT, BL, C
    deltaT, deltaL, BT, BL, C = torch.split(
        delta2BC,
        [dt_rank, dt_rank, d_state, d_state, d_state],
        dim=-1
    )

    deltaT = F.softplus(F.linear(deltaT, dt_projT_w, dt_projT_b))
    deltaL = F.softplus(F.linear(deltaL, dt_projL_w, dt_projL_b))

    return (
        deltaT.to(dtype=x.dtype),
        deltaL.to(dtype=x.dtype),
        AT.to(dtype=x.dtype),
        AL.to(dtype=x.dtype),
        BT.to(dtype=x.dtype),
        BL.to(dtype=x.dtype),
        C.to(dtype=x.dtype)
    )

@torch.compile
def fused_backward_math(dDAT, dDAL, dHS_total, deltaT, deltaL, AT, AL, BT, BL, x):
    """
    Compute gradients of discretized SSM parameters.
    """
    # 1. Recompute Discretized Gates (Fused Memory Optimization)
    deltaAT = torch.exp(deltaT.unsqueeze(-1) * AT)
    deltaAL = torch.exp(deltaL.unsqueeze(-1) * AL)

    # 2. Gradients from Gates (dZ terms)
    dZ_T = dDAT * deltaAT
    dZ_L = dDAL * deltaAL

    d_AT = (dZ_T * deltaT.unsqueeze(-1)).sum(dim=(0, 1, 2))
    d_AL = (dZ_L * deltaL.unsqueeze(-1)).sum(dim=(0, 1, 2))

    d_deltaT = (dZ_T * AT).sum(dim=-1)
    d_deltaL = (dZ_L * AL).sum(dim=-1)
    del dZ_T, dZ_L

    # 3. Gradients for Input Terms (BXT / BXL)
    _, H, W, _, _ = dHS_total.shape
    row_indices = torch.arange(H, device=dHS_total.device).view(1, H, 1, 1, 1)
    col_indices = torch.arange(W, device=dHS_total.device).view(1, 1, W, 1, 1)

    dHS_T = dHS_total * (row_indices > 0)
    dHS_L = dHS_total * ((col_indices > 0) | (row_indices == 0))

    d_deltaT += (dHS_T * BT.unsqueeze(-2) * x.unsqueeze(-1)).sum(dim=-1)
    d_deltaL += (dHS_L * BL.unsqueeze(-2) * x.unsqueeze(-1)).sum(dim=-1)

    # Gradients for BT, BL, and x (from scan part)
    term_T = dHS_T * deltaT.unsqueeze(-1)
    term_L = dHS_L * deltaL.unsqueeze(-1)

    d_BT = (term_T * x.unsqueeze(-1)).sum(dim=-2)
    d_BL = (term_L * x.unsqueeze(-1)).sum(dim=-2)

    d_x_scan_T = (term_T * BT.unsqueeze(-2)).sum(dim=-1)
    d_x_scan_L = (term_L * BL.unsqueeze(-2)).sum(dim=-1)
    d_x_scan = d_x_scan_T + d_x_scan_L

    return d_x_scan, d_deltaT, d_deltaL, d_AT, d_AL, d_BT, d_BL


# -----------------------------------------------------------------------------
# 3. Main Autograd Function (Hybrid Fusion)
# -----------------------------------------------------------------------------

class wf_cuda_fn_pr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, deltaT, deltaL, AT, AL, BT, BL, C, D):
        bs, H, W, E = x.shape
        _, N = AT.shape

        hs = torch.empty((bs, H, W, E, N), dtype=x.dtype, device=x.device)
        wf_fwd_op(hs, x, deltaT, deltaL, BT, BL, AT, AL)

        # Output projection
        y = (hs * C.unsqueeze(-2)).sum(dim=-1) + x * D

        # Save inputs; hs recomputed in backward
        ctx.save_for_backward(x, deltaT, deltaL, AT, AL, BT, BL, C, D)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, deltaT, deltaL, AT, AL, BT, BL, C, D = ctx.saved_tensors
        bs, H, W, E = x.shape
        _, N = AT.shape

        # Cast grad_output to common type
        grad_output = grad_output.to(x.dtype)

        # Recompute hs (BT/BL not materialised)
        hs = torch.empty((bs, H, W, E, N), dtype=x.dtype, device=x.device)
        wf_fwd_op(hs, x, deltaT, deltaL, BT, BL, AT, AL)

        # Backward Output Projection
        dD = (grad_output * x).sum(dim=(0,1,2))
        dC = (grad_output.unsqueeze(-1) * hs).sum(dim=-2)
        dHS = grad_output.unsqueeze(-1) * C.unsqueeze(-2) # Becomes dBX

        # Backward Scan
        dDAT = torch.empty_like(hs)
        dDAL = torch.empty_like(hs)
        omega = torch.empty((2, bs, H, E, N), dtype=hs.dtype, device=hs.device)
        wf_bwd_op(hs, x, deltaT, deltaL, AT, AL, dHS, dDAT, dDAL, omega)
        del hs, omega

        # dHS masking handled inside fused_backward_math
        d_x_scan, d_deltaT, d_deltaL, d_AT, d_AL, d_BT, d_BL = fused_backward_math(
            dDAT, dDAL, dHS,
            deltaT, deltaL, AT, AL, BT, BL, x
        )

        d_x_direct = grad_output * D
        d_x = d_x_scan + d_x_direct

        return d_x, d_deltaT, d_deltaL, d_AT, d_AL, d_BT, d_BL, dC, dD


class wf_cuda_fn_fr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, AT_log, AL_log, x_proj_w, dt_projT_w, dt_projT_b, dt_projL_w, dt_projL_b, D):
        bs, H, W, E = x.shape
        _, N = AT_log.shape # AT_log is (E, N)

        # 1. Compute projections; not saved — recomputed with grad in backward
        with torch.no_grad():
            deltaT, deltaL, AT, AL, BT, BL, C = proj_params(
                x, AT_log, AL_log, x_proj_w,
                dt_projT_w, dt_projT_b, dt_projL_w, dt_projL_b
            )

        # 2. Run Scan Kernel
        hs = torch.empty((bs, H, W, E, N), dtype=x.dtype, device=x.device)
        wf_fwd_op(hs, x, deltaT, deltaL, BT, BL, AT, AL)

        # 3. Output Project
        y = (hs * C.unsqueeze(-2)).sum(dim=-1) + x * D

        # 4. Save inputs/weights only; projections are recomputed in backward
        ctx.save_for_backward(x, AT_log, AL_log, x_proj_w, dt_projT_w, dt_projT_b, dt_projL_w, dt_projL_b, D)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, AT_log, AL_log, x_proj_w, dt_projT_w, dt_projT_b, dt_projL_w, dt_projL_b, D = ctx.saved_tensors
        bs, H, W, E = x.shape

        # Ensure consistent dtype
        grad_output = grad_output.to(x.dtype)

        # -----------------------------------------------------------
        # PART A: Recompute Forward Projections (With Gradients!)
        # -----------------------------------------------------------
        with torch.enable_grad():
            # Detach inputs but require grad so we can backprop through them
            x_detached = x.detach().requires_grad_(True)
            AT_log_detached = AT_log.detach().requires_grad_(True)
            AL_log_detached = AL_log.detach().requires_grad_(True)

            x_proj_w_d = x_proj_w.detach().requires_grad_(True)
            dt_projT_w_d = dt_projT_w.detach().requires_grad_(True)
            dt_projT_b_d = dt_projT_b.detach().requires_grad_(True)
            dt_projL_w_d = dt_projL_w.detach().requires_grad_(True)
            dt_projL_b_d = dt_projL_b.detach().requires_grad_(True)

            deltaT, deltaL, AT, AL, BT, BL, C = proj_params(
                 x_detached, AT_log_detached, AL_log_detached, x_proj_w_d,
                 dt_projT_w_d, dt_projT_b_d, dt_projL_w_d, dt_projL_b_d
            )

        # -----------------------------------------------------------
        # PART B: Recompute Hidden State & Run Scan Backward
        # -----------------------------------------------------------
        N = AT.shape[1]
        hs = torch.empty((bs, H, W, E, N), dtype=x.dtype, device=x.device)

        wf_fwd_op(hs, x, deltaT.detach(), deltaL.detach(), BT.detach(), BL.detach(), AT.detach(), AL.detach())

        # Output Proj Gradients
        dD = (grad_output * x).sum(dim=(0,1,2))
        dC_val = (grad_output.unsqueeze(-1) * hs).sum(dim=-2)

        dHS = grad_output.unsqueeze(-1) * C.detach().unsqueeze(-2)

        # Scan Backward Kernel
        dDAT = torch.empty_like(hs)
        dDAL = torch.empty_like(hs)
        omega = torch.empty((2, bs, H, E, N), dtype=hs.dtype, device=hs.device)

        wf_bwd_op(hs, x, deltaT.detach(), deltaL.detach(), AT.detach(), AL.detach(), dHS, dDAT, dDAL, omega)
        del hs, omega

        # -----------------------------------------------------------
        # PART C: Fused Math (Compute Gradients of Activations)
        # -----------------------------------------------------------
        d_x_scan, d_deltaT, d_deltaL, d_AT, d_AL, d_BT, d_BL = fused_backward_math(
            dDAT, dDAL, dHS,
            deltaT.detach(),
            deltaL.detach(),
            AT.detach(),
            AL.detach(),
            BT.detach(),
            BL.detach(),
            x
        )

        # -----------------------------------------------------------
        # PART D: Autograd Backprop through Projections
        # -----------------------------------------------------------
        outputs = [deltaT, deltaL, BT, BL, C, AT, AL]
        grad_outputs = [d_deltaT, d_deltaL, d_BT, d_BL, dC_val, d_AT, d_AL]

        inputs = [
            x_detached, AT_log_detached, AL_log_detached,
            x_proj_w_d, dt_projT_w_d, dt_projT_b_d, dt_projL_w_d, dt_projL_b_d
        ]

        grads = torch.autograd.grad(
            outputs, inputs, grad_outputs,
            retain_graph=False
        )

        (d_x_proj, d_AT_log_proj, d_AL_log_proj,
         d_x_proj_w, d_dt_projT_w, d_dt_projT_b, d_dt_projL_w, d_dt_projL_b) = grads

        # -----------------------------------------------------------
        # PART E: Combine Gradients for x
        # -----------------------------------------------------------
        d_x_direct = grad_output * D
        d_x = d_x_direct + d_x_scan + d_x_proj

        return d_x, d_AT_log_proj, d_AL_log_proj, d_x_proj_w, d_dt_projT_w, d_dt_projT_b, d_dt_projL_w, d_dt_projL_b, dD

def wavefront_scan_cuda(x, AT_log, AL_log, x_proj_w, dt_projT_w, dt_projT_b, dt_projL_w, dt_projL_b, D, recomp="partial"):
    """
    Args:
        recomp (str):
            - 'partial': Computes projections once. Faster forward/backward, but higher VRAM usage (stores activations).
            - 'full':    Recomputes projections in backward. Slower (~15%), but minimal VRAM usage.
    """
    _, H, W, _ = x.shape
    if H < 2 or W < 2:
        raise ValueError(f"Wavefront scan kernel requires spatial dims > 1. Got H={H}, W={W}.")

    # Maybe cast params (leave AT/AL for stability, cast in proj fn.)
    x_proj_w = x_proj_w.to(x.dtype)
    dt_projT_w = dt_projT_w.to(x.dtype)
    dt_projT_b = dt_projT_b.to(x.dtype)
    dt_projL_w = dt_projL_w.to(x.dtype)
    dt_projL_b = dt_projL_b.to(x.dtype)
    D = D.to(x.dtype)

    match recomp:
        case "partial":
            deltaT, deltaL, AT, AL, BT, BL, C = proj_params(x, AT_log, AL_log, x_proj_w,
                                                            dt_projT_w, dt_projT_b,
                                                            dt_projL_w, dt_projL_b)

            return wf_cuda_fn_pr.apply(x, deltaT, deltaL, AT, AL, BT, BL, C, D)

        case "full":
            return wf_cuda_fn_fr.apply(x, AT_log, AL_log, x_proj_w,
                                       dt_projT_w, dt_projT_b,
                                       dt_projL_w, dt_projL_b, D)
        case _:
            raise ValueError(f"Unknown recomputation mode, got {recomp}...")
