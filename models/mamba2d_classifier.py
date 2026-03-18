import torch
import torch.nn as nn
import lightning as L

from timm.loss import SoftTargetCrossEntropy
from torchmetrics.classification import Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

class Mamba2DClassifier(L.LightningModule):
    def __init__(self,
                 # Model Params
                 backbone: torch.nn.Module,
                 head: torch.nn.Module,
                 
                 # Input Params
                 n_classes: int = 1000,
                 cutmix: bool = True,

                 # Optimiser Params
                 lr: float = 0.004,
                 weight_decay: float = 0.05,
                 warmup_pct: float = 0.05,
                 ):
        super().__init__()

        self.n_classes = n_classes

        self.backbone = backbone

        self.head = head

        self.norm = nn.LayerNorm(backbone.embed_dim[-1])

        # Losses
        if (cutmix):
            self.train_loss = SoftTargetCrossEntropy()
        else:
            self.train_loss = nn.CrossEntropyLoss()

        # NOTE: When cutmix=True, val/ce_loss will be misreported (CutMix produces
        # soft labels but val uses hard labels, causing a shape mismatch that Lightning
        # silently logs as 0). Known issue; does not affect training or val/accuracy.
        self.val_loss = nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)

        # Optimiser
        self.lr = lr
        self.weight_decay = weight_decay

        if (0 <= warmup_pct < 1): self.warmup_pct = warmup_pct
        else: raise ValueError(f"warmup_pct ({warmup_pct}) must be between 0.0 "
                                "(0%) and 1.0 (100%)!")

        # Set model to channels last if required
        if backbone.channel_last: self.to(memory_format=torch.channels_last)

    def setup(self, stage):
        print(self)

    @torch.compile
    def forward(self, x):
        x = self.backbone(x)

        # x.mean[-2,-1] applies avg pooling on H,W dims
        x = self.head(self.norm(x.mean([-2,-1])))

        return x

    def configure_optimizers(self):
        # 1. Separate Parameters for Weight Decay
        decay, no_decay = [], []

        for p in self.parameters():
            if not p.requires_grad:
                continue

            # Check for explicit no_decay flag (Mamba params) OR 1D heuristic (Norms/Biases/Scales)
            if getattr(p, "_no_weight_decay", False) or p.dim() < 2:
                no_decay.append(p)
            else:
                decay.append(p)

        wd = self.weight_decay
        
        # 2. Param groups
        param_groups = [
            {"params": decay,    "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        # 3. Initialize AdamW
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 4. Define Scheduler (Linear Warmup -> Cosine Decay)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_pct)
        main_steps = total_steps - warmup_steps

        # Scheduler 1: Linear Warmup (0 -> lr)
        scheduler_warmup = LinearLR(
            optimizer, 
            start_factor=0.0001, # Start effectively at 0
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # Scheduler 2: Cosine Decay (lr -> min_lr)
        scheduler_cosine = CosineAnnealingLR(
            optimizer, 
            T_max=main_steps, 
            eta_min=1e-6
        )

        scheduler = SequentialLR(
            optimizer, 
            schedulers=[scheduler_warmup, scheduler_cosine], 
            milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def training_step(self, batch, batch_idx):
        x, labels = batch

        preds = self(x)

        loss = self.train_loss(preds, labels)

        self.log_dict({
                "train/ce_loss" : loss,
                "train/accuracy" : self.accuracy(preds, labels),
            },
            batch_size = self.trainer.datamodule.batch_size,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

        return(loss)

    def validation_step(self, batch, batch_idx):
        x, labels = batch

        with torch.inference_mode():
            preds = self(x)

        loss = self.val_loss(preds, labels)

        self.log_dict({
                "val/ce_loss" : loss,
                "val/accuracy" : self.accuracy(preds, labels),
            },
            batch_size = self.trainer.datamodule.batch_size,
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
