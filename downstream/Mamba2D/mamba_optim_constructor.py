import torch
import torch.nn as nn
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS

# Import the registry to build the wrapper
from mmengine.registry import OPTIM_WRAPPERS

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MambaOptimConstructor(DefaultOptimWrapperConstructor):
    def __call__(self, model):
        # 1. Get Configs
        if hasattr(self, 'optimizer_cfg'):
            optim_cfg = self.optimizer_cfg
        else:
            optim_cfg = self.optimizer

        base_lr = optim_cfg.get('lr', 0.0001)
        base_wd = optim_cfg.get('weight_decay', 0.05)

        # 2. Group Parameters
        param_groups = {}
        def add_param_to_group(param, name, group_key, lr, wd):
            if group_key not in param_groups:
                param_groups[group_key] = {'params': [], 'lr': lr, 'weight_decay': wd}
            param_groups[group_key]['params'].append(param)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # --- Logic ---
            is_backbone = 'backbone' in name

            # Smart Detection: 1D params are usually bias/norm/scale
            # This covers LayerNorm.weight, Bias, etc. automatically
            no_decay = (
                (param.ndim == 1)
                or getattr(param, "_no_weight_decay", False)
                or ('bias' in name)
                or ('norm' in name)
                or ('scale' in name)
            )

            # Optional base_lr mult
            target_lr = base_lr * 1.0 if is_backbone else base_lr
            target_wd = 0.0 if no_decay else base_wd

            group_key = f"{'backbone' if is_backbone else 'head'}_{'nodecay' if no_decay else 'decay'}"
            add_param_to_group(param, name, group_key, target_lr, target_wd)

        optim_groups = list(param_groups.values())

        # 3. Build Raw Optimizer
        optimizer_args = {k: v for k, v in optim_cfg.items() if k != 'type'}
        optimizer = torch.optim.AdamW(optim_groups, **optimizer_args)

        # Use the wrapper config (clip_grad, type, etc.) from self.optim_wrapper_cfg
        # which acts as the template, and inject our built optimizer.
        return OPTIM_WRAPPERS.build(
            self.optim_wrapper_cfg,
            default_args=dict(optimizer=optimizer)
        )
