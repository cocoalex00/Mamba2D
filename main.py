import os
import time
import warnings
from collections import deque
from typing import Any

import torch
import lightning as L
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import OmegaConf


# Adapted from https://github.com/BioinfoMachineLearning/bio-diffusion and NVIDIA/NeMo
class EMA(Callback):
    """
    Exponential Moving Averaging (EMA) callback.
    Maintains moving averages of trained parameters; evaluates/saves with EMA weights.

    Args:
        decay: EMA decay factor (0–1).
        apply_ema_every_n_steps: Update EMA every N global steps.
        start_step: Start applying EMA from this global step.
        save_ema_weights_in_callback_state: Include EMA weights in checkpoint callback state.
        evaluate_ema_weights_instead: Use EMA weights during validation/test.

    Adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py
    """

    def __init__(
        self,
        decay: float,
        apply_ema_every_n_steps: int = 1,
        start_step: int = 0,
        save_ema_weights_in_callback_state: bool = False,
        evaluate_ema_weights_instead: bool = False,
    ):
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self._ema_model_weights: None | list[torch.Tensor] = None
        self._overflow_buf: None | torch.Tensor = None
        self._cur_step: None | int = None
        self._weights_buffer: None | list[torch.Tensor] = None
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.start_step = start_step
        self.save_ema_weights_in_callback_state = save_ema_weights_in_callback_state
        self.evaluate_ema_weights_instead = evaluate_ema_weights_instead
        self.decay = decay

    def on_train_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        rank_zero_info("Creating EMA weights copy.")
        if self._ema_model_weights is None:
            self._ema_model_weights = [p.detach().clone() for p in pl_module.state_dict().values()]
        self._ema_model_weights = [p.to(pl_module.device) for p in self._ema_model_weights]
        self._overflow_buf = torch.IntTensor([0]).to(pl_module.device)

    def apply_ema(self, pl_module: "L.LightningModule") -> None:
        for orig_weight, ema_weight in zip(list(pl_module.state_dict().values()), self._ema_model_weights):
            if ema_weight.data.dtype != torch.long and orig_weight.data.dtype != torch.long:
                diff = ema_weight.data - orig_weight.data
                diff.mul_(1.0 - self.decay)
                ema_weight.sub_(diff)

    def should_apply_ema(self, step: int) -> bool:
        return step != self._cur_step and step >= self.start_step and step % self.apply_ema_every_n_steps == 0

    def on_train_batch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.should_apply_ema(trainer.global_step):
            self._cur_step = trainer.global_step
            self.apply_ema(pl_module)

    def state_dict(self) -> dict[str, Any]:
        if self.save_ema_weights_in_callback_state:
            return dict(cur_step=self._cur_step, ema_weights=self._ema_model_weights)
        return dict(cur_step=self._cur_step)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._cur_step = state_dict["cur_step"]
        if self._ema_model_weights is None:
            self._ema_model_weights = state_dict.get("ema_weights")

    def on_load_checkpoint(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", checkpoint: dict[str, Any]
    ) -> None:
        checkpoint_callback = trainer.checkpoint_callback
        if trainer.ckpt_path and checkpoint_callback is not None:
            ext = checkpoint_callback.FILE_EXTENSION
            if trainer.ckpt_path.endswith(f"-EMA{ext}"):
                rank_zero_info(
                    "Loading EMA weights. The callback will treat the loaded EMA weights as the main weights"
                    " and create a new EMA copy when training."
                )
                return
            ema_path = trainer.ckpt_path.replace(ext, f"-EMA{ext}")
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device("cpu"))
                self._ema_model_weights = ema_state_dict["state_dict"].values()
                del ema_state_dict
                rank_zero_info("EMA weights loaded successfully. Continuing training with saved EMA weights.")
            else:
                warnings.warn(
                    "Unable to find associated EMA weights when re-loading; training will start with new EMA weights.",
                    UserWarning,
                )

    def replace_model_weights(self, pl_module: "L.LightningModule") -> None:
        self._weights_buffer = [p.detach().clone().to("cpu") for p in pl_module.state_dict().values()]
        new_state_dict = {k: v for k, v in zip(pl_module.state_dict().keys(), self._ema_model_weights)}
        pl_module.load_state_dict(new_state_dict)

    def restore_original_weights(self, pl_module: "L.LightningModule") -> None:
        state_dict = pl_module.state_dict()
        new_state_dict = {k: v for k, v in zip(state_dict.keys(), self._weights_buffer)}
        pl_module.load_state_dict(new_state_dict)
        del self._weights_buffer

    @property
    def ema_initialized(self) -> bool:
        return self._ema_model_weights is not None

    def on_validation_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_validation_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)

    def on_test_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_test_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)


class EMAModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that also saves an EMA copy of weights alongside each checkpoint."""

    def _get_ema_callback(self, trainer: "L.Trainer") -> None | EMA:
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                return callback
        return None

    def _save_checkpoint(self, trainer: "L.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        ema_callback = self._get_ema_callback(trainer)
        if ema_callback is not None:
            ema_callback.replace_model_weights(trainer.lightning_module)
            ema_filepath = self._ema_format_filepath(filepath)
            if self.verbose:
                rank_zero_info(f"Saving EMA weights to separate checkpoint {ema_filepath}")
            super()._save_checkpoint(trainer, ema_filepath)
            ema_callback.restore_original_weights(trainer.lightning_module)

    def _ema_format_filepath(self, filepath: str) -> str:
        return filepath.replace(self.FILE_EXTENSION, f"-EMA{self.FILE_EXTENSION}")

    def _update_best_and_save(
        self, current: torch.Tensor, trainer: "L.Trainer", monitor_candidates: dict[str, torch.Tensor]
    ) -> None:
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"), device=current.device)

        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f}"
                f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )
        self._save_checkpoint(trainer, filepath)

        if del_filepath is not None and filepath != del_filepath:
            self._remove_checkpoint(trainer, del_filepath)
            self._remove_checkpoint(trainer, del_filepath.replace(self.FILE_EXTENSION, f"-EMA{self.FILE_EXTENSION}"))


class ETACallback(Callback):
    """
    Logs training ETA based on a rolling average of full epoch wall times
    (train + validation), so val overhead is automatically included.
    """
    def __init__(self, window: int = 20):
        self.window = window
        self._epoch_times: deque = deque(maxlen=window)
        self._epoch_start: float | None = None
        self._epoch_start_for_val: float | None = None

    def _log_eta(self, trainer, pl_module):
        if not self._epoch_times:
            return
        remaining_epochs = trainer.max_epochs - trainer.current_epoch - 1
        avg = sum(self._epoch_times) / len(self._epoch_times)
        pl_module.log("train/eta_s", remaining_epochs * avg, prog_bar=True, on_step=False, on_epoch=True)

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.perf_counter()
        self._epoch_start_for_val = None

    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_start is not None:
            # Tentatively record train-only time; val epoch will overwrite if it follows.
            self._epoch_times.append(time.perf_counter() - self._epoch_start)
            self._epoch_start_for_val = self._epoch_start
        self._log_eta(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if self._epoch_start_for_val is not None:
            # Overwrite last entry with full train+val time.
            if self._epoch_times:
                self._epoch_times[-1] = time.perf_counter() - self._epoch_start_for_val
            self._epoch_start_for_val = None
        self._log_eta(trainer, pl_module)


class ETAProgressBar(TQDMProgressBar):
    """Reformats the raw eta_s metric into a human-readable string in the pbar."""

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        eta_s = metrics.pop("train/eta_s", None)
        if eta_s is not None:
            secs = int(eta_s)
            h, rem = divmod(secs, 3600)
            m, s = divmod(rem, 60)
            d, h = divmod(h, 24)
            metrics["eta"] = f"{d}d {h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"
        return metrics


class CustomCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        # Resume from checkpoint if it exists, otherwise start fresh
        if hasattr(self.config, "fit"):
            run = self.config.fit
            msg = "Checkpoint not found, starting new run..."
        elif hasattr(self.config, "validate"):
            run = self.config.validate
            msg = "WARNING: Checkpoint not found, validating on untrained model..."
        elif hasattr(self.config, "test"):
            run = self.config.test
            msg = "WARNING: Checkpoint not found, testing on untrained model..."
        elif hasattr(self.config, "predict"):
            run = self.config.predict
            msg = "WARNING: Checkpoint not found, predicting on untrained model..."
        else:
            raise ValueError("Run type not implemented")

        if run.ckpt_path is None or not os.path.exists(run.ckpt_path):
            run.ckpt_path = None
            rank_zero_warn(msg)


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        config_str = self.parser.dump(self.config, skip_none=False)
        rank_zero_info(f"\nCONFIG:\n{config_str}")
        config_dict = OmegaConf.to_container(OmegaConf.create(config_str), resolve=True)
        for logger in trainer.loggers:
            logger.log_hyperparams(config_dict)


def cli_main():
    CustomCLI(save_config_callback=LoggerSaveConfigCallback,
              save_config_kwargs={"overwrite": True},
              parser_kwargs={"parser_mode": "omegaconf"},
              trainer_defaults={"callbacks": [ETACallback(), ETAProgressBar()]})


if __name__ == "__main__":
    cli_main()
