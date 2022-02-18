from utils.config import HydraRegistry
from dataclasses import dataclass


@HydraRegistry(name='checkpoint_best_on_val_loss', group='callback')
@dataclass
class CheckpointBestOnValLossConfig:
    _target_: str = 'pytorch_lightning.callbacks.ModelCheckpoint'
    mode: str = 'min'
    monitor: str = 'val_loss'
    verbose: bool = False


@HydraRegistry(name='early_stopping_on_val_loss', group='callback')
@dataclass
class EarlyStoppingOnValLossConfig:
    _target_: str = 'pytorch_lightning.callbacks.EarlyStopping'
    monitor: str = 'val_loss'
    min_delta: float = 0.1
    patience: int = 5
    verbose: bool = False
    mode: str = 'min'
    check_on_train_epoch_end: bool = False
