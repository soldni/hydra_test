from utils.config import HydraRegistry, FlexibleConfig
from typing import List



@HydraRegistry(name='adam', group='model.optimizer', auto_dc=True)
class AdamConfig(FlexibleConfig):
    _target_: str = 'torch.optim.Adam'
    lr: float = 1e-5


@HydraRegistry(name='adamw', group='model.optimizer', auto_dc=True)
class AdamWConfig(FlexibleConfig):
    _target_: str = 'torch.optim.AdamW'
    lr: float = 1e-5
    weight_decay: float = 0.001


@HydraRegistry(name='fused_adam', group='model.optimizer', auto_dc=True)
class DeepSpeedFusedAdamConfig(FlexibleConfig):
    _target_: str = 'deepspeed.ops.adam.FusedAdam'
    lr: float = 1e-5
    betas: List[float] = HydraRegistry.field(0.8, 0.999)
    eps: float = 1e-8
    weight_decay: float = 3e-7
    adam_w_mode: bool = False


@HydraRegistry(name='deepspeed_fused_adam', group='model.optimizer', auto_dc=True)
class DeepSpeedFusedAdamConfig(FlexibleConfig):
    _target_: str = 'deepspeed.ops.adam.FusedAdam'
    lr: float = 1e-5
    betas: List[float] = HydraRegistry.field(0.8, 0.999)
    eps: float = 1e-8
    weight_decay: float = 3e-7


@HydraRegistry(name='deepspeed_cpu_adam', group='model.optimizer', auto_dc=True)
class DeepSpeedCPUAdamConfig(FlexibleConfig):
    _target_: str = 'deepspeed.ops.adam.DeepSpeedCPUAdam'
    lr: float = 1e-5
    betas: List[float] = HydraRegistry.field(0.8, 0.999)
    eps: float = 1e-8
    weight_decay: float = 3e-7


@HydraRegistry(name='deepspeed_one_bit_adam', group='model.optimizer', auto_dc=True)
class DeepSpeedOnebitAdamConfig(FlexibleConfig):
    _target_: str = 'deepspeed.ops.adam.OnebitAdam'
    lr: float = 1e-5
    betas: List[float] = HydraRegistry.field(0.8, 0.999)
    eps: float = 1e-8
    weight_decay: float = 3e-7
    freeze_step: int = 400
    cuda_aware: bool = True
