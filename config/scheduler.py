from utils.config import HydraRegistry, FlexibleConfig


@HydraRegistry(name='constant_schedule_with_warmup', group='model.scheduler')
class ConstantScheduleWithWarmup(FlexibleConfig):
    _target_: str = 'transformers.get_constant_schedule_with_warmup'
    # float values determines percentage of training steps to use as warmup
    num_warmup_steps: int = 200


@HydraRegistry(name='constant_schedule', group='model.scheduler')
class ConstantSchedule(FlexibleConfig):
    _target_: str = 'transformers.get_constant_schedule'


@HydraRegistry(name='cosine_schedule_with_warmup', group='model.scheduler')
class CosineScheduleWithWarmup(FlexibleConfig):
    _target_: str = 'transformers.get_cosine_schedule_with_warmup'
    # -1 specifies to infer number of training steps
    num_training_steps: int = -1
    # float values determines percentage of training steps to use as warmup
    num_warmup_steps: int = 200
    num_cycles: float = 0.5


@HydraRegistry(name='cosine_with_hard_restarts_schedule_with_warmup', group='model.scheduler')
class CosineWithHardRestartsScheduleWithWarmup(FlexibleConfig):
    _target_: str = 'transformers.get_cosine_with_hard_restarts_schedule_with_warmup'
    # -1 specifies to infer number of training steps
    num_training_steps: int = -1
    # float values determines percentage of training steps to use as warmup
    num_warmup_steps: int = 200
    num_cycles: int = 1


@HydraRegistry(name='linear_schedule_with_warmup', group='model.scheduler')
class LinearScheduleWithWarmup(FlexibleConfig):
    _target_: str = 'transformers.get_linear_schedule_with_warmup'
    # -1 specifies to infer number of training steps
    num_training_steps: int = -1
    # float values determines percentage of training steps to use as warmup
    num_warmup_steps: int = 200


@HydraRegistry(name='polynomial_decay_schedule_with_warmup', group='model.scheduler')
class PolynomialDecayScheduleWithWarmup(FlexibleConfig):
    _target_: str = 'transformers.get_polynomial_decay_schedule_with_warmup'
    # -1 specifies to infer number of training steps
    num_training_steps: int = -1
    # float values determines percentage of training steps to use as warmup
    num_warmup_steps: int = 200
    lr_end: float = 1e-7
    power: float = 1.0
