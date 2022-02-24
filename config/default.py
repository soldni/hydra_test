from config.checkpoint import CheckpointBestOnValLossConfig, EarlyStoppingOnValLossConfig
from config.dataset import BaseDataset
from config.huggingface import AutoSeq2SeqTransformerConfig, AutoTokenizerConfig
from utils.config import HydraRegistry, BasePrimaryConfig, HydraDefaults
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvConfig:
    run_name: str = 'run_${now:%Y-%m-%d}_${now:%H-%M-%S}'
    root_dir: str = '${oc.env:HOME}/plruns'
    s3_prefix: Optional[str]= None
    seed: int = 5663


@HydraRegistry(name='default', group='trainer')
class TrainerConfig:
    _target_: str = 'trainer.TrainerWithHuggingFaceSaver'
    accelerator: str = 'auto'
    devices: int = 1
    max_epochs: int = 50
    precision: int = 32
    strategy: str = 'ddp'


@HydraRegistry(name='tensorboard', group='logger')
class TensorboardConfig:
    _target_: str = 'pytorch_lightning.loggers.TensorBoardLogger'
    log_graph: bool = True


@HydraRegistry(name='generation', group='model')
class GenerationModelConfig:
    _target_: str = 'model.QQGSeq2SeqModule'
    tokenizer: AutoTokenizerConfig = AutoTokenizerConfig()
    transformer: AutoSeq2SeqTransformerConfig = AutoSeq2SeqTransformerConfig.flexible()
    metrics: HydraRegistry.config_type(optional=True) = HydraRegistry.empty_dict()
    optimizer: HydraRegistry.config_type(optional=True) = None
    scheduler: HydraRegistry.config_type(optional=True) = None
    eval_accumulate_multiple_refs: bool = False


@HydraRegistry(name='generation', group='data')
class GenerationDataConfig:
    _target_: str = 'data.modules.QQGDataModule'
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    train_split_config: Optional[BaseDataset] = None
    valid_split_config: Optional[BaseDataset] = None
    test_split_config: Optional[BaseDataset] = None


@HydraRegistry(name='train_default')
class TrainConfig(BasePrimaryConfig):
    defaults: HydraDefaults.type_() = HydraDefaults(
        HydraDefaults.self_(),
        HydraDefaults.default_(key='model', value='generation'),
        HydraDefaults.default_(key='data', value='generation')
    )
    backbone: str = HydraRegistry.missing()
    env: EnvConfig = EnvConfig()

    logger: TensorboardConfig = TensorboardConfig()
    early_stopping: EarlyStoppingOnValLossConfig = EarlyStoppingOnValLossConfig()
    checkpoint: CheckpointBestOnValLossConfig = CheckpointBestOnValLossConfig()
    trainer: TrainerConfig = TrainerConfig()
    model: GenerationModelConfig = HydraRegistry.missing()
    data: GenerationDataConfig = HydraRegistry.missing()

    # configuring hydra's option has to be done through
    # a bit of a hack. Inspiration comes mostly from this
    # https://github.com/facebookresearch/hydra/issues/1903
    # hack is necessary because subclassing HydraConf
    # creates issues with searchpath
    hydra: HydraRegistry.config_type() =  HydraRegistry.field(
        run={"dir": "${env.root_dir}/${env.run_name}/logs"},
        output_subdir=None
    )
