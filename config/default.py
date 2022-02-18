from utils.config import HydraRegistry, BasePrimaryConfig
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict
from config.checkpoint import *
from config.huggingface import *
from config.optimizer import *

# @HydraRegistry(name='mysql', group='db')
# @dataclass
# class MySQLConfig:
#     host: str = 'this_host'
#     port: int = 420


# @HydraRegistry(name='postgresql', group='db')
# @dataclass
# class PostGreSQLConfig:
#     host: str = 'that_host'
#     port: int = 69

# @dataclass
# class HydraConfig(HydraConf):
#     run: RunDir = RunDir(dir='${env.root_dir}/${env.run_name}/logs')
#     output_subdir: Optional[str] = None
#     searchpath = None

@dataclass
class EnvConfig:
    run_name: str = 'run_${now:%Y-%m-%d}_${now:%H-%M-%S}'
    root_dir: str = '${oc.env:HOME}/plruns'
    s3_prefix: Optional[str]= None
    seed: int = 5663


@HydraRegistry(name='default', group='trainer', auto_dc=True)
class TrainerConfig:
    _target_: str = 'trainer.TrainerWithHuggingFaceSaver'
    accelerator: str = 'auto'
    devices: int = 1
    max_epochs: int = 50
    precision: int = 32
    strategy: str = 'ddp'


@HydraRegistry(name='tensorboard', group='logger', auto_dc=True)
class TensorboardConfig:
    _target_: str = 'pytorch_lightning.loggers.TensorBoardLogger'
    log_graph: bool = True


@HydraRegistry(name='generation', group='model', auto_dc=True)
class GenerationModelConfig:
    _target_: str = 'model.QQGSeq2SeqModule'
    tokenizer: AutoTokenizerConfig = AutoTokenizerConfig()
    transformer: AutoSeq2SeqTransformerConfig = AutoSeq2SeqTransformerConfig.flexible()
    metrics: Optional[List[Any]] = None
    optimizer: Optional[Any] = None
    scheduler: Optional[Any] = None
    eval_accumulate_multiple_refs: bool = False


@HydraRegistry(name='default', auto_dc=True)
class Config(BasePrimaryConfig):
    defaults: List[Any] = HydraRegistry.field(
        '_self_',
        {'model': 'generation'},
        {'model.optimizer': 'adamw'}
    )
    backbone: str = HydraRegistry.missing()
    env: EnvConfig = EnvConfig()

    logger: TensorboardConfig = TensorboardConfig()
    early_stopping: EarlyStoppingOnValLossConfig = EarlyStoppingOnValLossConfig()
    checkpoint: CheckpointBestOnValLossConfig = CheckpointBestOnValLossConfig()
    trainer: TrainerConfig = TrainerConfig()
    # model: GenerationModelConfig = GenerationModelConfig()
    model: GenerationModelConfig = HydraRegistry.missing()

    # configuring hydra's option has to be done through
    # a bit of a hack. Inspiration comes mostly from this
    # https://github.com/facebookresearch/hydra/issues/1903
    # hack is necessary because subclassing HydraConf
    # creates issues with searchpath
    hydra: Dict[str, Any] =  HydraRegistry.field(
        run={"dir": "${env.root_dir}/${env.run_name}/logs"},
        output_subdir=None
    )
