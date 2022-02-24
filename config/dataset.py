from typing import Optional
from utils.config import HydraRegistry, HydraDefaults


@HydraRegistry(name='default', group='data.loader')
class BaseLoader:
  _target_: str = 'datasets.load_dataset'
  path: str = HydraRegistry.missing()
  task: Optional[str] = None
  split: str = HydraRegistry.missing()


@HydraRegistry(name='generation', group='data.mapper')
class GenerationMapper:
  _target_: str = 'data.mappers.Seq2SeqFeaturesMapper'
  source_column_name: str = HydraRegistry.missing()
  target_column_name: str = HydraRegistry.missing()
  tokenizer: HydraRegistry.config_type() = HydraRegistry.missing()
  target_max_length: Optional[int] = None
  truncate_source: bool = True


@HydraRegistry(name='default', group='data.dataset')
class BaseDataset:
    loader: BaseLoader = BaseLoader()
    mapper: GenerationMapper = HydraRegistry.missing()


@HydraRegistry(name='qasper', group='data.dataset')
class QasperDataset(BaseDataset):
    defaults: HydraDefaults.type_() = HydraDefaults(
        HydraDefaults.self_(),
        HydraDefaults.default_(path='/data.mapper', key='mapper', value='generation'),
    )
    loader: BaseLoader = BaseLoader(path='qasper')
    mapper: GenerationMapper = GenerationMapper(
        source_column_name='abstract',
        target_column_name='qas.question',
        tokenizer='${model.tokenizer}'
    )
