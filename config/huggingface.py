from utils.config import HydraRegistry, FlexibleConfig
from dataclasses import dataclass

@HydraRegistry(name='default', group='model.tokenizer')
@dataclass
class AutoTokenizerConfig(FlexibleConfig):
    _target_: str = 'transformers.AutoTokenizer.from_pretrained'
    pretrained_model_name_or_path: str = '${backbone}'


@HydraRegistry(name='seq2seq', group='model.transformer')
@dataclass
class AutoSeq2SeqTransformerConfig(FlexibleConfig):
    _target_: str = 'transformers.AutoModelForSeq2SeqLM.from_pretrained'
    pretrained_model_name_or_path: str  = '${backbone}'
