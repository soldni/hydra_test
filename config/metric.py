from utils.config import HydraRegistry

@HydraRegistry(name='rouge', group='metric.text')
class RougeConfig:
    _target_: str = 'torchmetrics.text.rouge.ROUGEScore'
