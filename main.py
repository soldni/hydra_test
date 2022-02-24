import hydra
from utils.config import HydraRegistry
from config.default import TrainConfig


@TrainConfig.use
def my_app(cfg: TrainConfig) -> None:
    print(HydraRegistry.omega_conf_to_yaml(cfg))



if __name__ == "__main__":
    my_app()