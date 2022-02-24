import hydra
from utils.config import HydraRegistry
from config.default import EvalConfig


@EvalConfig.use
def my_app(cfg: EvalConfig) -> None:
    print(HydraRegistry.omega_conf_to_yaml(cfg))



if __name__ == "__main__":
    my_app()