import hydra
from utils.config import HydraRegistry
from config.default import Config


@Config.use
def my_app(cfg: Config) -> None:
    print(HydraRegistry.omega_conf_to_dict(cfg))



if __name__ == "__main__":
    my_app()