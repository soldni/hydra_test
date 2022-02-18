import inspect
from dataclasses import Field, dataclass, field, is_dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Union

import hydra
from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf


OmegaConfTypes = Union[int, str, float, ListConfig, DictConfig]
DictConfTypes = Union[int, str, float, List['DictConfTypes'], Dict[str, 'DictConfTypes']]


class HydraRegistry:
    cs = ConfigStore.instance()

    def __new__(cls, name: str = None, group: str = None, auto_dc: bool = False) -> Callable:
        def add_node_to_registry(node, name, group, auto_dc):
            node = dataclass(node) if dataclass else node
            name = node.__name__ if name is None else name
            cls.cs.store(group=group, name=name, node=node)
            return node
        return partial(add_node_to_registry,
                       name=name,
                       group=group,
                       auto_dc=auto_dc)

    @staticmethod
    def field(*args: List[Any], **kwargs: Dict[str, Any]) -> Field:
        if len(args) > 0 and len(kwargs) > 0:
            error_msg = ('When using HydraRegistry.field, provide either args'
                        f'or kwargs, but not both (len args: {len(args)}; '
                        f'len kwargs: {len(kwargs)})')
            raise ValueError(error_msg)
        if len(args) > 0:
            return field(default_factory=lambda: args)
        else:
            return field(default_factory=lambda: kwargs)

    @staticmethod
    def property(v: Any) -> Field:
        return field(default_factory=lambda: v, init=False)

    @staticmethod
    def missing() -> str:
        return MISSING

    @classmethod
    def omega_conf_to_dict(cls, cfg: OmegaConfTypes) -> DictConfTypes:
        """Turns an OmegaConf configuration into a nested list/dict"""
        if isinstance(cfg, ListConfig):
            return [cls.omega_conf_to_dict(c) for c in cfg]
        elif isinstance(cfg, DictConfig):
            return {k: cls.omega_conf_to_dict(c) for k, c in cfg.items()}
        else:
            return cfg


class BasePrimaryConfig:
    @classmethod
    def use(cls, main_func):
        # lets find out what's the current path for this config;
        # after all, we are going to usee it as primary, so the
        # path will be used to tell hydra where to look
        module_name = inspect.getmodule(cls).__name__

        # little bit of logic here: we look for the number of
        # '.' in the name to figure out whether we are in a submodule
        # or not; if we are, we set the path to the path of the
        # submodule; if not, we set it to None
        if '.' in module_name:
            config_path, config_name = module_name.rsplit('.', 1)
        else:
            config_path, config_name = None, module_name

        # return the decorated function by hydra
        return hydra.main(config_name=config_name, config_path=config_path)(main_func)


class FlexibleConfig:
    @classmethod
    def flexible(cls):
        assert is_dataclass(cls), f"{cls.__name__} must be a dataclass"

        cfg = OmegaConf.structured(cls)
        OmegaConf.set_struct(cfg, False)
        return cfg
