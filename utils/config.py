from collections import namedtuple
import inspect
from dataclasses import Field, dataclass, field, is_dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Union, Optional, Type

import yaml

import hydra
from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf


OmegaConfTypes = Union[int, str, float, ListConfig, DictConfig]
DictConfTypes = Union[int, str, float, List['DictConfTypes'], Dict[str, 'DictConfTypes']]
MissingType = type(MISSING)


class HydraDefaults:
    OverideType = Union[str, Dict[str, Any]]

    @staticmethod
    def type_() -> Type:
        return List[Any]

    def __new__(cls, *args: OverideType) -> Field:
        return HydraRegistry.field(*args)

    def default_(key: str, value: str, path: Optional[str] = None, is_override: bool = False) -> OverideType:
        if path:
            key = f'{path}@{key}'
        if is_override:
            key = f'override {key}'
        return {key: value}


    @staticmethod
    def self_() -> OverideType:
        return '_self_'


class HydraRegistry:
    cs = ConfigStore.instance()
    NodeProperties = namedtuple("NodeProperties", ['name', 'group'])

    def __new__(cls, name: str = None, group: str = None) -> Callable:
        def add_node_to_registry(node, name, group):
            node = node if is_dataclass(node) else dataclass(node)
            name = node.__name__ if name is None else name

            node._hydra_ = cls.NodeProperties(name=name, group=group)
            cls.cs.store(group=group, name=name, node=node)
            return node
        return partial(add_node_to_registry,
                       name=name,
                       group=group)

    @staticmethod
    def field(*args: List[Any], **kwargs: Dict[str, Any]) -> Field:
        if len(args) > 0 and len(kwargs) > 0:
            error_msg = ('When using HydraRegistry.field, provide either args'
                        f'or kwargs, but not both (len args: {len(args)}; '
                        f'len kwargs: {len(kwargs)})')
            raise ValueError(error_msg)
        if len(args) > 0:
            return field(default_factory=lambda: args)
        elif len(kwargs) > 0:
            return field(default_factory=lambda: kwargs)
        else:
            raise ValueError('No args or kwargs!')

    @staticmethod
    def empty_dict() -> Field:
        return field(default_factory=lambda: {})

    @staticmethod
    def property(v: Any) -> Field:
        return field(default_factory=lambda: v, init=False)

    @staticmethod
    def config_type(optional=False) -> Type:
        t = Dict[str, Any]
        if optional:
            t = Optional[t]
        return t

    @staticmethod
    def missing() -> MissingType:
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

    @classmethod
    def omega_conf_to_yaml(cls, cfg: OmegaConfTypes, *args, **kwargs) -> str:
        dict_conf = cls.omega_conf_to_dict(cfg)
        return yaml.dump(dict_conf, *args, **kwargs)


class BasePrimaryConfig:
    @classmethod
    def use(cls, main_func):
        assert is_dataclass(cls) and hasattr(cls, '_hydra_'), (
            f'Cannot use `{cls.__name__}` as base config; did you '
            'decorated it with `@HydraRegistry()` ?'
        )

        # lets find out what's the current path for this config;
        # after all, we are going to usee it as primary, so the
        # path will be used to tell hydra where to look
        config_path = inspect.getmodule(cls).__name__

        # little bit of logic here: we look for the number of
        # '.' in the name to figure out whether we are in a submodule
        # or not; if we are, we set the path to the path of the
        # submodule; if not, we set it to None
        if '.' in config_path:
            config_path, _ = config_path.rsplit('.', 1)

        # properties `_hydra_` were added during registration
        config_name = cls._hydra_.name

        # return the decorated function by hydra
        return hydra.main(config_name=config_name, config_path=config_path)(main_func)


class FlexibleConfig:
    @classmethod
    def flexible(cls):
        assert is_dataclass(cls), f"{cls.__name__} must be a dataclass"

        cfg = OmegaConf.structured(cls)
        OmegaConf.set_struct(cfg, False)
        return cfg
