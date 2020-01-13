# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-04 11:10
# @Author  : yingyuankai@aliyun.com
# @File    : registry.py


from collections import defaultdict
from typing import TypeVar, Type, Dict, List

T = TypeVar('T')


class Registry:
    r"""Class for registry object which acts as central source of truth
    for Aispace
    """
    state = dict()
    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)
    default_implementation: str = None

    @classmethod
    def register(cls: Type[T], name: str):
        registry = Registry._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]):
            # Add to registry, raise an error if key has already been used.
            if name in registry:
                message = "Cannot register %s as %s; name already in use for %s" % (
                    name, cls.__name__, registry[name].__name__)
                raise ValueError(message)
            registry[name] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    def putup_tool(cls, name, obj):
        r"""Register an item to registry with key 'name'
        Args:
            name: Key with which the item will be registered.
        Usage::
            from aispace.utils.registry import registry
            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.state

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        if name not in Registry._registry[cls]:
            raise ValueError("%s is not a registered name for %s" % (name, cls.__name__))
        return Registry._registry[cls].get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        """List default first if it exists"""
        keys = list(Registry._registry[cls].keys())
        default = cls.default_implementation

        if default is None:
            return keys
        elif default not in keys:
            message = "Default implementation %s is not registered" % default
            raise ValueError(message)
        else:
            return [default] + [k for k in keys if k != default]

    @classmethod
    def pickup_tool(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retreived.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for Aispace's
                               internal operations. Default: False
        Usage::

            from Aispace.common.registry import registry

            config = registry.get("config")
        """
        original_name = name
        name = name.split(".")
        state = cls.state
        for subname in name:
            value = state.get(subname, default)
            if value is default:
                break

        if (
            "logger" in cls.state
            and value == default
            and no_warning is False
        ):
            cls.state["logger"].logging(
                f"Key {original_name} is not present in registry, returning default value "
                f"of {default}"
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from Aispace.utils.registry import registry

            config = registry.unregister("config")
        """
        return cls.state.pop(name, None)


registry = Registry()
