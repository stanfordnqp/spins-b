from typing import Callable, Dict, NamedTuple

import collections


class SchemaEntry(
        NamedTuple("SchemaEntry", [("schema", type), ("creator", Callable),
                                   ("meta", Dict)])):
    """Represents a single entry in the context map.

    Attributes:
        schema: Schema used for the node.
        creator: Callable used to create the node. Usually, this is simply
            a class itself (and the constructor will be called).
    """


class SchemaRegistry:
    def __init__(self) -> None:
        self._node_map = collections.OrderedDict()

    def register(self,
                 name: str,
                 schema,
                 creator: Callable,
                 metadata: Dict = None):
        if name in self._node_map:
            raise ValueError(
                "Node with name {} already registered.".format(name))
        self._node_map[name] = SchemaEntry(schema, creator, metadata)

    def get(self, node_name: str):
        if node_name in self._node_map:
            return self._node_map[node_name]
        return None

    def get_map(self) -> collections.OrderedDict:
        return self._node_map


class SchemaRegistryStack:
    def __init__(self) -> None:
        self._stack = [SchemaRegistry()]

    def push(self, registry: SchemaRegistry = None):
        if not registry:
            registry = SchemaRegistry()
        self._stack.append(context)

    def pop(self):
        return self._stack.pop()

    def register(self, *args, **kwargs):
        self._stack[-1].register(*args, **kwargs)

    def get(self, node_name: str):
        for context in reversed(self._stack):
            node = context.get(node_name)
            if node:
                return node
        raise ValueError("Cannot find node {}".format(node_name))

    def get_map(self) -> collections.OrderedDict:
        node_map = collections.OrderedDict()
        for context in self._stack:
            node_map.update(context.get_map())
        return node_map
