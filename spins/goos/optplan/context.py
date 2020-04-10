from typing import Callable, Dict, NamedTuple


class ContextEntry(
        NamedTuple("ContextEntry", [("schema", type), ("creator", Callable)])):
    """Represents a single entry in the context map.

    Attributes:
        schema: Schema used for the node.
        creator: Callable used to create the node. Usually, this is simply
            a class itself (and the constructor will be called).
    """


class Context:

    def __init__(self) -> None:
        self._node_map = {}

    def register_node(self, node_name: str, schema, creator: Callable):
        if node_name in self._node_map:
            raise ValueError(
                "Node with name {} already registered.".format(node_name))
        self._node_map[node_name] = ContextEntry(schema, creator)

    def get_node(self, node_name: str):
        if node_name in self._node_map:
            return self._node_map[node_name]
        return None

    def get_node_map(self) -> Dict:
        return self._node_map


class ContextStack:

    def __init__(self) -> None:
        self._stack = [Context()]

    def push(self, context: Context = None):
        if not context:
            context = Context()
        self._stack.append(context)

    def pop(self):
        return self._stack.pop()

    def register_node(self, node_name: str, schema, creator: Callable):
        self._stack[-1].register_node(node_name, schema, creator)

    def get_node(self, node_name: str):
        for context in reversed(self._stack):
            node = context.get_node(node_name)
            if node:
                return node
        raise ValueError("Cannot find node {}".format(node_name))

    def get_node_map(self) -> Dict:
        node_map = {}
        for context in self._stack:
            node_map.update(context.get_node_map())
        return node_map
