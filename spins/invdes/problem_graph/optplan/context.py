"""This module implements optimization plan contexts.

Contexts
========
Contexts are used to manage namespaces and model registration for optimization
plans. Model registration is the mechanism by which new models are added to
the schema. Specifically, the context holds a mapping from the string type of
a node in the optplan to the corresponding schematics model. This enables
extending the spins schema without direct modification of the spins code.

Context Stack
=============
The contexts are maintained in a context stack, where contexts higher up on the
stack take precedence over contexts at the bottom. For example, when a model for
a particular node type is queried, the context at the top of the stack is
queried. If the top context does not know about the type, the next context
in the stack is queried, and so forth. This context stack mechanism allows:
    1) Clean separation between the default spins schema and any user extensions
       as they can live in separate contexts.
    2) Enable the user to override the schema for the default spins schema.

Model Registration
==================
A model is registered with a given node type and a node metatype. The node
metatype and node type combined are a unique string tuple that identifies the
model. The node metatype can be thought of as a namespace for node types.
For example, spins uses a separate node metatype for optplan nodes and
transformation types.
"""
from typing import Callable, Dict, Optional

from schematics import models

CreatorFunction = Callable[
    ["optplan.ProblemGraphNode", "spins.invdes.Workspace"], object]


class OptplanContext:
    """Wraps the context used to process optimization plans."""

    def __init__(self) -> None:
        """Creates a new context."""
        # This is of type Dict[str, Dict[str, (models.Model, Callable)]].
        self._optplan_node_map = {}

    def get_node_model(self, node_meta_type: str,
                       node_type: str) -> Optional[models.Model]:
        """Retrieves the schematics model corresponding to a given type.

        Args:
            node_meta_type: Node metatype.
            node_type: Node type.

        Returns:
            Model corresponding to node metatype and type. Returns `None`
            if no such model can be found.
        """
        if node_meta_type not in self._optplan_node_map:
            return None

        if node_type not in self._optplan_node_map[node_meta_type]:
            return None
        return self._optplan_node_map[node_meta_type][node_type][0]

    def get_node_creator(self, node_meta_type: str,
                         node_type: str) -> Optional[CreatorFunction]:
        """Retrieves the creator function corresponding to a given type.

        Args:
            node_meta_type: Node metatype.
            node_type: Node type.

        Returns:
            Creator function for the given node. Returns `None`
            if no such model can be found.
        """
        if node_meta_type not in self._optplan_node_map:
            return None

        if node_type not in self._optplan_node_map[node_meta_type]:
            return None
        return self._optplan_node_map[node_meta_type][node_type][1]

    def get_node_model_dict(self,
                            node_meta_type: str) -> Dict[str, models.Model]:
        """Returns a mapping from node type to model.

        Args:
            node_meta_type: Node metatype to use.

        Returns:
            A dictionary for the given node metatype that maps node types
            to models. If the node metatype is not used in this context, an
            empty dictionary is returned.
        """
        if node_meta_type not in self._optplan_node_map:
            return {}

        return {
            node_type: vals[0] for node_type, vals in self._optplan_node_map[
                node_meta_type].items()
        }

    def register_node_type(self, node_meta_type: str, node_type: str,
                           model: models.Model, fun: CreatorFunction) -> None:
        """Registers a optplan node.

        Args:
            node_meta_type: Metatype of the node.
            node_type: Node type.
            model: Schematics model used to (de)serialize the node.
            fun: Callable used to instantiate the node from the model.

        Raises:
            ValueError: If a model with the same node metatype and node type
                was already registered for this context.
        """
        node_map = self._optplan_node_map.get(node_meta_type, {})
        if node_type in node_map:
            raise ValueError(
                "Node type '{}' with metatype '{}' was registered twice.".
                format(node_type, node_meta_type))
        node_map[node_type] = (model, fun)
        self._optplan_node_map[node_meta_type] = node_map


class OptplanContextStack:
    """Maintains a stack of optimization plan contexts.

    The context stack essentially behaves as a context that is built from
    merging together all the contexts in the stack, where contexts on top of the
    stack have higher precedence over contexts on the bottom.

    Most spins code should rely on the context stack rather than the context
    directly. In particular, `get_node_model` and `get_node_model_dict` should
    be called for `OptplanContextStack` rather than `OptplanContext` when
    searching for the model.
    """

    def __init__(self) -> None:
        """Creates a new context stack."""
        # List of contexts. The top of the stack corresponds to the end of the
        # list.
        self._stack = []

    def push(self, context: OptplanContext) -> None:
        """Pushes a context onto the top of the stack.

        Args:
            context: Context to push.
        """
        self._stack.append(context)

    def pop(self) -> Optional[OptplanContext]:
        """Pops a context from the top of the stack.

        Returns:
            The context at the top, or `None` if stack is empty.
        """
        if self._stack:
            return self._stack.pop()
        return None

    def peek(self) -> Optional[OptplanContext]:
        """Returns the context from the top of the stack without popping.

        Returns:
            The context at the top, or `None` is stack is emptpy.
        """
        if self._stack:
            return self._stack[-1]
        return None

    def get_node_model(self, node_meta_type: str,
                       node_type: str) -> Optional[models.Model]:
        """Retrieves model associated with a node.

        This function iteratively searches for a node model from the top of
        the context stack to the bottom.

        Args:
            node_meta_type: Node metatype.
            node_type: Node type.

        Returns:
            Node model associated with the given node. Returns `None` if no
            model found.
        """
        for context in reversed(self._stack):
            model = context.get_node_model(node_meta_type, node_type)
            if model:
                return model

        return None

    def get_node_creator(self, node_meta_type: str,
                         node_type: str) -> Optional[CreatorFunction]:
        """Retrieves creator function for the given node.

        This function iteratively searches for a node model from the top of
        the context stack to the bottom.

        Args:
            node_meta_type: Node metatype.
            node_type: Node type.

        Returns:
            Creator function for the given node. Returns `None` if no
            model found.
        """
        for context in reversed(self._stack):
            model = context.get_node_creator(node_meta_type, node_type)
            if model:
                return model

        return None

    def get_node_model_dict(self,
                            node_meta_type: str) -> Dict[str, models.Model]:
        """Retrieves a mapping from node type to corresponding model.

        Note this returns a mapping for a given a metatype.

        Args:
            node_meta_type: Node metatype.

        Returns:
            Dictionary.
        """
        model_dict = {}
        for context in self._stack:
            model_dict.update(context.get_node_model_dict(node_meta_type))

        return model_dict
