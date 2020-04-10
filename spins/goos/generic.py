"""Defines nodes for generic operations."""
from typing import List, Optional, Union

from spins import goos
from spins.goos import flows
from spins.goos import optplan


def cast(node: goos.ProblemGraphNode,
         cls_type,
         name: Optional[str] = None) -> goos.ProblemGraphNode:
    """Casts a problem graph node into another type.

    The problem graph node can be cast into any arbitrary type. No checks are
    performed whatsoever, so the resulting graph may throw an error during
    execution.

    Casting works by creating a new `CastOp` each type whose superclass is
    determined by `cls_type`. This `CastOp` simply performs the identity
    operation. In order to handle serialization/deserialization, this `CastOp`
    class is not registered with the context. Instead, `build_cast_op` function
    is registered.

    Usage:
        numeric_node = goos.cast(node, goos.Function) + 3

    Args:
        node: Node to cast.
        cls_type: Class type.
        name: Name of the cast node.

    Returns:
        A dummy `CastOp` that has the target type `cls_type` and simply forwards
        the result of the underlying node. It is essentially an identity
        operation.
    """

    class CastOp(cls_type):
        node_type = "goos.cast"
        # We will register a custom function `build_cast_op` instead. We need
        # to do this as the superclass of `CastOp` needs to be parsed from
        # the schema during a load.
        goos_no_register = True
        Schema = CastSchema

        def __init__(self, node: goos.ProblemGraphNode, target_type: str):
            goos.ProblemGraphNode.__init__(self, node)
            self._node = node

        def eval(self, inputs):
            return inputs[0]

        def grad(self, inputs, grad_val):
            return [grad_val]

        def __getattr__(self, name: str):
            """Forwards any function calls to the underlying node."""
            return getattr(self._node, name)

    return CastOp(node, cls_type.node_type, name=name)


def build_cast_op(node: goos.ProblemGraphNode, target_type: str,
                  name: str) -> goos.ProblemGraphNode:
    """Constructs a cast operation from the schema.

    This function is registered with the context in order to perform casting
    operations.

    Args:
        node: The node to cast.
        target_type: The string name of the type to cast into.

    Returns:
        `CastOp` object. See `cast`.
    """
    return cast(node,
                optplan.GLOBAL_CONTEXT_STACK.get_node(target_type).creator,
                name=name)


class CastSchema(optplan.ProblemGraphNodeSchema, optplan.WildcardSchema):
    """Schema for `cast`."""
    type = goos.ModelNameType("goos.cast")
    node = goos.ReferenceType(optplan.ProblemGraphNodeSchema)
    target_type = goos.types.StringType()


optplan.GLOBAL_CONTEXT_STACK.register_node("goos.cast", CastSchema,
                                           build_cast_op)


def rename(node: goos.ProblemGraphNode, name: str) -> goos.ProblemGraphNode:
    """Renames a given node.

    Because the name of a node is fixed upon node creation, this function serves
    as a mechanism to change the name of a node. It does this by creating
    an identity op (by casting a node into the same type) with a new name
    `name`.

    Args:
        node: Node to rename.
        name: New name of the node.

    Returns:
        Node with the same type but with name `name`.
    """
    return cast(node, type(node), name=name)


class LogPrint(goos.Action):
    """Prints text out to the log.

    This is useful for debugging purposes.
    """
    node_type = "goos.log_print"

    def __init__(
        self,
        obj: Union[str, goos.ProblemGraphNode,
                   List[goos.ProblemGraphNode]] = None
    ) -> None:
        super().__init__()

        self._obj = obj

    def run(self, plan: goos.OptimizationPlan) -> None:
        if isinstance(self._obj, str):
            plan.logger.info(self._obj)
            return

        if isinstance(self._obj, goos.Function):
            nodes = [self._obj]
        else:
            nodes = self._obj

        values = plan.eval_nodes(nodes)
        for node, val in zip(nodes, values):
            plan.logger.info("{}: {}".format(node._goos_name, val))


def log_print(*args, **kwargs) -> LogPrint:
    action = LogPrint(*args, **kwargs)
    goos.get_default_plan().add_action(action)
    return action
