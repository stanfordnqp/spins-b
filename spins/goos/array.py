"""Contains nodes and flows for handling array flows."""
from typing import List, Union

import dataclasses

from spins import goos
from spins.goos import generic
from spins.goos import flows


class ArrayFlow(flows.Flow):
    """Represents a flow composed of smaller flows.

    `ArrayFlow` is essentially a list of flows.
    """

    @dataclasses.dataclass
    class ConstFlags:
        flow_flags: List[flows.Flow.ConstFlags] = dataclasses.field(
            default_factory=lambda: [])

        def set_all(self, value: bool) -> None:
            for flow in self.flow_flags:
                flow.set_all(value)

        def __bool__(self):
            return all(flow for flow in self.flow_flags)

    @dataclasses.dataclass
    class Grad(flows.Flow.Grad):
        flows_grad: List[flows.Flow.Grad] = dataclasses.field(
            default_factory=lambda: [])

        def __iadd__(self, value):
            for i in range(len(self.flows_grad)):
                # `None + obj == obj` where `obj` is any valid `Grad` object.
                if not self.flows_grad[i]:
                    self.flows_grad[i] = value[i]
                elif value[i]:
                    self.flows_grad[i] += value[i]
            return super().__iadd__(value)

        def __getitem__(self, key: int) -> flows.Flow:
            return self.flows_grad[key]

        def __setitem__(self, key: int, flow: flows.Flow) -> None:
            self.flows_grad[key] = flow

    def __init__(self, array: List[flows.Flow]) -> None:
        """Creates a new array flow.

        Args:
            array: The list of flows. The flows need not be the same type.
        """
        super().__init__()
        self._flows = array

    def __iadd__(self, value: List[flows.Flow]) -> "ArrayFlow":
        for flow, inc_flow in zip(self._flows, value):
            # We allow `inc_flow` to be `None`, in which case we skip the
            # addition. This is done so that one can add a flow to one
            # particular part of the array flow without worrying about all
            # the types in the array flow (see `IndexOp.grad`).
            if inc_flow:
                flow += inc_flow
        return super().__iadd__(value)

    def __getitem__(self, key: int) -> flows.Flow:
        """Retrieves a flow.

        Args:
            key: Index of the flow.

        Returns:
            Flow at index `key`.
        """
        return self._flows[key]

    def __setitem__(self, key: int, value: flows.Flow) -> None:
        """Sets a flow in the array flow.

        Args:
            key: Index of flow to set.
        """
        self._flows[key] = value

    def __len__(self) -> int:
        return len(self._flows)

    def __eq__(self, value) -> bool:
        return self._flows == value._flows

    def __repr__(self):
        return "ArrayFlow({})".format(self._flows)


class ArrayFlowOpMixin:
    """Indicates that an operation that produces an array flow.

    Such an op is called an array flow op. An array flow op has a list
    of output nodes, each of which corresponds to flow in the array flow.
    """

    def __init__(self,
                 *args,
                 flow_types: List = None,
                 flow_names: List = None,
                 prepend_parent_name: bool = True,
                 **kwargs) -> None:
        """Creates a new array flow op.

        Args:
            flow_types: List of types corresponding to the output of the
                array flow. This is used to cast the array flow
                into the correct type when indexed.
            flow_names: Names to give the casted flows. A name can be set
                to `None` to indicate use automatic naming. If the list is
                `None`, then all flow outputs are automatically named.
            prepend_parent_name: If `True`, the flow names produced by indexing
                are prepended with the name of this node. Note that this only
                applies when `flow_names` is not `None`.

        Raises:
            ValueError: If there are duplicate flow names.
        """
        super().__init__(*args, **kwargs)
        self._flow_types = flow_types
        if not flow_names:
            flow_names = [None] * len(flow_types)

        # Check for duplicate flow names here.
        names = set()
        for name in flow_names:
            if not name:
                continue
            if name in names:
                raise ValueError(
                    "Duplicate flow name found, got {}".format(name))
            names.add(name)

        self._flow_names = flow_names
        self._prepend_parent_name = prepend_parent_name

        # Cache the cast objects because we can only have a single cast object
        # with a name given by `flow_names`. If we don't cache, multiple
        # indexing options will lead to multiple cast objects, each with the
        # same name.
        self._cast_objs = [None] * len(flow_types)

    def __getitem__(self, ind: Union[int, str]) -> goos.ProblemGraphNode:
        """Retrieves the node of the array flow op.

        Args:
            ind: If integer, index of node to retrieve. If string, name of the
                node to retrieve. The index is inferred from the flow names.

        Returns:
            Node at index `ind` of the array flow op. The flow op is cast
            into the correct output type in accordance to `flow_types` defined
            during class construction.
        """
        if not isinstance(ind, int):
            # Let the user pass in an `ind` that has a string name and convert
            # it into a string.
            ind = str(ind)
            ind = self._flow_names.index(ind)

        if self._cast_objs[ind] is None:
            flow_name = self._flow_names[ind]
            # We prepend now instead of the constructor because `_goos_name`
            # is not set until after object construction.
            if self._prepend_parent_name and flow_name:
                flow_name = self._goos_name + "." + flow_name
            self._cast_objs[ind] = generic.cast(IndexOp(self, ind),
                                                self._flow_types[ind],
                                                name=flow_name)
        return self._cast_objs[ind]


class IndexOp(goos.ProblemGraphNode):
    """Retrieves a particular flow from an array flow."""
    node_type = "goos.index_op"

    def __init__(self, node: goos.ProblemGraphNode, index: int) -> None:
        """Creates an index op that retrieves the `index`th node.

        Args:
            node: Node that produces an array flow.
            index: Index of the output node to extract.
        """
        super().__init__(node)
        self._ind = index

    def eval(self, inputs: List[ArrayFlow]) -> flows.Flow:
        return inputs[0][self._ind]

    def grad(self, inputs: List[ArrayFlow],
             grad_val: flows.Flow) -> List[ArrayFlow.Grad]:
        grad_flow = [None] * len(inputs[0])
        grad_flow[self._ind] = grad_val
        return [ArrayFlow.Grad(grad_flow)]
