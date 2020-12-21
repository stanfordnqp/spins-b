from typing import List, Iterable, Union

import copy
import numbers

import numpy as np

from spins import goos
from spins.goos import flows


class Function(goos.ProblemGraphNode):
    node_type = "goos.function"

    def __add__(self, obj) -> "Sum":
        if isinstance(obj, Sum):
            return obj.__add__(self)
        if isinstance(obj, Function):
            return Sum([self, obj])
        # Because adding to zero is so common, special case it.
        if np.isscalar(obj) and obj == 0:
            return self
        if isinstance(obj, numbers.Number):
            return Sum([self, Constant(obj)])
        if isinstance(obj, np.ndarray):
            return Sum([self, Constant(obj)])
        raise TypeError("Attempting to add node with type {}".format(type(obj)))

    def __mul__(self, obj) -> "Product":
        if isinstance(obj, Product):
            return obj.__mul__(self)
        if isinstance(obj, Function):
            return Product([self, obj])
        if np.isscalar(obj) and obj == 1:
            return self
        if isinstance(obj, numbers.Number):
            return Product([self, Constant(obj)])
        if isinstance(obj, np.ndarray):
            return Product([self, Constant(obj)])
        raise TypeError("Attempting to multiply node with type {}".format(
            type(obj)))

    def __radd__(self, obj) -> "Sum":
        return self.__add__(obj)

    def __rmul__(self, obj) -> "Product":
        return self.__mul__(obj)

    def __sub__(self, obj) -> "Sum":
        return self + (-obj)

    def __rsub__(self, obj) -> "Sum":
        return -self + obj

    def __neg__(self) -> "Product":
        return -1 * self

    def __truediv__(self, obj) -> "Product":
        return self * obj**-1

    def __rtruediv__(self, obj) -> "Product":
        return self**-1 * obj

    def __pow__(self, obj) -> "Power":
        if isinstance(obj, numbers.Number):
            return Power(self, obj)
        raise TypeError("Attempting to exponenitate node with type {}".format(
            type(obj)))


class Constant(Function):
    node_type = "goos.function.constant"

    def __init__(self, value: np.ndarray):
        super().__init__()
        self._value = np.array(value)

    def eval_const_flags(
        self, inputs: List[flows.NumericFlow.ConstFlags]
    ) -> flows.NumericFlow.ConstFlags:
        return flows.NumericFlow.ConstFlags(True)

    def eval(self, inputs: List[flows.NumericFlow],
             context: goos.EvalContext) -> flows.NumericFlow:
        return flows.NumericFlow(self._value)

    def grad(self, inputs: List[flows.NumericFlow],
             grad_val: flows.NumericFlow.Grad,
             context: goos.EvalContext) -> List[flows.NumericFlow.Grad]:
        return [flows.NumericFlow.Grad(np.zeros_like(self._value))]


class Variable(Function):
    node_type = "goos.function.variable"

    def __init__(
            self,
            init_val: np.ndarray,
            lower_bounds: Union[float, np.ndarray] = None,
            upper_bounds: Union[float, np.ndarray] = None,
            parameter: bool = False,
    ) -> None:
        """Creates a new variable.

        Args:
            init_val: Initial value for the variable.
            lower_bounds: Array with same sahep as `init_val` containing the
                lower bounds for each variable. If there is no lower bound,
                `-np.inf` should be used. If it is a single number, then that
                number is used as a lower bound for all entries.
            upper_bounds: Same as `lower_bounds` but for the upper bounds.
            parameter: If `True`, the variable is treated as a parameter. A
                parameter is always frozen and cannot be thawed.
        """
        super().__init__()
        self._value = np.array(init_val)
        # Make sure that we always have at least a 1D vector.
        if np.ndim(self._value) == 0:
            self._value = np.array([init_val])

        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

        self._is_param = parameter

    def eval(self, inputs: List) -> flows.NumericFlow:
        return flows.NumericFlow(goos.get_default_plan().get_var_value(self))

    def set(self, value):
        if isinstance(value, numbers.Number) or isinstance(value, np.ndarray):
            value = Constant(value)
        goos.get_default_plan().add_action(SetVariable(self, value))

    def freeze(self):
        goos.get_default_plan().add_action(FreezeVariable(self))

    def thaw(self):
        if self._is_param:
            raise ValueError("Cannot thaw a parameter.")
        goos.get_default_plan().add_action(ThawVariable(self))


class SetVariable(goos.Action):
    node_type = "goos.action.set_variable"

    def __init__(self, var: Variable, value: Function) -> None:
        super().__init__(var)
        self._var = var
        self._value = value

        if (not isinstance(value, numbers.Number) and
                not isinstance(value, Function)):
            raise TypeError(
                "`value` must be either numeric or a `goos.Function`,"
                " got {}".format(value))

    def run(self, plan: goos.OptimizationPlan) -> None:
        value = self._value
        if isinstance(value, Function):
            value = plan.eval_node(value).array
        plan.set_var_value(self._var,
                           value,
                           check_frozen=not self._var._is_param)


class FreezeVariable(goos.Action):
    node_type = "goos.action.freeze_variable"

    def __init__(self, var: Variable) -> None:
        super().__init__(var)
        self._var = var

    def run(self, plan: goos.OptimizationPlan) -> None:
        plan.freeze_var(self._var)


class ThawVariable(goos.Action):
    node_type = "goos.action.thaw_variable"

    def __init__(self, var: Variable) -> None:
        super().__init__(var)
        self._var = var

    def run(self, plan: goos.OptimizationPlan) -> None:
        plan.thaw_var(self._var)


def find_dominant_numeric_flow(flow_list: List[goos.NumericFlow],
                               return_ind: bool = False) -> goos.NumericFlow:
    """Finds the dominant `goos.NumericFlow`."""
    # Only one of the flows can be a non-numeric-flow.
    # This is a necessary check because the inputs could be flows that
    # inherit `NumericFlow`. It is ambiguous how to define `a + b` where
    # neither `a` nor `b` has type `flows.NumericFlow`, so an error is
    # thrown in that case.
    dom_flow = None
    ind = 0
    for i, flow in enumerate(flow_list):
        if type(flow) == flows.NumericFlow:
            continue
        if not dom_flow:
            dom_flow = flow
            ind = i
        else:
            raise ValueError(
                "Input arguments to sum must have at most one flow where"
                " type is not `flows.NumericFlow`.")

    if not dom_flow:
        # Check whether all flows have same shape.
        flow_shapes = [np.shape(flow.array) for flow in flow_list]

        # Check if all flows have same shape.
        same_shape = True
        for shape in flow_shapes:
            if shape != flow_shapes[0]:
                same_shape = False
                break

        if same_shape:
            dom_flow = flow_list[0]
            ind = 0

    if not dom_flow:
        # Check whether all flows but one is a scalar.
        for i, flow in enumerate(flow_list):
            if np.size(flow.array) > 1:
                if not dom_flow:
                    dom_flow = flow
                    ind = i
                else:
                    raise ValueError("Only one non-scalar is permitted.")

    if not dom_flow:
        dom_flow = flow_list[0]
        ind = 0

    if return_ind:
        return dom_flow, ind
    else:
        return dom_flow


class Sum(Function):
    node_type = "goos.function.sum"

    def __init__(self, functions: List[Function]):
        super().__init__(functions)
        self._fun = functions

    def eval(self, inputs: List[flows.NumericFlow],
             context: goos.EvalContext) -> flows.NumericFlow:
        flow = copy.deepcopy(find_dominant_numeric_flow(inputs))
        flow.array = np.sum([x.array for x in inputs], axis=0)
        return flow

    def grad(self, inputs: List[flows.NumericFlow],
             grad_val: flows.NumericFlow.Grad,
             context: goos.EvalContext) -> List[flows.NumericFlow.Grad]:
        grads = []
        _, ind = find_dominant_numeric_flow(inputs, return_ind=True)
        for i in range(len(inputs)):
            if ind == i:
                grads.append(copy.deepcopy(grad_val))
            else:
                grads.append(goos.NumericFlow.Grad(grad_val.array_grad))
        return grads

    def __add__(self, obj):
        if isinstance(obj, Sum):
            return Sum(self._fun + obj._fun)
        if isinstance(obj, Function):
            return Sum(self._fun + [obj])
        if isinstance(obj, numbers.Number) or isinstance(obj, np.ndarray):
            return Sum(self._fun + [Constant(obj)])
        if isinstance(obj, np.ndarray):
            return Sum(self._fun + [Constant(obj)])
        raise TypeError(
            "Attempting to add a node with type {} to type `Sum`.".format(
                type(obj)))


class Product(Function):
    node_type = "goos.function.product"

    def __init__(self, functions: List[Function]) -> None:
        super().__init__(functions)
        self._fun = functions

    def eval(self, inputs: List[flows.NumericFlow]) -> flows.NumericFlow:
        flow = copy.deepcopy(find_dominant_numeric_flow(inputs))
        flow.array = np.prod([x.array for x in inputs], axis=0)
        return flow

    def grad(self, input_vals: List[flows.NumericFlow],
             grad_val: flows.NumericFlow.Grad) -> List[flows.NumericFlow.Grad]:
        input_val_arr = [f.array for f in input_vals]
        grads = []
        for i in range(len(input_val_arr)):
            # Calculate product of everything except ith objective value.
            prod_before = np.prod(input_val_arr[:i], axis=0)
            prod_after = np.prod(input_val_arr[i + 1:], axis=0)
            grads.append(prod_before * prod_after * grad_val.array_grad)

        grad_flows = []
        _, ind = find_dominant_numeric_flow(input_vals, return_ind=True)
        for i, grad in enumerate(grads):
            if ind == i:
                flow = copy.deepcopy(grad_val)
                flow.array_grad = grad
                grad_flows.append(flow)
            else:
                grad_flows.append(goos.NumericFlow.Grad(grad))
        return grad_flows

    def __mul__(self, obj):
        if isinstance(obj, Product):
            return Product(self._fun + obj._fun)
        if isinstance(obj, Function):
            return Product(self._fun + [obj])
        if isinstance(obj, numbers.Number) or isinstance(obj, np.ndarray):
            return Product(self._fun + [Constant(obj)])
        if isinstance(obj, np.ndarray):
            return Product(self._fun + [Constant(obj)])
        raise TypeError(
            "Attempting to add a node with type {} to type `Sum`.".format(
                type(obj)))


class Power(Function):
    node_type = "goos.function.power"

    def __init__(self, fun: Function, power: float) -> None:
        super().__init__(fun)
        self._pow = power

    def eval(self, inputs: List[flows.NumericFlow],
             context: goos.EvalContext) -> flows.NumericFlow:
        value = copy.deepcopy(inputs[0])
        value.array = value.array**self._pow
        return value

    def grad(self, input_vals: List[flows.NumericFlow],
             grad_val: flows.NumericFlow.Grad,
             context: goos.EvalContext) -> List[flows.NumericFlow.Grad]:
        new_grad_val = copy.deepcopy(grad_val)
        new_grad_val.array_grad *= (self._pow *
                                    input_vals[0].array**(self._pow - 1))
        return [new_grad_val]


class AbsoluteValue(Function):
    node_type = "goos.function.abs"

    def __init__(self, fun: Function) -> None:
        super().__init__(fun)

    def eval(self, input_vals: List[goos.NumericFlow]) -> goos.NumericFlow:
        val = copy.deepcopy(input_vals[0])
        val.array = np.abs(val.array)
        return val

    def grad(self, input_vals: List[goos.NumericFlow],
             grad_val: goos.NumericFlow.Grad) -> List[goos.NumericFlow.Grad]:
        grad = np.conj(input_vals[0].array) / np.abs(
            input_vals[0].array) * grad_val.array_grad
        # TODO(logansu): Fix gradients here. The fundamental issue is that for
        # `f(z) = |z|`, df/dz differs by a factor of 2 depending on whether z is
        # considered complex or real. For now, we attempt to detect the type of
        # z via `iscomplexobj` but this seems a bit fragile (as one could
        # pass in a real number into what is supposed to be considered a complex
        # input).
        if np.iscomplexobj(input_vals[0].array):
            grad /= 2

        grad_val = copy.deepcopy(grad_val)
        grad_val.array_grad = grad
        return [grad_val]


def abs(fun: Function, **kwargs) -> AbsoluteValue:
    return AbsoluteValue(fun, **kwargs)


class DotProduct(Function):
    """Computes the dot product between two numeric functions.

    The dot product is defined as `sum(x_i * y_i)` where the `i` index runs
    through all the dimensions of `x` and `y`.
    """
    node_type = "goos.function.dot_product"

    def __init__(self, fun1: Function, fun2: Function) -> None:
        super().__init__([fun1, fun2])

    def eval(self, input_vals: List[goos.NumericFlow]) -> goos.NumericFlow:
        return goos.NumericFlow(
            np.sum(input_vals[0].array * input_vals[1].array))

    def grad(self, input_vals: List[goos.NumericFlow],
             grad_val: goos.NumericFlow.Grad) -> List[goos.NumericFlow.Grad]:
        in1_grad = type(input_vals[0]).Grad()
        in2_grad = type(input_vals[1]).Grad()

        in1_grad.array_grad = grad_val.array_grad * input_vals[1].array
        in2_grad.array_grad = grad_val.array_grad * input_vals[0].array
        return [in1_grad, in2_grad]


def dot(fun1: Function, fun2: Function, **kwargs) -> DotProduct:
    if not isinstance(fun1, Function):
        fun1 = Constant(fun1)
    if not isinstance(fun2, Function):
        fun2 = Constant(fun2)
    return DotProduct(fun1, fun2, **kwargs)


class Norm(Function):
    """Computes the norm of a vector.

    If the vector is multi-dimensional, the vector is flattened before
    computing the norm. If the norm of the vector is zero, then the gradient
    is given as zero.
    """
    node_type = "goos.function.norm"

    def __init__(self, fun: Function, order: int = 2) -> None:
        """Creates a new norm node.

        Args:
            fun: Function to compute norm.
            order: Order of the norm.
        """
        super().__init__(fun)

        self._ord = order

    def eval(self, input_vals: List[goos.NumericFlow]) -> goos.NumericFlow:
        return goos.NumericFlow(
            np.linalg.norm(input_vals[0].array.flatten(), ord=self._ord))

    def grad(self, input_vals: List[goos.NumericFlow],
             grad_val: goos.NumericFlow.Grad) -> List[goos.NumericFlow.Grad]:
        arr = input_vals[0].array.flatten()
        norm_grad = np.conj(arr) * (
            np.abs(arr) / np.linalg.norm(arr, ord=self._ord))**(self._ord - 1)
        # Avoid division by zero.
        norm_grad[np.abs(arr) > 0] /= np.abs(arr[np.abs(arr) > 0])
        norm_grad = np.reshape(norm_grad, input_vals[0].array.shape)

        # TODO(logansu): Fix gradients here. The fundamental issue is that for
        # `f(z) = |z|`, df/dz differs by a factor of 2 depending on whether z is
        # considered complex or real. For now, we attempt to detect the type of
        # z via `iscomplexobj` but this seems a bit fragile (as one could
        # pass in a real number into what is supposed to be considered a complex
        # input).
        if np.iscomplexobj(input_vals[0].array):
            norm_grad /= 2

        grad = type(input_vals[0]).Grad()
        grad.array_grad = grad_val.array_grad * norm_grad
        return [grad]


def norm(fun: Function, order: int = 2, **kwargs) -> Norm:
    return Norm(fun, order, **kwargs)


class Max(Function):
    """Computes the maximum of several vectors.

    This function computes the element-wise maximum of the input vectors.
    """
    node_type = "goos.function.max"

    def __init__(self, funs: List[Function]) -> None:
        super().__init__(funs)

    def eval(self, input_vals: List[goos.NumericFlow]) -> goos.NumericFlow:
        # Concatenate all the arrays along the first axis and then take the
        # max over the first axis.
        arr = np.stack([node.array for node in input_vals], axis=0)
        max_arr = np.max(arr, axis=0)

        flow = copy.deepcopy(input_vals[0])
        flow.array = max_arr
        return flow

    def grad(self, input_vals: List[goos.NumericFlow], grad_val):
        arr = np.stack([node.array for node in input_vals], axis=0)
        max_arr_ind = np.argmax(arr, axis=0)

        grad_list = []
        for i, node in enumerate(input_vals):
            grad = np.array(grad_val.array_grad)
            grad[max_arr_ind != i] = 0
            grad_list.append(goos.NumericFlow.Grad(grad))
        return grad_list


def max(*args, **kwargs):
    return Max(list(args), **kwargs)


class Sigmoid(Function):
    node_type = "goos.function.sigmoid"

    def __init__(self, fun: Function) -> None:
        super().__init__(fun)

    def eval(self, inputs: List[goos.NumericFlow]) -> goos.NumericFlow:
        self._sig = 1 / (1 + np.exp(-inputs[0].array))

        out = copy.deepcopy(inputs[0])
        out.array = self._sig
        return out

    def grad(self, inputs: List[goos.NumericFlow],
             grad_val: goos.NumericFlow.Grad) -> goos.NumericFlow.Grad:
        grad_flow = copy.deepcopy(grad_val)
        grad_flow.array_grad *= self._sig * (1 - self._sig)
        return [grad_flow]


class Slice(Function):
    node_type = "goos.function.slice"

    def __init__(self, fun: Function, slices: List[Union[int, List[int],
                                                         str]]) -> None:
        super().__init__(fun)
        self._slices = slices

    def eval(self, inputs: List[goos.NumericFlow]) -> goos.NumericFlow:
        shape = inputs[0].array.shape
        slices = self._make_slices(shape)

        return goos.NumericFlow(inputs[0].array[slices])

    def grad(self, inputs: List[goos.NumericFlow],
             grad_val: goos.NumericFlow.Grad) -> goos.NumericFlow.Grad:
        shape = inputs[0].array.shape
        slices = self._make_slices(shape)

        grad = type(inputs[0]).Grad()
        grad.array_grad = np.zeros_like(inputs[0].array)
        grad.array_grad[slices] = grad_val.array_grad
        return [grad]

    def _make_slices(self, shape) -> List[slice]:
        slices = []
        for i, dim in enumerate(shape):
            if isinstance(self._slices[i], int):
                slices += [slice(self._slices[i], self._slices[i] + 1)]
            elif isinstance(self._slices[i], List):
                slices += [slice(self._slices[i][0], self._slices[i][1])]
            elif self._slices[i] == "c" or self._slices[i] == "center":
                slices += [slice(dim // 2, dim // 2 + 1)]
            elif self._slices[i] is None:
                slices += [slice(0, dim)]
            else:
                raise ValueError("Invalid slice value, got " +
                                 str(self._slices[i]))
        return tuple(slices)

class Conjugate(Function):
    node_type = "goos.function.conjugate"

    def __init__(self, fun: Function) -> None:
        super().__init__(fun)

    def eval(self, input_vals: List[goos.NumericFlow]) -> goos.NumericFlow:
        return goos.NumericFlow(np.conj(input_vals[0].array))

    def grad(self, input_vals: List[goos.NumericFlow],
            grad_val: goos.NumericFlow.Grad) -> goos.NumericFlow.Grad:
        grad = type(input_vals[0]).Grad()
        grad.array_grad = np.conj(grad_val.array_grad)
        return [grad]
