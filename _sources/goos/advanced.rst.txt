Advanced Topics
===============

Parallelization
---------------
By default, with the exception of simulations, SPINS computes the computational
graph serially and does not exploit any parallelism. Simulations, on the other
hand, are parallelized as much as possible by default. Because function and
gradient evaluation is dominated by the simulation time, you typically do not
need to change these defaults. However, if you may choose to
change this behavior on any node by calling the `parallelize` method on any
`ProblemGraphNode`:

.. code-block:: python

    # Parallelize computation of the node.
    node.parallelize()

    # Disables parallelization.
    node.parallelize(False)

Note that turning on parallelization may actually cause a decrease in
performance due to the overhead in the setup. Keep in mind that the
parallelization operates by grouping together and executing in parallel
operations that (1) are marked for parallel computing and (2) can be executed
independently (i.e. no direct or indirect dependency between the nodes.
Therefore, there may not be any true parallelization in effect if these
conditons are never met during the execution of the computational graph.

Array Flows
-----------
An *array flow* is, in essence, a list of other flows. Array flows are used to
group together multiple flows into a single flow. Working with array flows
is similar to working with arrays:


.. code-block:: python

    # Array flow is created by passing an array of flows.
    flow = goos.ArrayFlow([goos.NumericFlow(4), goos.ShapeFlow()])

    # Use indexing to set and get nth flow.
    # Prints 4.
    print(flow[0].array)
    flow[1].pos = np.array([3, 4, 5])

    # Prints 2.
    print(len(flow))


`ArrayFlow.Grad` works similarly:

.. code-block:: python

    # Array flow is created by passing an array of flows.
    flow = goos.ArrayFlow.Grad([goos.NumericFlow.Grad(4), goos.ShapeFlow.Grad()])

    # Use indexing to set and get nth flow.
    # Prints 4.
    print(flow[0].array_array_grad)
    flow[1].pos_grad = np.array([3, 4, 5])

Additionally, `ArrayFlow.Grad` supports adding multiple array flows together.
When doing this summation, a flow added to `None` is just the flow itself:

.. code-block:: python

    flow1 = goos.ArrayFlow.Grad([goos.NumericFlow.Grad(1),
                                 goos.NumericFlow.Grad(2)])
    flow2 = goos.ArrayFlow.Grad([goos.NumericFlow.Grad(3),
                                 goos.NumericFlow.Grad(4)])
    flow3 = goos.ArrayFlow.Grad([None, goos.NumericFlow.Grad(5)])
    flow4 = goos.ArrayFlow.Grad([None, None])

    flow1 + flow2 == goos.ArrayFlow.Grad([goos.NumericFlow.Grad(4),
                                          goos.NumericFlow.Grad(6)])

    flow1 + flow3 == goos.ArrayFlow.Grad([goos.NumericFlow.Grad(1),
                                          goos.NumericFlow.Grad(5)])

    flow1 + flow4 == flow1


Using `ArrayFlowOpMixin`
########################
For any node that produces an array flow, it is recommended that the node
inherits `ArrayFlowOpMixin`. This mixin overloads the indexing operator so that
individual elements of the output array flow can be easily accessed. Suppose
we have a node `MyNode` that produces an array flow with two elements. Then,
by inheriting from `ArrayFlowOpMixin`, we can compute the sum as follows:

.. code-block:: python

    # Computes the next two elements in the Fibonacci sequence.
    class FibonacciNode(goos.ArrayFlowOpMixin, goos.ProblemGraphNode):

      def __init__(self, in1: goos.Function, in2: goos.Function) -> None:
        super().__init__([in1, in2], flow_types=[goos.Function, goos.Function])
        ...

      def eval(self, inputs: List[goos.NumericFlow]) -> goos.ArrayFlow:
        fib_next = inputs[0].array + inputs[1].array
        fib_next_next = inputs[1].array + fib_next
        return goos.ArrayFlow([goos.NumericFlow(fib_next),
                               goos.NumericFlow(fib_next_next)])

      ...

    node = FibonacciNode(...)
    out_sum = node[0] * node[1]


Note that order of inheritance. Because it is a mixin, you should inherit from
`ArrayFlowOpMixin` before `ProblemGraphNode` (or any other node class).
Additionally, we pass an array `flow_types` to the mixin constructor. This
array sets the type of node that is returned when performing the indexing
operation.

You may also choose to set `flow_names`, which enables indexing by name
instead of by number:

.. code-block:: python

    class FibonacciNode(goos.ArrayFlowOpMixin, goos.ProblemGraphNode):

      def __init__(self, ...) -> None:
        super().__init__(...,
                         flow_types=[goos.Function, goos.Function],
                         flow_names=["first", "second"])
        ...

      ...

    node = FibonacciNode(...)
    out_sum = node["first"] * node["second"]

Using `IndexOp`
###############
You can also "manually" extract an element from an `ArrayFlow` node by using
the `IndexOp` node:


.. code-block:: python

    # Computes the next two elements in the Fibonacci sequence.
    class FibonacciNode(goos.ProblemGraphNode):

      def __init__(self, in1: goos.Function, in2: goos.Function) -> None:
        super().__init__([in1, in2])
        ...

      def eval(self, inputs: List[goos.NumericFlow]) -> goos.ArrayFlow:
        fib_next = inputs[0].array + inputs[1].array
        fib_next_next = inputs[1].array + fib_next
        return goos.ArrayFlow([goos.NumericFlow(fib_next),
                               goos.NumericFlow(fib_next_next)])

      ...

    node = FibonacciNode(...)
    out_sum = (goos.cast(IndexOp(node, 0), goos.Function)
               * goos.cast(IndexOp(node, 1), goos.Function)


Note that we had to cast the output node into `goos.Function` before being able
to use arithemetic operations. This arises from the fact that `IndexOp`
inherits directly from `ProblemGraphNode`, so arithmetic operations, which can
only operate on `Function` cannot be directly performed.

.. Multi-type Flows
   ----------------
   Flows can inherit from multiple flow types.
   Casting
   -------
   Custom Node Schemas
   -------------------
   Contexts
   --------
