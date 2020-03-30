Core Concepts
=============

Here we describe the core concepts behind Goos.

Optimization Plan
-----------------
The *optimization plan* is the central object in SPINS. It consists of two
parts: *nodes* that form the *problem graph* and *actions* that use the
problem graph to update the state of the optimization. The problem graph
provides a complete description of the entire problem, from details of the
simulation to the exact objective functions. A sequence of actions define
the optimization strategy. Actions may change values of variables, minimize
a particular objective function, perform a discretization optimization, and so
forth. Actions use the problem graph to compute any quantities, such as the
objective function.

Optimization Plan Example
#########################

Consider a simple optimization plan to minimize the function :math:`(x - 1)^2`:

.. code-block:: python

   with goos.OptimizationPlan() as plan:
      x = goos.Variable(3, name="x")
      obj = (x - 1)**2

      goos.opt.scipy_minimize(obj, method="L-BFGS-B", max_iters=30)
      plan.run()

In this example, :code:`x` and :code:`obj` are nodes defined in the problem
graph and `goos.opt.scipy_minimize` creates an optimization action. Note that
the expressions :code:`(x-1)**2` implicitly defines several nodes and is a
shorthand for the following:

.. code-block:: python

   # The following is identical in behavior to `(x - 1)**2`.
   obj = goos.Power(goos.Sum([x, goos.Constant(-1)]), 2)

In goos, by using the optimization plan with a `with` statement, nodes and
actions that are defined within the `with`-block are added automatically to
the enclosing optimization plan. The above example could alternatively have
been written as follows:

.. code-block:: python

   plan = goos.OptimizationPlan()
   x = goos.Variable(3, name="x")
   obj = (x - 1)**2
   plan.add_node(obj)

   opt_action = goos.opt.ScipyOptimizer(obj, method="L-BFGS-B", max_iters=30)
   plan.add_action(opt_action)

   plan.run()

Notice that we do not have to add `x`, `obj`, and the implicit sum node
explicitly; `add_node` will automatically add all the dependencies of a given
node into the optimization plan.

For the rest of this documentation, we will assume that you are using context
managers (`with`-block) with the optimization plan, but you can always choose
to call the underlying optimization plan function to achieve the same results.

Executing Plans
###############
When an optimization plan is defined, no code is actually executed. This means
that you can write out your entire optimization methodology and make sure
it is syntactically correct. Additionally, goos has some basic type checks
to make sure you do not accidentally make a mistake in passing arguments to
goos node. That way, you do not run a long-running optimization, only to have it
suddenly fail because of a silly typo.

To actually execute a plan, you call `run`. All the actions will be executed
up to that point in time. If you add additional actions afterwards, you can
call `run` again to execute them:

.. code-block:: python

   with goos.OptimizationPlan() as plan:
      x = goos.Variable(3)

      # Add an action to set the variable to 5.
      # After this call, `x` still is 3.
      x.set(5)

      # After this call, `x` will be 5.
      plan.run()

      # Add another action.
      x.set(6)
      # After this call, `x` will be 6.
      plan.run()

Retrieving Values
#################

You can retrieve the value of any node by calling `get` and the gradient of
numerical nodes with respect to any node by calling `get_grad`:

.. code-block:: python

   with goos.OptimizationPlan() as plan:
     x = goos.Variable(3)
     y = 5 * x + 2

     x.get()  # Returns `goos.NumericFlow(array=3)`.
     y.get()  # Returns `goos.NumericFlow(array=17)`.

     y.get_grad([x])  # Returns `[goos.NumericFlow.Grad(array_grad=5)]`.

Note that the return value of `get` and `get_grad` are *flows*. The reason is
that some nodes do not return numeric values but instead return other types,
such as shapes. For numeric functions and variables, they always return
`goos.NumericFlow` for the function and `goos.NumericFlow.Grad` for the
gradient.


Keep in mind that calling `get` does not execute actions. It evaluates the node
with the current optimization plan state. You can pass `run=True` to `get`
in order to call `run` on the plan before retrieving the value:

.. code-block:: python

   with goos.OptimizationPlan() as plan:
     x = goos.Variable(3)

     x.set(5)
     x.get()  # Returns `goos.NumericFlow(array=3)`.

     # The following is equivalent to the lines:
     #     plan.run()
     #     x.get()
     x.get(run=True)  # Returns `goos.NumericFlow(array=5)`.

Logging and Checkpoints
#######################
Some actions, such as optimizations, will generate logging information and
periodically save the state of the optimization plan. This data will be saved
in the optimization plan save directory, which is set by passing in `save_path`
when creating an optimization plan:

.. code-block:: python

   with goos.OptimizationPlan(save_path="/path/to/myplan") as plan:
      x = goos.Variable(3)
      obj = (x - 1)**2
      goos.opt.scipy_minimize((x - 1)**2, "CG", monitor_list=[obj])
      plan.run()

      # You should see `/path/to/myplan` contain Pickle files containing the
      # state of each optimization step. Each file contains information
      # about `x` as well as the objective function value `obj`.

Instead of relying on actions to save the state, you can force *checkpoints* to
be saved at any time:

.. code-block:: python

   with goos.OptimizationPlan(save_path="/path/to/myplan") as plan:
      x = goos.Variable(3)
      # The following will write the state to
      # `/path/to/myplan/mycheckpoint.chkpt`.
      plan.write_checkpoint("mycheckpoint.chkpt")

Saving and Loading Plans
########################

Optimization plans can be loaded and saved using the `load` and `save` commands.
Note that these functions only load and save the problem graph and actions but
do not save any variable state. See :ref:`Logging and Checkpoints` for saving
actual state.

.. code-block:: python

   with goos.OptimizationPlan(save_path="/path/to/myplan") as plan:
      x = goos.Variable(3, name="x")
      goos.opt.scipy_minimize((x - 1)**2, "CG")
      # The following creates `/path/to/myplan/optplan.json`.
      plan.save()


   with goos.OptimizationPlan(save_path="/path/to/myplan") as plan:
      # The following loads from `/path/to/myplan/optplan.json`.
      # You could also explicitly state the save folder:
      # `plan.load("/path/to/myplan")`.
      plan.load()
      x = plan.get_node("x")
      # `x.get() == 3`

      plan.run()
      # `x.get() == 1`.


Debugging Plans
###############

Because plans are not executed as soon as nodes are declared, you may find it
useful to declare temporary debugging plans to test out the behavior of your
code:

.. code-block:: python

   with goos.OptimizationPlan() as plan:

      # Some code that involves a lot of computation (e.g. electromagnetic
      # simulations).
      ...

      x = goos.Variable(3)
      y = x**2

      x.set(3)

      # We want to know what the value of `y` would be here but we do not
      # want to run the plan and trigger the code that involves a lot of
      # computation. Instead, we create a temporary plan which only includes
      # `x` and `y`.
      with goos.OptimizationPlan() as temp_plan:
          # This new plan does NOT have the `x.set` action so we repeat it.
          x.set(3)
          # This executes the `temp_plan` and not the original `plan`.
          y.get(run=True)  # Returns 9.

          # We could continue to test out what `y` equals with different
          # values of `x`.
          x.get(4)
          y.get(run=True)

      # Here, `temp_plan` is destroyed, along with all of its actions and state.
      # (`x.get()` returns 3 and `y.get()` returns 9 still).

Problem Graph
-------------


Variable Nodes
##############
Variable nodes can be thought of as nodes that hold raw numeric data. Variables
act as sources of data for the rest of the computational graph, and the state
of the optimization plan is fully captured by the state of all the variables.

To create a variable, simply pass in the initial value:

.. code-block:: python

    var_scalar = goos.Variable(4)
    var_vector = goos.Variable([3, 4, 5], name="vector")

As the optimization plan is executed, the actions will change the values of
of the variables. However, you can directly set the value of a variable using
the `set` method:

.. code-block:: python

    var = goos.Variable(4)
    var.set(5)

    # We can also use one variable to set the value of another.
    var2 = goos.Variable(8)
    var2.set(var)

In SPINS, a variable state contains the following:

- a numeric value, stored as a NumPy array
- upper and lower bounds on each element of array
- its frozen state

The upper and lower bounds of a variable (i.e. box constraints) are used by
optimizers when performing an optimization. These are treated differently than
generic constraints as there are many optimization algorithms that can handle
these box constraints but not general constraints. Upper and lower bounds
can be set during initialization:

.. code-block:: python

    # Constraint the variable to between 0 and 10.
    var = goos.Variable(5, lower_bounds=0, upper_bounds=10)

    # Each entry can have different bounds. The following constraints
    # the first entry to be between 0 and 1 and the second entry to be between
    # 0 and 2.
    var2 = goos.Variable([0.1, 0.2], lower_bounds=[0, 0], upper_bounds=[1, 2])

Variables also have a boolean flag indicating whether they are *frozen*. Frozen
variables have zero gradient and are not (usually) modified by actions. The
"frozenness" can be changed by calling `freeze` and `thaw`:

.. code-block:: python

    x = goos.Variable(3)
    y = goos.Variable(4)


    x.freeze()
    goos.opt.scipy_minimize(x**2 + y**2, "CG")
    # `x` remains at 3 because it is frozen. `y` is now zero.
    assert x.get() == 3
    assert y.get() == 0

    x.thaw()
    goos.opt.scipy_minimize(x**2 + y**2, "CG")
    # Now both `x` and `y` are zero.
    assert x.get() == 0
    assert y.get() == 0

    x.freeze()
    # The following will raise an exception during execution!
    x.set(5)

Sometimes a variable is meant as a parameter and should never be
optimized over. In these cases, the variable can be declared as a *parameter*.
Parameters are always frozen but can be set explicitly using `set`:

.. code-block:: python

    param = goos.Variable(3, parameter=True)
    y = goos.Variable(4)

    goos.opt.scipy_minimize(param**2 * y**2, "CG")
    # `param` does not change because it is frozen by default.
    assert param.get() == 3
    assert y.get() == 0

    # The following does NOT raise an exception because it is initialized as
    # a parameter.
    param.set(3)

Math Nodes
##########
Math (`goos.Function`) nodes are nodes that perform mathematical operations,
such as addition, multiplication, and dot products.  They take in numerical
input and produce numerical output (specifically, they take other
`goos.Function` nodes as input and produce `goos.NumericFlow` as output).

SPINS has implemented a common subset of useful mathematical functions and
provided operator overloads for the basic operations.

.. code-block:: python

   x = goos.Variable([1, 2])
   y = goos.Variable([3, 4])

   # Element-wise operations.
   sum_node = x + y
   prod_node = x * y
   sub_node = x - y
   div_node = x / y
   # Note that the power must be a constant. It can NOT be a variable.
   power_node = x**2

   # Vector operations.
   # Computes `||x||`.
   norm_node = goos.norm(x)
   dot_prod_node = goos.dot(x, y)


Shape Nodes
###########
Shape nodes are those that represent a permittivity distribution and include
objects such as cylinders and boxes. Parametrized permittivity distributions
are also shapes.

Simple shapes, such as cylinders, as straightforward to create:

.. code-block:: python

    box = goos.Cuboid(pos=goos.Constant([0, 0, 0]),
                      extents=goos.Constant([1000, 400, 220]),
                      material=goos.material.Material(index=2))
    cyl = goos.Cylinder(pos=goos.Constant([100, 0, 0]),
                        radius=goos.Constant([50]),
                        height=goos.Constant([220]),
                        material=goos.material.Material(index=3))

    # Notice that you can also compute these quantities dynamically.
    start_pos = goos.Constant([0, 0, 0])
    delta_pos = goos.Constant([100, 0, 0])
    boxes = []
    for i in range(10):
      boxes.append(goos.Cuboid(pos=start_pos + i * delta_pos,
                               extents=goos.Constant([10, 10, 10]),
                               material=goos.material.Material(index=2))

SPINS also defines certain shapes useful for inverse design. For example,
the pixelated continuous shape represents a shape composed of voxels that can
take on permittivities continuously between that of two materials. Often there
are special functions defined to help create these shapes.

.. code-block:: python

    # The initializer is a function that accepts a single parameter `shape` and
    # must return an array of numbers with shape `shape`.
    def initializer(shape):
      return np.random.random(shape) * 0.2 + 0.5

    # `vae` is a `goos.Variable` node that controls the value of the shape node
    # `design`.
    var, design = goos.pixelated_cont_shape(
        initializer=initializer,
        pos=goos.Constant([0, 0, 0]),
        extents=[2000, 2000, 220],
        pixel_size=[40, 40, 220],  # Each voxel is 40 x 40 x 220.
        # The pixels can have refractive indices between 1 and 2.
        material=goos.material.Material(index=1),
        material2=goos.material.Material(index=2))

Simulation Nodes
################

Simulations are nodes themselves. Simulation nodes take as input the
permittivity distribution and produces as output the electric fields and other
related quantities, such as modal overlaps. Note that each simulation node
has a different set of capabilities, so you should consult the documentation
for each simulation node. Typically, setting up the simulation involves the
following components:

- Specification of the simulation space, i.e. simultaions extents, meshing,
  boundary conditions, etc.
- Permittivity distribution to simulate
- Source specification
- Output specification, e.g. electric fields, modal overlaps, etc.
- Additional simulation-specific parameters.

As an example, below is how to setup a FDFD simulation using the built-in
Maxwell solver:

.. code-block:: python

    # Import the desired simulator.
    from spins.goos.simulator import maxwell

    waveguide = goos.Cuboid(...)
    var, design = goos.pixelated_cont_shape(...)

    # Group the waveguide and design together into one permittivity
    # distribution.
    eps = goos.GroupShape([waveguide, design])

    sim = maxwell.fdfd_simulation(
        name="sim",
        wavelength=1550,  # Wavelength of the simulation.
        simulation_space=maxwell.SimulationSpace(
            mesh=maxwell.UniformMesh(dx=40),
            sim_region=goos.Box3d(
                center=[0, 0, 0],
                extents=[4000, 4000, sim_z_extent],
            ),
            pml_thickness=[400, 400, 400, 400, 0, 0]),
        eps=eps,
        sources=[
            # Add a single waveguide mode source.
            maxwell.WaveguideModeSource(center=[-1400, 0, 0],
                                        extents=[0, 2500, 1000],
                                        normal=[1, 0, 0],
                                        mode_num=0,
                                        power=1)
        ],
        background=goos.material.Material(index=1.0),
        outputs=[
            maxwell.Epsilon(name="eps"),
            maxwell.ElectricField(name="field"),
            maxwell.WaveguideModeOverlap(name="overlap",
                                         center=[0, 1400, 0],
                                         extents=[2500, 0, 1000],
                                         normal=[0, 1, 0],
                                         mode_num=0,
                                         power=1),
        ],
        solver="local_direct",
    )

    # We can now extract the simulation outputs either as `sim[0]`, `sim[1] `,
    # etc. or as `sim["eps"]`, `sim["field"]`, etc. because we added a name
    # for each output.

    # Define an objective function based on the modal overlap integral.
    obj = 1 - goos.abs(sim["overlap"])**2

Flows
#####
Flows are data objects that are generated by *nodes*. These include general
NumPy arrays and shape objects that represent permittivity distributions.
Flows are essentially a generalization of tensors used in machine learning.
Unless you are implementing your own nodes, you will mainly only encounter nodes
when evaluating the result of a node (i.e. call `node.get`).

NumericFlow
~~~~~~~~~~~
Numeric flows have a single field called `array`, which contains a
multi-dimensional NumPy array. There is no instrinsic meaning behind the values
in the array. Math nodes (those that inherit from `goos.Function`) return
numeric flows.

Numeric flows have some basic overloads for `==` so that you can quickly compare
numeric flows and the underlying array.

.. code-block:: python

    x = goos.Variable(3)
    y = x + 1
    flow = y.get()

    # Flows have an `array` property.
    assert flow.array == 4
    # But for simple cases, you can drop the `array`.
    assert flow == 4


Actions
-------
Where as nodes setup the problem grpah and determine how values are computed,
actions actually perform the computation and are able to modify variable values.
In fact, only actions are allowed to modify the state of any variables.
The most common action is to run an optimization, but a single optimization plan
may contain many actions, including setting/changing variable values and running
a discretization procedure.

You can distinguish an action from a node in that actions always inherit from
:code:`goos.Action`, though it should be clear from context whether something
is an action.

In the code snippet below, the calls to :code:`goos.opt.scipy_minimize` and
:code:`goos.Variable.set` generate actions.

.. code-block:: python

    with goos.OptimizationPlan() as plan:
      x = goos.Variable(1)
      target = goos.Variable(3, parameter=True)
      obj = (x - target)**2

      goos.opt.scipy_minimize(obj, "L-BFGS-B", max_iters=10)

      target.set(4)
      goos.opt.scipy_minimize(obj, "L-BFGS-B", max_iters=10)


Remember that actions are not actually executed until :code:`OptimizationPlan.run`
is invoked. The optimization plan maintains an action pointer that remembers
that last executed action, so you can execute an action, add more actions,
and then call :code:`run` again to execute only the newly added actions:

.. code-block:: python

    with goos.OptimizationPlan() as plan:
      x = goos.Variable(1)

      # Action to increment `x`.
      x.set(x + 1)

      print(x.get().array)  # Prints 1.

      plan.run()

      print(x.get().array)  # Prints 2.

      x.set(x + 1)
      plan.run()

      print(x.get().array)  # Prints 3.
