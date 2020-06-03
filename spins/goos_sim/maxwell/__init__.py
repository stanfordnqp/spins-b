from spins.goos import schema_registry

# For sources and outputs.
SIM_REGISTRY = schema_registry.SchemaRegistryStack()
# For geometries.
GEOM_REGISTRY = schema_registry.SchemaRegistryStack()


def register(schema, **kwargs):

    def wrapper(cls):
        from spins.goos import flows
        if issubclass(schema, flows.Flow):
            name = schema.__name__
            registry = GEOM_REGISTRY
        else:
            name = schema._schema.fields["type"].choices[0]
            registry = SIM_REGISTRY
        registry.register(name, schema, cls, kwargs)
        return cls

    return wrapper

from spins.goos_sim.maxwell.render import *
from spins.goos_sim.maxwell.simspace import *
from spins.goos_sim.maxwell.simulate import *
