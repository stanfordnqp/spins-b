"""Defines the optimization plan schema and associated functions.

Note that the order of the imports here is important! This is a package rather
than a single module merely to split up code amongst multiple files to make it
more manageable. It behaves like a single module though.
"""
from spins.goos.optplan import context

# Declare globals.
# Stack of optimization plans that are active.
GLOBAL_PLAN_STACK = []
# GLobal stack context used for node registration.
GLOBAL_CONTEXT_STACK = context.ContextStack()
# Global variable for managing name collisions.
# Maps type names to number of instances instantiated with given node.
PROBLEM_GRAPH_NAME_MAP = {}  # pylint: disable=invalid-name

# pylint: disable=wildcard-import, wrong-import-position
# Make it easier to setup a schema by importing the required components into
# `optplan` module.
from schematics import types
from spins.goos.optplan import schema_utils
ModelNameType = schema_utils.polymorphic_model_type
polymorphic_model = schema_utils.polymorphic_model
from spins.goos.optplan.schema_utils import Model

from spins.goos.optplan.schema_optplan import *
from spins.goos.optplan.optplan import *
from spins.goos.optplan import schema
# pylint: enable=wildcard-import, wrong-import-position
