"""Defines utility functions for defining schemas with `schematics` library."""
import copy
from typing import List, Optional, Tuple, Union
import warnings

from schematics import models
from schematics import types


def polymorphic_model_type(name: str) -> types.StringType:
    """Returns a `StringType` for defining polymorphic models.

    See `polymorphic_model` decorator for more details.

    Args:
        name: Name identifying polymorphic model.
    """
    return types.StringType(default=name, choices=(name,), required=True)


def polymorphic_model(type_list: Optional[Union[List, Tuple[List]]] = None):
    """Registers class as a polymorphic model.

    This function returns a decorator that adds a class attribute a method
    `_claim_polymorphic` and adds the class to `type_list`. The decorator
    assumes that there is a class attribute called `type` which is generated
    using `polymorphic_model_type`.

    Polymorphic models are models that can occupy the same field but are
    different types, e.g. both `WaveguideMode` and `GaussianMode` are modes
    but require two different sets of parameters (models) to describe.

    Here, we differentiate different models with a string parameter `type` that
    differ in value based on the model. In the previous example, `WaveguideMode`
    would have a parameter `type` with value "waveguide_mode". For this reason,
    `type` must have distinct values across all polymorphic types.

    One way to distinguish between these possibilities during parsing is to
    include a `_claim_polymorphic` class function that accepts the data
    dictionary and reports whether the data matches the current model.

    Note that we do not attempt to add the `type` field using the decorator
    because the current implementation of `schematics` relies on metaclasses
    to build the schema before class creation. Thus, attempting to add a class
    attribute using the decorator would not affect the schema.

    Example:
    ```python
    @polymorphic_model(EXAMPLE_MODELS):
    class ExampleModel(models.Model):
        type = polymorphic_field_type("example")
        field1 =  ...
        ...
    ```

    is equivalent to

    ```python
    class ExampleModel(models.Model):
        def _claim_polymorphic(data):
            return data["type"] == "example"

        type = models.StringType(default="example", choices=("example",),
                                 required=True)
        field1 = ...
        ...

    EXAMPLE_MODELS.append(ExampleModel)
    ```

    Args:
        type_list: Either a list or a tuple of lists into which model should be
            registered.

    Returns:
        A class decorator.
    """

    def decorator(cls):
        if isinstance(type_list, tuple):
            for type_list_ in type_list:
                type_list_.append(cls)
        elif type_list is not None:
            type_list.append(cls)

        # Immediately fail if there is more than one possible choice for field
        # value as this would imply that this is not a proper polymorphic model
        # as we have defined here.
        assert len(cls._schema.fields["type"].choices) == 1

        def _claim_polymorphic(data):
            return data["type"] == cls._schema.fields["type"].choices[0]

        cls._claim_polymorphic = _claim_polymorphic  # pylint: disable=protected-access
        return cls

    return decorator


class CopyModelMixin:
    """Implements copying operations for `schematics.models.Model`.

    Because of implementation details, `schematics` models cannot be copied
    with normal conventions using `copy.copy` and `copy.deepcopy` (`copy.copy`
    will succeed but modifications to the fields will propagate to the copied
    object, and `copy.deepcopy` throws an error). This mixin is intended to
    be included with a `schematics` model and implements expected `copy` and
    `deepcopy` semantics as if the models were simply tuples.
    """

    def __copy__(self):
        new_model = type(self)()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for key, value in self.items():
                new_model[key] = value
        return new_model

    def __deepcopy__(self, memo):
        new_model = type(self)()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for key, value in self.items():
                new_model[key] = copy.deepcopy(value, memo)
        return new_model


class KwargsModelMixin:  # pylint: disable=too-few-public-methods
    """Implements ability to initialize a model using keyword arguments.

    In `schematics`, models can only be initialized using a dictionary. The
    reason is that model initialization can involve options that take keyword
    options. Using a dictionary as initialization can be a bit cumbersome,
    especially in cases where models are nested. This mixin enables
    initialization using keyword arguments directly. Any non-keyword arguments
    are passed to the constructor for `models.Model` and therefore dictionary
    initialization is also supported.

    Example:
    ```python
    class InnerModel(KwargsModelMixin, models.Model):
        value = types.IntField()

    class SampleModel(KwargsModelMixin, models.Model):
        value = types.IntField()
        inner = types.ModelField(InnerModel)

    # Initialization using keyword arguments.
    SampleModel(
        value=3,
        inner=InnerModel(value=4),
    )

    # Initialization using dictionary works still.
    SampleModel({
        "value": 3,
        "inner": {
            "value": 4,
        }
    })
    ```
    """

    def __init__(self, *args, **kwargs):
        """Initializes a new model.

        `args` is forwarded to the next `super` constructor. Any values in
        `kwargs` is set after model construction.
        """
        # Find all the values that should be passed to the initializer, i.e.
        # all the arguments that do not refer to field names.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            init_kwargs = {}
            for key, value in kwargs.items():
                if key not in self._schema.fields.keys():
                    init_kwargs[key] = value

            super().__init__(*args, **init_kwargs)
            for key, item in kwargs.items():
                if key in self._schema.fields.keys():
                    self[key] = item


class Model(KwargsModelMixin, models.Model, CopyModelMixin):
    """Defines a copyable model that can be inherited."""

    class Options:
        # Leave out any fields set to `None` during serialization.
        serialize_when_none = False
