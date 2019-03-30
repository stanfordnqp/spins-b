from typing import Any


def is_scalar(var: Any) -> bool:
    """
    Alias for 'not hasattr(var, "__len__")'

    :param var: Checks if var has a length.
    """
    return not hasattr(var, "__len__")
