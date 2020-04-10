from spins import goos

def Vec3d(**kwargs) -> goos.types.ListType:
    """Returns a 3D vector type."""
    return goos.types.ListType(goos.types.FloatType(), min_size=3, max_size=3, **kwargs)


class Box3d(goos.Model):
    """Represents an axis-aligned 3D rectangular prism.

    Attributes:
        center: Center of the box.
        extents: Length, width, and height of the box.
    """
    center = Vec3d()
    extents = Vec3d()
