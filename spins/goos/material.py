from spins import goos


class Material(goos.Model):
    """Defines a material.

    A material can be defined either by a name (e.g. "silicon") or by refractive
    refractive index.

    Attributes:
        mat_name: Name of a material. This needs to be a material defined in
            `spins.material`.
        mat_file: Path of CSV containing wavelength (microns),n,k columns.
            The format is the same as CSV's from refractiveindex.info.
        index: Refractive index value.
    """
    mat_name = goos.types.StringType()
    mat_file = goos.types.StringType()
    index = goos.schema_types.ComplexNumberType()


class ConstantMaterial:

    def __init__(self, index: complex) -> None:
        self._index = index

    def permittivity(self, wlen: float) -> complex:
        return self._index**2

    def __eq__(self, other) -> bool:
        if type(other) == ConstantMaterial:
            return self._index == other._index
        return False


def get_material(mat: Material):
    if mat is None:
        return None
    return ConstantMaterial(mat.to_native()["index"])
