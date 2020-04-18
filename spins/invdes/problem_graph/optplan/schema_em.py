"""Defines schema for electromagnetic-related nodes."""
import enum

from schematics import types

from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import schema_utils

BOUNDARY_CONDITION_TYPES = []
MESH_TYPES = []


class Material(schema_utils.Model):
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
    mat_name = types.StringType()
    mat_file = types.StringType()
    index = types.PolyModelType(optplan.ComplexNumber)


class GdsMaterialStackLayer(schema_utils.Model):
    """Defines a single layer in a material stack.

    Attributes:
        foreground: Material to fill any structure in the layer.
        background: Material to fill any non-structure areas in the layer.
        extents: Start and end coordiantes of the layer stack.
        gds_layer: Name of GDS layer that contains the polygons for this layer.
    """
    foreground = types.ModelType(Material)
    background = types.ModelType(Material)
    extents = optplan.vec2d()
    gds_layer = types.ListType(types.IntType())


class GdsMaterialStack(schema_utils.Model):
    """Defines a material stack.

    This is used by `GdsEps` to define the permittivity distribution.

    Attributes:
        background: Material to fill any regions that are not covered by
            a material stack layer.
        stack: A list of `MaterialStackLayer` that defines permittivity for
            each layer.
    """
    background = types.ModelType(Material)
    stack = types.ListType(types.ModelType(GdsMaterialStackLayer))


class EpsilonSpec(schema_utils.Model):
    """Describes a specification for permittivity distribution."""


@schema_utils.polymorphic_model()
class GdsEps(EpsilonSpec):
    """Defines a permittivity distribution using a GDS file.

    The GDS file will be flattened so that each layer only contains polygons.
    TODO(logansu): Expand description.

    Attributes:
        type: Must be "gds_epsilon".
        gds: URI of GDS file.
        mat_stack: Description of each GDS layer permittivity values and
            thicknesses.
        stack_normal: Direction considered the normal to the stack.
    """
    type = schema_utils.polymorphic_model_type("gds")
    gds = types.StringType()
    mat_stack = types.ModelType(GdsMaterialStack)
    stack_normal = optplan.vec3d()


class Mesh(schema_utils.Model):
    """Defines a mesh to draw.

    Meshes are used to define permittivities through `GdsMeshEps`.
    """


@schema_utils.polymorphic_model()
class GdsMesh(Mesh):
    """Defines a mesh by using polygons from a GDS file.

    The mesh is defined by extruding the polygon along the stack normal with
    coordinates given by `extents`.

    Attributes:
        material: Material to use for mesh.
        extents: Start and end location of mesh in the extrusion direction.
        gds_layer: Tuple `(layer, datatype)` of the GDS file from which to
            extract the polygons.
    """
    type = schema_utils.polymorphic_model_type("mesh.gds_mesh")
    material = types.ModelType(Material)
    extents = optplan.vec2d()
    gds_layer = types.ListType(types.IntType())


@schema_utils.polymorphic_model()
class SlabMesh(Mesh):
    """Defines a slab.

    A slab is a rectangular prism that has a finite extent along the extrusion
    axis and infinite extent in the other two directions. Slabs are commonly
    used to draw a background permittivity distribution before drawing
    other meshes.

    Attributes:
        material: Material to use for slab.
        extents: Start and end location of slab in the extrusion direction.
    """
    type = schema_utils.polymorphic_model_type("mesh.slab")
    material = types.ModelType(Material)
    extents = optplan.vec2d()


@schema_utils.polymorphic_model()
class GdsMeshEps(EpsilonSpec):
    """Defines a permittivity distribution by a lits of meshes.

    The meshes are drawn in order of the list. Consequently, if meshes overlap,
    the mesh drawn later will take precedence.

    Attributes:
        gds: GDS file to use for `GdsMesh` types.
        background: Default background permittivity.
        mesh_list: List of meshes to draw.
        stack_normal: Direction considered the normal to the stack.
    """
    type = schema_utils.polymorphic_model_type("gds_mesh")
    gds = types.StringType()
    background = types.ModelType(Material)
    mesh_list = types.ListType(types.PolyModelType(Mesh))
    stack_normal = optplan.vec3d()


@schema_utils.polymorphic_model()
class ParamEps(EpsilonSpec):
    """Defines a permittivity distribution based on a parametriation.

    Attributes:
        type: Must be "parametrization".
        parametrization: Name of the parametrization.
        simulation_space: Name of the simulation space.
        wavelength: Wavelength.
    """
    type = schema_utils.polymorphic_model_type("parametrization")
    parametrization = optplan.ReferenceType(optplan.Parametrization)
    simulation_space = optplan.ReferenceType(optplan.SimulationSpaceBase)
    wavelength = types.FloatType()


@schema_utils.polymorphic_model(MESH_TYPES)
class UniformMesh(schema_utils.Model):
    """Defines a uniform mesh.

    Attributes:
        type: Must be "uniform".
        dx: Unit cell distance for EM grid (nm).
    """
    type = schema_utils.polymorphic_model_type("uniform")
    dx = types.FloatType()


@schema_utils.polymorphic_model(BOUNDARY_CONDITION_TYPES)
class BlochBoundary(schema_utils.Model):
    """Represents a Bloch boundary condition.

    Attributes:
        bloch_vector: 3D Bloch optplan.vector.
    """
    type = schema_utils.polymorphic_model_type("bloch")
    bloch_vector = optplan.vec3d(default=[0, 0, 0])


@schema_utils.polymorphic_model(BOUNDARY_CONDITION_TYPES)
class PecBoundary(schema_utils.Model):
    """Represents PEC boundary."""
    type = schema_utils.polymorphic_model_type("pec")


@schema_utils.polymorphic_model(BOUNDARY_CONDITION_TYPES)
class PmcBoundary(schema_utils.Model):
    """Represents PMC boundary."""
    type = schema_utils.polymorphic_model_type("pmc")


class SelectionMatrixType(enum.Enum):
    """Defines possible types for selection matrices."""
    # Direct lattice selection matrix where we select out all points in the
    # Yee grid.
    DIRECT = "direct_lattice"
    # Same as `DIRECT` but permittivity values along the extrusion direction
    # are not constrained to be equal to each other.
    FULL_DIRECT = "full_direct"
    # Design dimensions is reduced by factor of 4 by parametrizing only the "z"
    # component.
    REDUCED = "uniform"


@optplan.register_node_type()
class SimulationSpace(optplan.SimulationSpaceBase):
    """Defines a simulation space.

    A simulation space contains information regarding the permittivity
    distributions but not the fields, i.e. no information regarding sources
    and wavelengths.

    Attributes:
        name: Name to identify the simulation space. Must be unique.
        eps_fg: Foreground permittivity.
        eps_bg: Background permittivity.
        mesh: Meshing information. This describes how the simulation region
            should be meshed.
        sim_region: Rectangular prism simulation domain.
        selection_matrix_type: The type of selection matrix to form. This
            is subject to change.
    """
    type = schema_utils.polymorphic_model_type("simulation_space")
    eps_fg = types.PolyModelType(EpsilonSpec)
    eps_bg = types.PolyModelType(EpsilonSpec)
    mesh = types.PolyModelType(MESH_TYPES)
    sim_region = types.ModelType(optplan.Box3d)
    boundary_conditions = types.ListType(
        types.PolyModelType(BOUNDARY_CONDITION_TYPES), min_size=6, max_size=6)
    pml_thickness = types.ListType(types.IntType(), min_size=6, max_size=6)
    selection_matrix_type = types.StringType(
        default=SelectionMatrixType.DIRECT.value,
        choices=tuple(select_type.value for select_type in SelectionMatrixType),
    )


@optplan.register_node_type()
class WaveguideMode(optplan.ProblemGraphNode):
    """Represents basic information for a waveguide mode.

    This class is not intended to be instantiable.

    Attributes:
        center: Waveguide center.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
    """
    type = schema_utils.polymorphic_model_type("em.waveguide_mode")
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()
    mode_num = types.IntType()
    power = types.FloatType()


@optplan.register_node_type()
class WaveguideModeSource(optplan.EmSource):
    """Represents a waveguide mode.

    The waveguide is assumed to be axis-aligned.

    Attributes:
        center: Waveguide center.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
    """
    type = schema_utils.polymorphic_model_type("source.waveguide_mode")
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()
    mode_num = types.IntType()
    power = types.FloatType()


@optplan.register_node_type()
class WaveguideModeOverlap(optplan.EmOverlap):
    """Represents a waveguide mode.

    The waveguide is assumed to be axis-aligned.

    Attributes:
        center: Waveguide center.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
    """
    type = schema_utils.polymorphic_model_type("overlap.waveguide_mode")
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()
    mode_num = types.IntType()
    power = types.FloatType()


@optplan.register_node_type()
class ImportOverlap(optplan.EmOverlap):
    """Represents a imported overlap vector.

    Attributes:
        file_name: .mat file containing the overlap vector.
        center: the center coordinate of the overlap, allows for translation
            of the overlap to the specified center.
    """
    type = schema_utils.polymorphic_model_type("overlap.import_field_vector")
    file_name = types.StringType()
    center = optplan.vec3d()


@optplan.register_node_type()
class PlaneWaveSource(optplan.EmSource):
    """Represents a plane wave source.

    Attributes:
        type: Must be "source.plane_wave".
    """
    type = schema_utils.polymorphic_model_type("source.plane_wave")
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()
    theta = types.FloatType()
    psi = types.FloatType()
    polarization_angle = types.FloatType()
    overwrite_bloch_vector = types.BooleanType()
    border = types.ListType(types.FloatType())
    power = types.FloatType()
    normalize_by_sim = types.BooleanType(default=False)


@optplan.register_node_type()
class GaussianSource(optplan.EmSource):
    """Represents a gaussian source.

    Attributes:
        type: Must be "source.gaussian_beam".
        normalize_by_sim: If `True`, normalize the power by running a
            simulation.
    """
    type = schema_utils.polymorphic_model_type("source.gaussian_beam")
    w0 = types.FloatType()
    center = optplan.vec3d()
    beam_center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()
    theta = types.FloatType()
    psi = types.FloatType()
    polarization_angle = types.FloatType()
    overwrite_bloch_vector = types.BooleanType()
    power = types.FloatType()
    normalize_by_sim = types.BooleanType(default=False)


@optplan.register_node_type()
class DipoleSource(optplan.EmSource):
    """Represents a dipole source.

    Attributes:
        position: Position of the dipole (will snap to grid).
        axis: Direction of the dipole (x:0, y:1, z:2).
        phase: Phase of the dipole source (in radian).
        power: Power assuming uniform dielectric space with the permittivity.
    """
    type = schema_utils.polymorphic_model_type("source.dipole_source")
    position = optplan.vec3d()
    axis = types.IntType()
    phase = types.FloatType()
    power = types.FloatType()
    normalize_by_sim = types.BooleanType(default=False)


@optplan.register_node_type()
class WaveguideModeEigSource(optplan.EmSource):
    """Represents a photonic crystal waveguide mode.

    The waveguide does NOT have to be axis-aligned. The waveguide mode is
    computed as a 3D eigenmode solve.

    Attributes:
        center: Waveguide center.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
    """
    type = schema_utils.polymorphic_model_type("source.waveguide_mode_eig")
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()
    mode_num = types.IntType()
    power = types.FloatType()


@optplan.register_node_type()
class WaveguideModeEigOverlap(optplan.EmOverlap):
    """Represents a photonic crystal waveguide mode.

    The waveguide does NOT have to be axis-aligned. The waveguide mode is
    computed as a 3D eigenmode solve.

    Attributes:
        center: Waveguide center.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
    """
    type = schema_utils.polymorphic_model_type("overlap.waveguide_mode_eig")
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()
    mode_num = types.IntType()
    power = types.FloatType()


@optplan.register_node_type()
class FdfdSimulation(optplan.Function):
    """Defines a FDFD simulation.

    Attributes:
        type: Must be "function.fdfd_simulation".
        name: Name of simulation.
        simulation_space: Simulation space name.
        source: Source name.
        wavelength: Wavelength at which to simulate.
        solver: Name of solver to use.
        bloch_vector: bloch optplan.vector at which to simulate.
    """
    type = schema_utils.polymorphic_model_type("function.fdfd_simulation")
    simulation_space = optplan.ReferenceType(optplan.SimulationSpaceBase)
    epsilon = optplan.ReferenceType(optplan.Function)
    source = optplan.ReferenceType(optplan.EmSource)
    wavelength = types.FloatType()
    solver = types.StringType(choices=("maxwell_bicgstab", "maxwell_cg",
                                       "local_direct"))
    bloch_vector = types.ListType(types.FloatType())


@optplan.register_node_type()
class Epsilon(optplan.Function):
    """Defines a Epsilon Grid.

    Attributes:
        type: Must be "function.epsilon".
        name: Name of epsilon.
        simulation_space: Simulation space name.
        wavelength: Wavelength at which to calculate epsilon.
    """
    type = schema_utils.polymorphic_model_type("function.epsilon")
    simulation_space = optplan.ReferenceType(optplan.SimulationSpaceBase)
    wavelength = types.FloatType()
    structure = optplan.ReferenceType(optplan.Parametrization)


@optplan.register_node_type()
class Overlap(optplan.Function):
    """Defines an overlap integral.

    Attributes:
        type: Must be "function.overlap".
        simulation: Simulation from which electric fields are obtained.
        overlap: Overlap type to use.
    """
    type = schema_utils.polymorphic_model_type("function.overlap")
    simulation = optplan.ReferenceType(optplan.Function)
    overlap = optplan.ReferenceType(optplan.EmOverlap)


@optplan.register_node_type()
class DiffEpsilon(optplan.Function):
    """Defines a function that finds the L1 norm between two permittivities.

    Specifially, the function is defined as `sum(|epsilon - epsilon_ref|)`.

    Attributes:
        type: Must be "function.diff_epsilon".
        epsilon: Permittivity.
        epsilon_ref: Base permittivity to compare to.
    """
    type = schema_utils.polymorphic_model_type("function.diff_epsilon")
    epsilon = optplan.ReferenceType(optplan.Function)
    epsilon_ref = types.PolyModelType(EpsilonSpec)
