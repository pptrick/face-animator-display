import copy
import pyrender
import numpy as np
import trimesh

def from_trimesh(mesh, normal, material=None, is_visible=True,
                     poses=None, wireframe=False, smooth=True):
    """Create a Mesh from a :class:`~trimesh.base.Trimesh`.

    Parameters
    ----------
    mesh : :class:`~trimesh.base.Trimesh` or list of them
        A triangular mesh or a list of meshes.
    normal: normal of the vertices
    material : :class:`Material`
        The material of the object. Overrides any mesh material.
        If not specified and the mesh has no material, a default material
        will be used.
    is_visible : bool
        If False, the mesh will not be rendered.
    poses : (n,4,4) float
        Array of 4x4 transformation matrices for instancing this object.
    wireframe : bool
        If `True`, the mesh will be rendered as a wireframe object
    smooth : bool
        If `True`, the mesh will be rendered with interpolated vertex
        normals. Otherwise, the mesh edges will stay sharp.

    Returns
    -------
    mesh : :class:`Mesh`
        The created mesh.
    """

    if isinstance(mesh, (list, tuple, set, np.ndarray)):
        meshes = list(mesh)
    elif isinstance(mesh, trimesh.Trimesh):
        meshes = [mesh]
    else:
        raise TypeError('Expected a Trimesh or a list, got a {}'
                        .format(type(mesh)))

    primitives = []
    for m in meshes:
        positions = None
        normals = None
        indices = None

        # Compute positions, normals, and indices
        if smooth:
            positions = m.vertices.copy()
            normals = normal
            indices = m.faces.copy()
        else:
            positions = m.vertices[m.faces].reshape((3 * len(m.faces), 3))
            normals = np.repeat(m.face_normals, 3, axis=0)

        # Compute colors, texture coords, and material properties
        color_0, texcoord_0, primitive_material = pyrender.Mesh._get_trimesh_props(m, smooth=smooth, material=material)

        # Override if material is given.
        if material is not None:
            #primitive_material = copy.copy(material)
            primitive_material = copy.deepcopy(material)  # TODO

        if primitive_material is None:
            # Replace material with default if needed
            primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.2,
                roughnessFactor=0.8
            )

        primitive_material.wireframe = wireframe

        # Create the primitive
        primitives.append(pyrender.primitive.Primitive(
            positions=positions,
            normals=normals,
            texcoord_0=texcoord_0,
            color_0=color_0,
            indices=indices,
            material=primitive_material,
            mode=pyrender.constants.GLTF.TRIANGLES,
            poses=poses
        ))

    return pyrender.Mesh(primitives=primitives, is_visible=is_visible)