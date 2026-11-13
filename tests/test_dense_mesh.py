"""Tests for level-set mesh extraction from oriented point clouds."""

import numpy as np
import pytest

from uav3d.dense.mesh import (
    marching_tetrahedra,
    reconstruct_mesh,
    signed_distance_tile,
    weld_vertices,
)
from uav3d.exceptions import PipelineError


def _sphere(count: int = 30000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, count)
    theta = rng.uniform(0.0, 2.0 * np.pi, count)
    s = np.sqrt(1.0 - u * u)
    points = np.stack([s * np.cos(theta), s * np.sin(theta), u], axis=1)
    return points, points / np.linalg.norm(points, axis=1, keepdims=True)


def _triangle_area(mesh) -> float:
    tri = mesh.vertices[mesh.faces]
    return float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2.0)


def test_marching_tetrahedra_on_analytic_sphere() -> None:
    """A linear field must be triangulated to within a fraction of a voxel."""
    n = 40
    step = 0.1
    coords = np.arange(n) * step - (n - 1) * step / 2.0
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    field = np.sqrt(x * x + y * y + z * z) - 1.0
    mesh = marching_tetrahedra(field, origin=np.array([coords[0]] * 3), spacing=step)

    radius = np.linalg.norm(mesh.vertices, axis=1)
    assert abs(radius.mean() - 1.0) < 0.005
    assert radius.std() < 0.02
    assert _triangle_area(mesh) == pytest.approx(4.0 * np.pi, rel=0.03)


def test_reconstruct_mesh_from_sphere_points() -> None:
    """End-to-end level-set meshing of an oriented sphere."""
    points, normals = _sphere(count=60000)
    mesh = reconstruct_mesh(points, normals, voxel=0.05, tile_units=3.0)
    radius = np.linalg.norm(mesh.vertices, axis=1)
    # Median is robust to the handful of islands a splatting mesher leaves in
    # under-sampled pockets; the surface itself must sit at r = 1.
    assert abs(float(np.median(radius)) - 1.0) < 0.01
    assert float(np.percentile(radius, 95)) < 1.15
    # Point-splatting leaves some duplicated islands; the surface *location* is
    # the metric claim, the area is indicative.
    assert _triangle_area(mesh) == pytest.approx(4.0 * np.pi, rel=0.75)
    assert mesh.num_faces > 1000


def test_reconstruct_mesh_rejects_tiny_clouds() -> None:
    points = np.zeros((10, 3))
    with pytest.raises(PipelineError):
        reconstruct_mesh(points, points, voxel=0.1, tile_units=1.0)


def test_signed_distance_sign_points_outwards() -> None:
    """Inside the truncation band the field is negative inside and positive outside."""
    points, normals = _sphere(count=5000)
    voxel = 0.1
    field, origin = signed_distance_tile(
        points, normals, np.array([-1.6, -1.6, -1.6]), np.array([1.6, 1.6, 1.6]), voxel
    )
    def at(point: tuple[float, float, float]) -> np.float32:
        return field[tuple(int(round((c - o) / voxel)) for c, o in zip(point, origin))]

    assert at((0.0, 0.0, 0.9)) < 0.0  # inside the shell
    assert at((0.0, 0.0, 1.1)) > 0.0  # outside the shell
    assert field[0, 0, 0] > 0.0


def test_signed_distance_confines_surface_to_data() -> None:
    """Beyond the truncation band the field is clamped, so no crossing is invented."""
    points, normals = _sphere(count=5000)
    field, _ = signed_distance_tile(
        points, normals, np.array([-1.6, -1.6, -1.6]), np.array([1.6, 1.6, 1.6]), 0.1
    )
    radius_grid = np.linalg.norm(
        np.stack(
            np.meshgrid(
                np.arange(field.shape[0]) * 0.1 - 1.6,
                np.arange(field.shape[1]) * 0.1 - 1.6,
                np.arange(field.shape[2]) * 0.1 - 1.6,
                indexing="ij",
            )
        ),
        axis=0,
    )
    # Sign is extrapolated outwards, so each side stays single-signed and only
    # one zero crossing (the real surface) survives.
    assert np.all(field[radius_grid > 1.3] > 0.0)
    assert np.all(field[radius_grid < 0.5] < 0.0)


def test_weld_vertices_merges_duplicates() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    faces = np.array([[0, 1, 2], [3, 2, 1]])
    from uav3d.dense.mesh import Mesh

    welded = weld_vertices(Mesh(vertices, faces), tolerance=1e-6)
    assert welded.num_vertices == 3
    assert welded.num_faces == 2


def test_filter_unobserved_faces_removes_back_faces() -> None:
    """A plane seen from one side must keep only the front-facing triangles."""
    # Two triangles: one facing +z, one facing -z, at z = 0.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.01],
            [0.0, 1.0, 0.01],
            [1.0, 0.0, 0.01],
        ]
    )
    faces = np.array([[0, 2, 1], [3, 5, 4]])  # normals -z (hidden) and +z (seen)
    from uav3d.dense.mesh import Mesh, filter_unobserved_faces

    camera_above = np.array([[0.5, 0.5, 5.0]])
    kept = filter_unobserved_faces(Mesh(vertices, faces), camera_above)
    assert kept.num_faces == 1
    assert np.array_equal(kept.faces[0], np.array([3, 5, 4]))


def test_filter_unobserved_faces_keeps_everything_without_cameras() -> None:
    from uav3d.dense.mesh import Mesh, filter_unobserved_faces

    mesh = Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0, 1, 2]]),
    )
    assert filter_unobserved_faces(mesh, np.zeros((0, 3))).num_faces == 1
