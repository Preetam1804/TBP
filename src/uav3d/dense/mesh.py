"""Mesh reconstruction from a fused point cloud, without external 3D libraries.

The surface is extracted as a zero level set of an unsigned-distance field that
is signed by the point normals, then triangulated with marching *tetrahedra*
(the Kuhn decomposition of each cube). Tetrahedra avoid the 256-case marching
cubes table while still producing a watertight, crack-free surface.

The field is built per spatial tile so peak memory stays bounded regardless of
scene extent.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import ndimage

from uav3d.exceptions import PipelineError

# Corner offsets for cube vertex c = i + 2j + 4k.
_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int64,
)

# Six tetrahedra sharing the 0-7 diagonal (Kuhn decomposition: consistent across
# neighbouring cubes, so no cracks appear at cube boundaries).
_TETS = np.array(
    [
        [0, 1, 3, 7],
        [0, 1, 5, 7],
        [0, 2, 3, 7],
        [0, 2, 6, 7],
        [0, 4, 5, 7],
        [0, 4, 6, 7],
    ],
    dtype=np.int64,
)


@dataclass
class Mesh:
    """Triangle mesh with vertex positions only."""

    vertices: np.ndarray
    faces: np.ndarray

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])


def _edge_point(p0: np.ndarray, p1: np.ndarray, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """Linear interpolation of a zero crossing on an edge."""
    denom = v0 - v1
    safe = np.abs(denom) > 1e-12
    t = np.where(safe, v0 / np.where(safe, denom, 1.0), 0.5)
    t = np.clip(t, 0.0, 1.0)[:, None]
    return p0 + t * (p1 - p0)


def marching_tetrahedra(
    field_grid: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    chunk: int = 8,
) -> Mesh:
    """Triangulate the zero level set of ``field_grid`` (negative inside).

    Args:
        field_grid: ``(nx, ny, nz)`` scalar field.
        origin: world coordinate of grid node ``(0, 0, 0)``.
        spacing: uniform grid spacing in world units.
        chunk: number of k-slices processed at once (memory control).

    Returns:
        :class:`Mesh` in world coordinates.
    """
    nx, ny, nz = field_grid.shape
    if nx < 2 or ny < 2 or nz < 2:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int64))

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0
    for k0 in range(0, nz - 1, chunk):
        k1 = min(nz - 1, k0 + chunk)
        cube_k = np.arange(k0, k1, dtype=np.int64)
        ii, jj, kk = np.meshgrid(
            np.arange(nx - 1, dtype=np.int64),
            np.arange(ny - 1, dtype=np.int64),
            cube_k,
            indexing="ij",
        )
        shape = ii.shape
        ii = ii.ravel()
        jj = jj.ravel()
        kk = kk.ravel()

        values = np.empty((8, ii.size), np.float32)
        for c in range(8):
            dx, dy, dz = _CORNERS[c]
            values[c] = field_grid[ii + dx, jj + dy, kk + dz]

        vmin = values.min(axis=0)
        vmax = values.max(axis=0)
        active = (vmin < 0.0) & (vmax >= 0.0)
        if not np.any(active):
            continue
        ii, jj, kk = ii[active], jj[active], kk[active]
        values = values[:, active]
        pts = np.empty((8, ii.size, 3), np.float64)
        for c in range(8):
            dx, dy, dz = _CORNERS[c]
            pts[c, :, 0] = (ii + dx) * spacing + origin[0]
            pts[c, :, 1] = (jj + dy) * spacing + origin[1]
            pts[c, :, 2] = (kk + dz) * spacing + origin[2]

        tris: list[np.ndarray] = []
        for tet in _TETS:
            tv = values[tet]
            tp = pts[tet]
            inside = tv < 0.0
            # Case A: exactly one vertex differs from the other three.
            for p in range(4):
                others = [o for o in range(4) if o != p]
                unique_inside = inside[p] & ~inside[others[0]] & ~inside[others[1]] & ~inside[others[2]]
                unique_outside = ~inside[p] & inside[others[0]] & inside[others[1]] & inside[others[2]]
                mask = unique_inside | unique_outside
                if not np.any(mask):
                    continue
                e = [
                    _edge_point(tp[p][mask], tp[o][mask], tv[p][mask], tv[o][mask])
                    for o in others
                ]
                tris.append(np.stack(e, axis=1))
            # Case B: exactly two vertices inside.
            for p, q in combinations(range(4), 2):
                r, s = [o for o in range(4) if o not in (p, q)]
                mask = inside[p] & inside[q] & ~inside[r] & ~inside[s]
                if not np.any(mask):
                    continue
                e_pr = _edge_point(tp[p][mask], tp[r][mask], tv[p][mask], tv[r][mask])
                e_ps = _edge_point(tp[p][mask], tp[s][mask], tv[p][mask], tv[s][mask])
                e_qr = _edge_point(tp[q][mask], tp[r][mask], tv[q][mask], tv[r][mask])
                e_qs = _edge_point(tp[q][mask], tp[s][mask], tv[q][mask], tv[s][mask])
                tris.append(np.stack([e_pr, e_ps, e_qs], axis=1))
                tris.append(np.stack([e_pr, e_qs, e_qr], axis=1))

        if not tris:
            continue
        tri = np.concatenate(tris, axis=0)
        faces = np.arange(tri.shape[0] * 3, dtype=np.int64).reshape(-1, 3) + vertex_offset
        all_vertices.append(tri.reshape(-1, 3))
        all_faces.append(faces)
        vertex_offset += tri.shape[0] * 3

    if not all_vertices:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int64))
    return Mesh(np.concatenate(all_vertices, axis=0), np.concatenate(all_faces, axis=0))


def signed_distance_tile(
    points: np.ndarray,
    normals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    voxel: float,
    trunc_voxels: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse oriented points into a signed-distance field by splatting.

    Each point votes for the signed distance to its own tangent plane in the
    voxels around it, weighted by a tent kernel that reaches zero at the
    truncation distance (Curless & Levoy). Averaging many planes makes the field
    continuous, which is what a level-set mesher needs; a nearest-point
    (Voronoi) field is discontinuous and produces spurious, crumpled surfaces.

    Negative values are behind the surface (along ``-normal``).

    Returns:
        ``(field, origin)``.
    """
    dims = np.maximum(np.ceil((upper - lower) / voxel).astype(int) + 1, 2)
    n_voxels = int(np.prod(dims))
    if n_voxels > 200_000_000:
        raise PipelineError(f"Tile grid too large ({n_voxels} voxels); reduce mesh voxel or tile size")

    origin = lower.astype(np.float64)
    base = np.floor((points - origin) / voxel).astype(np.int64)
    np.clip(base, 0, dims - 1, out=base)

    trunc = float(trunc_voxels) * voxel
    radius = int(np.ceil(trunc_voxels))
    acc = np.zeros(n_voxels, np.float32)
    wacc = np.zeros(n_voxels, np.float32)

    offsets = [
        (dx, dy, dz)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
    ]
    for dx, dy, dz in offsets:
        ijk = base + np.array([dx, dy, dz], dtype=np.int64)
        inside = np.all((ijk >= 0) & (ijk < dims), axis=1)
        if not np.any(inside):
            continue
        ijk = ijk[inside]
        flat = ijk[:, 0] * (dims[1] * dims[2]) + ijk[:, 1] * dims[2] + ijk[:, 2]
        centres = origin[None, :] + ijk.astype(np.float64) * voxel
        sdf = np.sum(normals[inside] * (centres - points[inside]), axis=1)
        weight = np.clip(1.0 - np.abs(sdf) / trunc, 0.0, 1.0).astype(np.float32)
        np.add.at(acc, flat, weight * sdf.astype(np.float32))
        np.add.at(wacc, flat, weight)

    observed = (wacc > 0).reshape(tuple(dims))
    field = np.divide(
        acc,
        np.maximum(wacc, 1e-6),
        out=np.zeros(n_voxels, np.float32),
        where=(wacc > 0),
    )
    np.clip(field, -trunc, trunc, out=field)
    field = field.reshape(tuple(dims))

    # Unobserved voxels need a definite sign. Anything reachable from the tile
    # boundary is outside the object; anything enclosed by the observed band is
    # inside it. Without this the interior of a closed surface becomes "unknown",
    # which the mesher turns into a duplicate inner shell.
    labels, count = ndimage.label(~observed)
    if count:
        border = np.concatenate(
            [
                labels[0, :, :].ravel(),
                labels[-1, :, :].ravel(),
                labels[:, 0, :].ravel(),
                labels[:, -1, :].ravel(),
                labels[:, :, 0].ravel(),
                labels[:, :, -1].ravel(),
            ]
        )
        outside_labels = np.unique(border[border > 0])
        enclosed = ~observed & ~np.isin(labels, outside_labels)
        field = np.where(~observed, np.float32(trunc), field)
        field[enclosed] = -trunc

    return field.astype(np.float32), origin


def reconstruct_mesh(
    points: np.ndarray,
    normals: np.ndarray,
    voxel: float,
    tile_units: float,
    max_tile_voxels: int = 6_000_000,
) -> Mesh:
    """Reconstruct a triangle mesh from oriented points by tiled level-set extraction."""
    if points.shape[0] < 100:
        raise PipelineError(f"Only {points.shape[0]} points available; cannot reconstruct a mesh")

    lo = np.percentile(points, 0.2, axis=0)
    hi = np.percentile(points, 99.8, axis=0)
    extent = np.maximum(hi - lo, voxel * 4.0)

    # Tile size that respects the memory guard, capped so tiles stay local.
    max_side = (max_tile_voxels ** (1.0 / 3.0)) * voxel
    tile = float(tile_units) if tile_units > 0 else float(min(max_side, max(extent.max() / 4.0, 8.0 * voxel)))
    tile = float(min(tile, max_side))
    tile = float(max(tile, 4.0 * voxel))

    counts = np.maximum(np.ceil(extent / tile).astype(int), 1)
    margin = 3.0 * voxel

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for ix in range(counts[0]):
        for iy in range(counts[1]):
            for iz in range(counts[2]):
                lower = lo + np.array([ix, iy, iz]) * tile - margin
                upper = lower + tile + 2.0 * margin
                inside = np.all((points >= lower) & (points <= upper), axis=1)
                if int(inside.sum()) < 50:
                    continue
                sub_points = points[inside]
                sub_normals = normals[inside]
                field_grid, origin = signed_distance_tile(
                    sub_points, sub_normals, lower, upper, voxel
                )
                mesh = marching_tetrahedra(field_grid, origin, voxel)
                if mesh.num_faces == 0:
                    continue
                vertices.append(mesh.vertices)
                faces.append(mesh.faces + offset)
                offset += mesh.num_vertices

    if not vertices:
        raise PipelineError("Mesh reconstruction produced no triangles; check the fused point cloud")
    return Mesh(np.concatenate(vertices, axis=0), np.concatenate(faces, axis=0))


def weld_vertices(mesh: Mesh, tolerance: float) -> Mesh:
    """Merge duplicate vertices so the mesh is index-shared and file-size sane."""
    if mesh.num_faces == 0:
        return mesh
    quantised = np.round(mesh.vertices / max(tolerance, 1e-12)).astype(np.int64)
    _, first_index, inverse = np.unique(
        quantised, axis=0, return_index=True, return_inverse=True
    )
    faces = inverse[mesh.faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return Mesh(mesh.vertices[first_index], faces[keep])


def filter_unobserved_faces(mesh: Mesh, centres: np.ndarray, min_cos: float = 0.0) -> Mesh:
    """Drop faces that no camera ever sees from the front.

    A surface observed from one side only would otherwise be meshed as a thin
    double-sided shell, doubling the triangle count and the apparent area. Faces
    whose normal points away from every camera centre are the back half of that
    shell and carry no measurement.
    """
    if mesh.num_faces == 0 or centres.shape[0] == 0:
        return mesh

    tri = mesh.vertices[mesh.faces]
    centroids = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)

    visible = np.zeros(mesh.num_faces, dtype=bool)
    chunk = 100_000
    for start in range(0, mesh.num_faces, chunk):
        stop = min(start + chunk, mesh.num_faces)
        rays = centres[None, :, :] - centroids[start:stop, None, :]
        rays /= np.maximum(np.linalg.norm(rays, axis=2, keepdims=True), 1e-12)
        dots = np.einsum("fci,fi->fc", rays, normals[start:stop])
        visible[start:stop] = (dots > min_cos).any(axis=1)
    return Mesh(mesh.vertices, mesh.faces[visible])


def orient_faces(mesh: Mesh, points: np.ndarray, normals: np.ndarray) -> Mesh:
    """Flip triangles so their normals agree with the local point-cloud normals.

    Marching tetrahedra fixes winding per tetrahedron, not globally; this pass
    makes back-face culling behave.
    """
    if mesh.num_faces == 0 or points.shape[0] == 0:
        return mesh
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    _, idx = tree.query(centroids, k=1)
    corners = mesh.vertices[mesh.faces]
    tri_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    reference = normals[idx]
    flip = np.sum(tri_normals * reference, axis=1) < 0
    faces = mesh.faces.copy()
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return Mesh(mesh.vertices, faces)
