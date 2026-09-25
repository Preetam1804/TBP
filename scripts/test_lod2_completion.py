import trimesh
import open3d as o3d
import numpy as np
from pathlib import Path
from uav3d.modeling.completion import _align_mesh_upright

def complete_mesh_lod2(input_ply: str, output_ply: str):
    # 1. Load with open3d and align upright so ground plane is at z=0
    raw_o3d = o3d.io.read_triangle_mesh(input_ply)
    upright_o3d = _align_mesh_upright(raw_o3d)
    
    verts = np.asarray(upright_o3d.vertices, dtype=np.float64)
    faces = np.asarray(upright_o3d.triangles, dtype=np.int64)
    cols = np.asarray(upright_o3d.vertex_colors, dtype=np.float64)
    if len(cols) == 0:
        cols = np.full((len(verts), 3), 0.7)
    colors = (np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    # In upright frame, ground plane is centered near z = 0
    # Let's find ground datum from lowest 10% of vertices
    z = verts[:, 2]
    z_ground = float(np.percentile(z, 5))
    print(f"Upright ground z: {z_ground:.2f}m (span: {z.min():.2f} to {z.max():.2f})")
    
    # 2. Extract directed boundary edges
    directed_edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])
    sorted_edges = np.sort(directed_edges, axis=1)
    unq, idx_unq, inv, counts = np.unique(sorted_edges, axis=0, return_index=True, return_inverse=True, return_counts=True)
    boundary_mask = (counts[inv] == 1)
    boundary_directed = directed_edges[boundary_mask]
    
    # 3. Filter boundary edges that are elevated above ground
    v0_idx = boundary_directed[:, 0]
    v1_idx = boundary_directed[:, 1]
    z0 = verts[v0_idx, 2]
    z1 = verts[v1_idx, 2]
    
    elev_thresh = z_ground + 1.2
    elevated_mask = (z0 > elev_thresh) | (z1 > elev_thresh)
    extrude_edges = boundary_directed[elevated_mask]
    print(f"Elevated boundary edges to extrude: {len(extrude_edges)}")
    
    orig_n_verts = len(verts)
    new_verts = []
    new_colors = []
    new_faces = []
    
    for edge in extrude_edges:
        i_a = edge[0]
        i_b = edge[1]
        p_a = verts[i_a]
        p_b = verts[i_b]
        c_a = colors[i_a]
        c_b = colors[i_b]
        
        # Ground points straight down along gravity (Z axis)
        p_a_g = np.array([p_a[0], p_a[1], z_ground], dtype=np.float64)
        p_b_g = np.array([p_b[0], p_b[1], z_ground], dtype=np.float64)
        
        # Architectural facade shading for extruded walls
        wall_c_a = np.clip(c_a.astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
        wall_c_b = np.clip(c_b.astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
        
        idx_a_g = orig_n_verts + len(new_verts)
        new_verts.append(p_a_g)
        new_colors.append(wall_c_a)
        
        idx_b_g = orig_n_verts + len(new_verts)
        new_verts.append(p_b_g)
        new_colors.append(wall_c_b)
        
        new_faces.append([i_a, i_b, idx_b_g])
        new_faces.append([i_a, idx_b_g, idx_a_g])
        
    all_verts = np.vstack([verts, np.array(new_verts)])
    all_colors = np.vstack([colors, np.array(new_colors)])
    all_faces = np.vstack([faces, np.array(new_faces)])
    
    # Save as Open3D mesh
    res_mesh = o3d.geometry.TriangleMesh()
    res_mesh.vertices = o3d.utility.Vector3dVector(all_verts)
    res_mesh.triangles = o3d.utility.Vector3iVector(all_faces)
    res_mesh.vertex_colors = o3d.utility.Vector3dVector(all_colors[:, :3] / 255.0)
    res_mesh.compute_vertex_normals()
    
    o3d.io.write_triangle_mesh(output_ply, res_mesh)
    print(f"Saved completed mesh to {output_ply}: {len(all_verts)} verts, {len(all_faces)} faces")

if __name__ == "__main__":
    complete_mesh_lod2("data/outputs/run_20260924_163620/mesh.ply", "debug/mesh_lod2_upright.ply")
