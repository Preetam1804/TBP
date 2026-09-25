import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from uav3d.modeling.completion import _align_mesh_upright
import cv2
from uav3d.modeling.lod2 import _build_prism

# Load original mesh and upright
mesh = o3d.io.read_triangle_mesh('data/outputs/run_20260924_163620/mesh.ply')
mesh = _align_mesh_upright(mesh)
v = np.asarray(mesh.vertices)
f = np.asarray(mesh.triangles)
cols = np.asarray(mesh.vertex_colors)
z = v[:, 2]
z_ground = float(np.percentile(z, 5))

# Separate ground terrain (z <= z_ground + 0.5m)
terr_mask = np.all(z[f] <= z_ground + 0.5, axis=1)
v_terr = v
f_terr = f[terr_mask]
c_terr = cols

# Elevated triangles
tri_elev = np.any(z[f] > z_ground + 0.8, axis=1)
struct_submesh = o3d.geometry.TriangleMesh()
struct_submesh.vertices = mesh.vertices
struct_submesh.vertex_colors = mesh.vertex_colors
struct_submesh.triangles = o3d.utility.Vector3iVector(f[tri_elev])
struct_submesh.remove_unreferenced_vertices()

tri_clusters, cluster_sizes, _ = struct_submesh.cluster_connected_triangles()
tri_c = np.asarray(tri_clusters)
sizes = np.asarray(cluster_sizes)

all_v = [v[f_terr].reshape(-1, 3)]
all_f = [np.arange(len(all_v[0])).reshape(-1, 3)]
all_c = [c_terr[f_terr].reshape(-1, 3)]
v_offset = len(all_v[0])

sub_v = np.asarray(struct_submesh.vertices)
sub_f = np.asarray(struct_submesh.triangles)
sub_cols = np.asarray(struct_submesh.vertex_colors)

for c_id, sz in enumerate(sizes):
    if sz < 15:
        continue
    c_tri_mask = (tri_c == c_id)
    c_vert_indices = np.unique(sub_f[c_tri_mask].ravel())
    c_pts = sub_v[c_vert_indices]
    c_col = np.mean(sub_cols[c_vert_indices], axis=0) if len(sub_cols) else np.array([0.7, 0.7, 0.7])
    
    pts_2d = c_pts[:, :2].astype(np.float32)
    (cx, cy), (w, d), angle = cv2.minAreaRect(pts_2d)
    if min(w, d) < 0.8:
        continue
        
    w_est = max(w, d)
    d_est = max(min(w, d), 0.7 * w_est)  # Complete missing depth
    
    rect_pts = cv2.boxPoints(((cx, cy), (w_est, d_est), angle))
    z_roof = float(np.percentile(c_pts[:, 2], 90))
    z_peak = float(np.percentile(c_pts[:, 2], 99))
    
    vp, cp, tp = _build_prism(rect_pts, z_ground, z_roof, z_peak=z_peak if (z_peak - z_roof > 0.4) else None, color=c_col)
    all_v.append(vp)
    all_c.append(cp)
    all_f.append(np.array([[idx + v_offset for idx in tri] for tri in tp]))
    v_offset += len(vp)

final_v = np.concatenate(all_v, axis=0)
final_c = np.concatenate(all_c, axis=0)
final_f = np.concatenate(all_f, axis=0)

full_model = o3d.geometry.TriangleMesh()
full_model.vertices = o3d.utility.Vector3dVector(final_v)
full_model.vertex_colors = o3d.utility.Vector3dVector(np.clip(final_c, 0.0, 1.0))
full_model.triangles = o3d.utility.Vector3iVector(final_f)
full_model.compute_vertex_normals()
o3d.io.write_triangle_mesh('debug/complete_city_model.ply', full_model)
print(f'Wrote debug/complete_city_model.ply with {len(final_v)} verts, {len(final_f)} faces')

# Render
fig = plt.figure(figsize=(14, 7))
tri_pts = final_v[final_f]
face_cols = np.clip(final_c[final_f].mean(axis=1), 0.0, 1.0)
center = final_v.mean(axis=0)
extent = final_v.max(axis=0) - final_v.min(axis=0)
span = float(np.linalg.norm(extent))
r = span * 0.45

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
pc1 = Poly3DCollection(tri_pts, facecolors=face_cols, edgecolors='none')
ax1.add_collection3d(pc1)
ax1.set_xlim(center[0] - r, center[0] + r)
ax1.set_ylim(center[1] - r, center[1] + r)
ax1.set_zlim(center[2] - r * 0.3, center[2] + r * 0.7)
ax1.set_box_aspect((1, 1, 0.5))
ax1.view_init(elev=35, azim=-55)
ax1.set_title("Estimated Full 3D City (Complete Buildings)", fontsize=11, fontweight="bold")
ax1.set_axis_off()

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
pc2 = Poly3DCollection(tri_pts, facecolors=face_cols, edgecolors='none')
ax2.add_collection3d(pc2)
ax2.set_xlim(center[0] - r, center[0] + r)
ax2.set_ylim(center[1] - r, center[1] + r)
ax2.set_zlim(center[2] - r * 0.3, center[2] + r * 0.7)
ax2.set_box_aspect((1, 1, 0.5))
ax2.view_init(elev=20, azim=40)
ax2.set_title("Corridor Boundary View (No Sliced Husks)", fontsize=11, fontweight="bold")
ax2.set_axis_off()

plt.tight_layout()
plt.savefig('debug/render_complete_city.png', dpi=140)
plt.close()
print("Rendered debug/render_complete_city.png")
