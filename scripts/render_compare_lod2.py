import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from uav3d.modeling.completion import _align_mesh_upright

def render_mesh_views(mesh_path: str, out_img: str, title: str, is_raw: bool = False):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if is_raw:
        mesh = _align_mesh_upright(mesh)
    if len(mesh.triangles) > 40000:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=40000)
    mesh.compute_vertex_normals()
    
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    cols = np.asarray(mesh.vertex_colors)
    if len(cols) == 0:
        cols = np.full((len(v), 3), 0.7)
        
    tri_pts = v[f]
    face_cols = np.clip(cols[f].mean(axis=1), 0.0, 1.0)
    
    fig = plt.figure(figsize=(14, 6))
    
    center = v.mean(axis=0)
    extent = v.max(axis=0) - v.min(axis=0)
    span = float(np.linalg.norm(extent))
    r = span * 0.45
    
    # View 1: Oblique overview
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    pc1 = Poly3DCollection(tri_pts, facecolors=face_cols, edgecolors='none')
    ax1.add_collection3d(pc1)
    ax1.set_xlim(center[0] - r, center[0] + r)
    ax1.set_ylim(center[1] - r, center[1] + r)
    ax1.set_zlim(center[2] - r * 0.3, center[2] + r * 0.7)
    ax1.set_box_aspect((1, 1, 0.5))
    ax1.view_init(elev=35, azim=-55)
    ax1.set_title(f"{title} (Oblique View)", fontsize=11, fontweight="bold")
    ax1.set_axis_off()
    
    # View 2: Side / End of Corridor Facade View
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    pc2 = Poly3DCollection(tri_pts, facecolors=face_cols, edgecolors='none')
    ax2.add_collection3d(pc2)
    ax2.set_xlim(center[0] - r, center[0] + r)
    ax2.set_ylim(center[1] - r, center[1] + r)
    ax2.set_zlim(center[2] - r * 0.3, center[2] + r * 0.7)
    ax2.set_box_aspect((1, 1, 0.5))
    ax2.view_init(elev=20, azim=40)
    ax2.set_title(f"{title} (Corridor Edge View)", fontsize=11, fontweight="bold")
    ax2.set_axis_off()
    
    fig.tight_layout()
    fig.savefig(out_img, dpi=130)
    plt.close(fig)
    print(f"Rendered {out_img}")

if __name__ == "__main__":
    render_mesh_views("data/outputs/run_20260924_163620/mesh.ply", "debug/render_original_upright.png", "Raw TSDF Mesh (Open Husks)", is_raw=True)
    render_mesh_views("debug/mesh_lod2_upright.ply", "debug/render_lod2_upright.png", "Completed LoD-2 Mesh (Extruded & Sealed)", is_raw=False)
