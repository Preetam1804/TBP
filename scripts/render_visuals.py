"""Render textured 3D model viewpoints using Open3D OffscreenRenderer with unlit photographic vibrancy."""

from pathlib import Path
import numpy as np
import open3d as o3d

def render_views():
    model_obj = Path("data/outputs/run_20260921_130415/map_3d.obj")
    artifact_dir = Path(r"C:\Users\Preetam\.gemini\antigravity-ide\brain\0280f994-1ca4-4294-972c-67d314205298")
    
    # Read mesh
    mesh = o3d.io.read_triangle_model(str(model_obj))
    print(f"Loaded TriangleModel with {len(mesh.meshes)} submeshes, {len(mesh.materials)} materials")

    # Use unlit shader to display pure high-res texture colors
    for m in mesh.materials:
        m.shader = "defaultUnlit"

    width, height = 1920, 1080
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([0.05, 0.07, 0.10, 1.0])  # Sleek dark slate background

    scene.add_model("map_3d", mesh)

    bbox = scene.bounding_box
    center = bbox.get_center()
    extent = bbox.get_extent()
    print(f"BBox center: {center}, extent: {extent}")

    views = [
        {
            "name": "lod2_oblique_aerial",
            # Elevated oblique view looking down at the monument and grounds
            "eye": center + np.array([0.0, -extent[1] * 1.5, extent[2] * 2.2 + 25.0]),
            "lookat": center + np.array([0.0, 0.0, 3.0]),
            "up": np.array([0.0, 0.0, 1.0]),
        },
        {
            "name": "lod2_facade_closeup",
            # Frontal approach view showing monument facade and minarets
            "eye": center + np.array([0.0, -extent[1] * 0.95, 8.0]),
            "lookat": center + np.array([0.0, extent[1] * 0.1, 4.0]),
            "up": np.array([0.0, 0.0, 1.0]),
        },
        {
            "name": "lod2_survey_overview",
            # Oblique corner overview showing both facade and top-down garden
            "eye": center + np.array([extent[0] * 1.1, -extent[1] * 1.3, extent[2] * 2.0 + 15.0]),
            "lookat": center + np.array([0.0, 0.0, 2.0]),
            "up": np.array([0.0, 0.0, 1.0]),
        },
    ]

    for v in views:
        renderer.setup_camera(55.0, center, v["eye"], v["up"])
        renderer.scene.camera.look_at(v["lookat"], v["eye"], v["up"])
        img = renderer.render_to_image()
        out_path = artifact_dir / f"{v['name']}.png"
        o3d.io.write_image(str(out_path), img)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    render_views()
