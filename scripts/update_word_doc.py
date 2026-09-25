import sys
sys.stdout.reconfigure(encoding='utf-8')

import docx
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT

# Start from clean backup
doc = docx.Document("CO_PO_PSO_SDG Mapping_backup.docx")

FONT_NAME = "Tahoma"

def format_run(run, bold=False, italic=False, size=Pt(11)):
    run.font.name = FONT_NAME
    run.font.size = size
    run.bold = bold
    run.italic = italic

def set_paragraph_content(p, runs_data, align=WD_ALIGN_PARAGRAPH.LEFT):
    """
    runs_data: list of tuples (text, is_bold, is_italic, font_size)
    """
    p.text = ""
    p.alignment = align
    for text, bold, italic, size in runs_data:
        r = p.add_run(text)
        format_run(r, bold=bold, italic=italic, size=size)

# P4: Project Title
set_paragraph_content(
    doc.paragraphs[4],
    [
        ("Project Title: ", True, False, Pt(11)),
        ("AI-Enabled Georeferenced and Metrically Accurate 3D Reconstruction from Single-Pass Drone Video (SIH 26158)", False, False, Pt(11))
    ]
)

# P5: Theme
set_paragraph_content(
    doc.paragraphs[5],
    [
        ("Theme: ", True, False, Pt(11)),
        ("Robotics and Drones", False, False, Pt(11))
    ]
)

# P6: Project Objective
set_paragraph_content(
    doc.paragraphs[6],
    [
        ("Project Objective: ", True, False, Pt(11)),
        (
            "To design and develop an automated, end-to-end computer vision and AI pipeline to reconstruct georeferenced "
            "and metrically accurate 3D models (dense point clouds, surface meshes, and digital twin assets) from "
            "single-pass aerial drone/UAV video footage, incorporating intelligent keyframe extraction, telemetry "
            "synchronization, incremental Structure-from-Motion (SfM), neural depth estimation, and volumetric TSDF fusion.",
            False, False, Pt(11)
        )
    ]
)

# P8: Deliverables Heading
set_paragraph_content(
    doc.paragraphs[8],
    [
        ("Project Deliverables (specific outcomes for the project):", True, False, Pt(11))
    ]
)

# P9: Deliverable 1
set_paragraph_content(
    doc.paragraphs[9],
    [
        ("1. ", True, False, Pt(11)),
        ("Automated Video Ingestion, Quality-Based Keyframing & Telemetry Synchronization: ", True, False, Pt(11)),
        (
            "Ingest single-pass drone video, evaluate frame sharpness and parallax, select optimal keyframes within "
            "bounded computational budgets, and synchronize frames with onboard GPS/IMU telemetry logs.",
            False, False, Pt(11)
        )
    ]
)

# P10: Deliverable 2
set_paragraph_content(
    doc.paragraphs[10],
    [
        ("2. ", True, False, Pt(11)),
        ("Structure-from-Motion Pose Estimation & Metric Trajectory Validation: ", True, False, Pt(11)),
        (
            "Estimate accurate 6-DoF camera poses and generate sparse 3D point clouds using incremental pycolmap SfM, "
            "with metric similarity alignment and Absolute Trajectory Error (ATE) validation against ground-truth coordinates.",
            False, False, Pt(11)
        )
    ]
)

# P11: Deliverable 3
set_paragraph_content(
    doc.paragraphs[11],
    [
        ("3. ", True, False, Pt(11)),
        ("AI Monocular Depth Fusion, 3D Mesh Generation & Interactive Visualization: ", True, False, Pt(11)),
        (
            "Estimate high-resolution dense depth using Depth Anything V2, integrate multi-view frames into ScalableTSDF "
            "volumetric meshes, produce watertight textured 3D models (PLY/OBJ/GLB), and provide interactive visual inspection dashboards.",
            False, False, Pt(11)
        )
    ]
)

# P12: Modern tools used
set_paragraph_content(
    doc.paragraphs[12],
    [
        ("Modern tools used: ", True, False, Pt(11)),
        (
            "Python 3.12, PyCOLMAP / COLMAP, OpenCV, Open3D, ONNX Runtime (Depth Anything V2), NumPy, SciPy, Trimesh, Dash / Plotly, uv, Git & GitHub.",
            False, False, Pt(11)
        )
    ]
)

# P14: Table Section Header
set_paragraph_content(
    doc.paragraphs[14],
    [
        ("PO-PSO and SDG Mapping", True, False, Pt(11))
    ]
)

def format_cell(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=Pt(10.5)):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if text:
        r = p.add_run(text)
        format_run(r, bold=bold, size=size)

# TABLE 0: PO Mapping
# Columns: [Deliverables/Outcomes, PO1, PO2, PO3, PO4, PO5, PO6, PO7, PO8, PO9, PO10, PO11]
t0 = doc.tables[0]
d1_po = ["1", "✓", "✓", "✓", "✓", "✓", "", "✓", "✓", "✓", "✓", "✓"]
d2_po = ["2", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "✓"]
d3_po = ["3", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "✓"]

for col_idx, val in enumerate(d1_po):
    format_cell(t0.rows[1].cells[col_idx], val, bold=(col_idx == 0))

for col_idx, val in enumerate(d2_po):
    format_cell(t0.rows[2].cells[col_idx], val, bold=(col_idx == 0))

for col_idx, val in enumerate(d3_po):
    format_cell(t0.rows[3].cells[col_idx], val, bold=(col_idx == 0))

# TABLE 1: PSO Mapping
# Columns: [Outcomes, PSO1, PSO2, PSO3]
t1 = doc.tables[1]
for r_idx in range(1, 4):
    format_cell(t1.rows[r_idx].cells[0], str(r_idx), bold=True)
    format_cell(t1.rows[r_idx].cells[1], "✓")
    format_cell(t1.rows[r_idx].cells[2], "✓")
    format_cell(t1.rows[r_idx].cells[3], "✓")

# TABLE 2: SDG Mapping
# Row 0: ['', 'SDG GOALS']
# Row 1: ['Project Title', '1', '2', ..., '17']
# Row 2: [Title, '✓' under mapped goals]
t2 = doc.tables[2]

# Project Title in Table 2 Row 2 Col 0
format_cell(
    t2.rows[2].cells[0],
    "AI-Enabled Georeferenced and Metrically Accurate 3D Reconstruction from Single-Pass Drone Video",
    bold=False,
    align=WD_ALIGN_PARAGRAPH.LEFT,
    size=Pt(9)
)

# Mapped SDG goals:
# Goal 9: Industry, Innovation and Infrastructure
# Goal 11: Sustainable Cities and Economies
# Goal 13: Climate Action (rapid aerial damage assessment & environmental monitoring)
mapped_sdgs = {9, 11, 13}

for col_idx in range(1, 18):
    if col_idx in mapped_sdgs:
        format_cell(t2.rows[2].cells[col_idx], "✓", bold=True, size=Pt(10.5))
    else:
        format_cell(t2.rows[2].cells[col_idx], "", bold=False)

# Save document
doc.save("CO_PO_PSO_SDG Mapping.docx")
print("Updated CO_PO_PSO_SDG Mapping.docx successfully.")
