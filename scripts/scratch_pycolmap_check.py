"""Scratch script to inspect installed pycolmap API signatures and attributes."""

import contextlib

import pycolmap

print("=== pycolmap version ===")
print("Version:", pycolmap.__version__)

print("\n=== Top-level functions ===")
for fn_name in ["extract_features", "match_sequential", "incremental_mapping"]:
    if hasattr(pycolmap, fn_name):
        fn = getattr(pycolmap, fn_name)
        print(f"{fn_name}: {fn}")
        with contextlib.suppress(Exception):
            print(f"  doc: {fn.__doc__}")
    else:
        print(f"{fn_name}: NOT FOUND")

print("\n=== Reconstruction attributes & methods ===")
if hasattr(pycolmap, "Reconstruction"):
    print("Reconstruction found")
    rec = pycolmap.Reconstruction()
    print("Reconstruction attrs/methods:", [a for a in dir(rec) if not a.startswith("_")])

print("\n=== Image class ===")
if hasattr(pycolmap, "Image"):
    print("Image attrs/methods:", [a for a in dir(pycolmap.Image) if not a.startswith("_")])
    img = pycolmap.Image()
    print("Image default init:", img)
    print("Image has cam_from_world:", hasattr(img, "cam_from_world"))
    print("Image has qvec:", hasattr(img, "qvec"))
    print("Image has tvec:", hasattr(img, "tvec"))
    print("Image has has_pose:", hasattr(img, "has_pose"))
    print("Image has num_points2D:", hasattr(img, "num_points2D"))
    print("Image has num_points3D:", hasattr(img, "num_points3D"))

print("\n=== Camera class ===")
if hasattr(pycolmap, "Camera"):
    print("Camera attrs/methods:", [a for a in dir(pycolmap.Camera) if not a.startswith("_")])

print("\n=== Pipeline / Options classes ===")
opts = [a for a in dir(pycolmap) if "Option" in a or "Pipeline" in a or "Incremental" in a]
print("Pipeline/Options in pycolmap:", opts)

if hasattr(pycolmap, "IncrementalPipelineOptions"):
    ipo = pycolmap.IncrementalPipelineOptions()
    print("IncrementalPipelineOptions attrs:", [a for a in dir(ipo) if not a.startswith("_")])
    if hasattr(ipo, "ba_refine_focal_length"):
        print("ba_refine_focal_length:", ipo.ba_refine_focal_length)
    if hasattr(ipo, "ba_refine_principal_point"):
        print("ba_refine_principal_point:", ipo.ba_refine_principal_point)
    if hasattr(ipo, "ba_refine_extra_params"):
        print("ba_refine_extra_params:", ipo.ba_refine_extra_params)

if hasattr(pycolmap, "SiftExtractionOptions"):
    seo = pycolmap.SiftExtractionOptions()
    print("SiftExtractionOptions attrs:", [a for a in dir(seo) if not a.startswith("_")])

if hasattr(pycolmap, "SequentialMatchingOptions"):
    smo = pycolmap.SequentialMatchingOptions()
    print("SequentialMatchingOptions attrs:", [a for a in dir(smo) if not a.startswith("_")])

if hasattr(pycolmap, "Database"):
    print("Database class found!")
    db_attrs = [a for a in dir(pycolmap.Database) if not a.startswith("_")]
    print("Database attrs:", db_attrs)
