"""Run the full pipeline on one video file and report what came out.

One command, from an mp4 to a scored 3-D model:

    inspect -> keyframes -> sfm -> dense [-> validate when GT exists]

Written because a video run spans four subcommands and three run-directory
path hand-offs, and getting any of them wrong degrades the result silently
rather than loudly. The stages are the CLI's own handlers, not a reimplementation,
so anything this script produces can be reproduced by hand with the documented
commands.

Reports the numbers that actually decide whether to trust the model --
registration rate, reprojection error, aspect ratio, and the two independent
monocular scale estimates -- instead of just "finished".

Usage:
    python scripts/run_video_pipeline.py --video data/city_flyover.mp4
    python scripts/run_video_pipeline.py --video data/city_flyover.mp4 --preview
    python scripts/run_video_pipeline.py --video data/city_flyover.mp4 --stage dense
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402

from uav3d.cli import main  # noqa: E402
from uav3d.dense.coverage import coverage_gate_ok  # noqa: E402
from uav3d.dense.evaluate import load_ply_points  # noqa: E402
from uav3d.models.calibration_io import read_calibration  # noqa: E402
from uav3d.sfm.focal_gate import (  # noqa: E402
    check_focal_drift,
    describe_focal_source,
    solved_focal_px,
)


def _host_fs_path(path: Path) -> Path:
    """Map a repo-relative video path to the path this process can actually open."""
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def probe_video(video: Path) -> dict:
    """Read container properties before spending minutes on feature extraction."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    info = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    info["seconds"] = info["frames"] / info["fps"] if info["fps"] > 0 else float("nan")
    return info


def newest_run_dir() -> Path | None:
    runs = sorted((REPO_ROOT / "data" / "outputs").glob("run_*"))
    return runs[-1] if runs else None


def stage_keyframes(video: Path, config: Path, start_frame: int = 0, end_frame: int | None = None,
                    focal_px: float | None = None) -> Path:
    argv = ["keyframes", "--video", str(video), "--config", str(config)]
    if start_frame:
        argv += ["--start-frame", str(start_frame)]
    if end_frame is not None:
        argv += ["--end-frame", str(end_frame)]
    if focal_px is not None:
        argv += ["--focal-px", str(focal_px)]
    code = main(argv)
    if code != 0:
        raise SystemExit(f"keyframes stage failed with exit code {code}")
    run_dir = newest_run_dir()
    if run_dir is None:
        raise SystemExit("keyframes produced no run directory")
    return run_dir


def stage_sfm(run_dir: Path, config: Path, focal_px: float | None = None) -> dict:
    argv = ["sfm", "--config", str(config), "--run-dir", str(run_dir)]
    if focal_px is not None:
        argv += ["--focal-px", str(focal_px)]
    code = main(argv)
    if code != 0:
        raise SystemExit(f"sfm stage failed with exit code {code}")
    poses = json.loads((run_dir / "poses.json").read_text())
    return poses


def stage_dense(run_dir: Path, config: Path, max_width: int, ref_stride: int,
                mesh: bool, prior: bool = False) -> dict:
    argv = [
        "dense",
        "--config", str(config),
        "--run-dir", str(run_dir),
        "--max-width", str(max_width),
        "--ref-stride", str(ref_stride),
    ]
    if not mesh:
        argv.append("--no-mesh")
    if prior:
        argv.append("--prior")
    code = main(argv)
    if code != 0:
        raise SystemExit(f"dense stage failed with exit code {code}")
    return json.loads((run_dir / "dense" / "dense_report.json").read_text())


def summarise(video: Path, info: dict, run_dir: Path, poses: dict | None, dense: dict | None) -> bool:
    """Print the trustworthiness report. Returns True if the run is usable."""
    ok = True
    print("\n" + "=" * 72)
    print(f"VIDEO     {video.name}")
    print(f"          {info['width']}x{info['height']}, {info['frames']} frames, "
          f"{info['fps']:.2f} fps, {info['seconds']:.1f} s")
    print(f"RUN       {run_dir}")

    # poses.json carries the camera that SfM actually solved with (post-bundle-
    # adjustment), which is the one that matters. calibration.json is only the
    # input prior written by the keyframes stage, before the video policy pins a
    # single shared focal.
    stored = read_calibration(run_dir)
    solved = (poses or {}).get("camera")
    if solved:
        model = solved["model"]
        params = ", ".join(f"{v:.2f}" for v in solved["params"])
        if model.upper().endswith("SIMPLE_PINHOLE"):
            aspect = "1.000 (single focal)"
        else:
            aspect = f"{solved['params'][0] / solved['params'][1]:.3f}"
        print(f"CAMERA    {model} [{params}] solved  aspect {aspect}")
        if not model.upper().endswith("SIMPLE_PINHOLE"):
            ratio = solved["params"][0] / solved["params"][1]
            if abs(ratio - 1.0) > 0.05:
                print(f"  WARNING  aspect ratio {ratio:.3f} is not physical; "
                      "shape will be stretched along one image axis")
                ok = False
        # Focal-drift gate (Phase 2): forward flight trades focal length
        # against depth, so a solved focal far from the keyframes-stage prior
        # usually means a depth stretch was absorbed into the camera. The
        # gate flags it; it never clamps the focal.
        if stored is not None:
            print(f"          prior focal from {describe_focal_source(stored.source)} "
                  f"({stored.source})")
            try:
                gate = check_focal_drift(
                    solved_focal_px(stored.model, list(stored.params)),
                    solved_focal_px(model, list(solved["params"])),
                    max_rel_drift=0.10,
                )
                print(f"          focal drift {gate.message}")
                if gate.exceeded:
                    print("  WARNING  focal drift exceeds the 10% gate; the model "
                          "may be stretched along the flight axis — consider "
                          "re-running with refine_intrinsics=false")
                    ok = False
            except ValueError as exc:
                print(f"  WARNING  focal-drift gate could not be evaluated: {exc}")
    elif stored is not None:
        params = ", ".join(f"{v:.2f}" for v in stored.params)
        aspect = (
            "1.000 (single focal)"
            if stored.model.upper().endswith("SIMPLE_PINHOLE")
            else f"{stored.params[0] / stored.params[1]:.3f}"
        )
        print(f"CAMERA    {stored.model} [{params}]  aspect {aspect}")
        print(f"          prior focal from {describe_focal_source(stored.source)} "
              f"({stored.source})")
        if not stored.model.upper().endswith("SIMPLE_PINHOLE"):
            ratio = stored.params[0] / stored.params[1]
            if abs(ratio - 1.0) > 0.05:
                print(f"  WARNING  aspect ratio {ratio:.3f} is not physical; "
                      "shape will be stretched along one image axis")
                ok = False

    if poses is not None:
        metrics = poses.get("metrics") or {}
        registered = metrics.get("num_images_registered")
        total = metrics.get("num_images_total")
        reproj = metrics.get("mean_reprojection_error_px")
        points = metrics.get("num_points3D")
        if total:
            rate = 100.0 * (registered or 0) / total
            print(f"SfM       {registered}/{total} frames registered ({rate:.1f}%), "
                  f"reproj {reproj:.3f} px, {points:,} sparse points")
            if rate < 85.0:
                print(f"  WARNING  registration {rate:.1f}% is below the 85% gate; "
                      "SfM likely mis-registered part of the flight")
                ok = False
            if reproj is not None and reproj > 1.5:
                print(f"  WARNING  reprojection {reproj:.3f} px exceeds the 1.5 px gate")
                ok = False

    if dense is not None:
        geometry = dense.get("geometry") or {}
        depth = dense.get("depth") or {}
        fusion = dense.get("fusion") or {}
        scale = dense.get("scale") or {}
        print(f"DENSE     {geometry.get('num_registered_views')} views, "
              f"{geometry.get('num_reference_views')} reference, "
              f"working {geometry.get('working_width')}x{geometry.get('working_height')}")
        print(f"          depth valid {100.0 * (depth.get('valid_fraction') or 0):.1f}% "
              f"of pixels, median cost {depth.get('median_cost')}")
        fused = fusion.get("num_points")
        voxel = fusion.get("voxel_units")
        spacing = fusion.get("point_spacing_meters")
        detail = f"voxel {voxel:.4f} units" if isinstance(voxel, (int, float)) else "voxel n/a"
        if isinstance(spacing, (int, float)):
            detail += f", spacing {spacing:.4f} m"
        print(f"          fused {fused:,} points, {detail}")
        print(f"SCALE     {scale.get('status')} (source: {scale.get('source')})")
        if scale.get("status") == "unverified_monocular_scale":
            print("          Units are SfM units, not metres. Nothing in monocular")
            print("          footage fixes a scale; treat distances as ratios.")
        valid = depth.get("valid_fraction") or 0.0
        if valid < 0.20:
            print(f"  WARNING  only {100.0 * valid:.1f}% of pixels produced depth; "
                  "the scene may be too textureless or the baseline too small")
            ok = False

        coverage = dense.get("coverage") or {}
        if coverage:
            summary = coverage.get("per_view_summary") or {}
            fused = coverage.get("fused") or {}
            grid = coverage.get("grid") or {}
            unit = coverage.get("unit") or "SfM units"
            area_unit = "m^2" if unit == "m" else "SfM unit^2"
            lo = 100.0 * (summary.get("min_valid_fraction") or 0.0)
            hi = 100.0 * (summary.get("max_valid_fraction") or 0.0)
            med = 100.0 * (summary.get("median_valid_fraction") or 0.0)
            provenance = (
                "final maps include prior-filled (inferred) pixels"
                if (coverage.get("prior") or {}).get("enabled")
                else "all valid pixels are measured"
            )
            print(f"COVERAGE  per-view valid {lo:.1f}%-{hi:.1f}% "
                  f"(median {med:.1f}% over {summary.get('num_views')} views); {provenance}")
            print(f"          fused density {fused.get('points_per_unit_area', 0):,.0f} "
                  f"points/{area_unit} over {fused.get('footprint_area', 0):,.1f} {area_unit}")
            print(f"          grid {grid.get('empty_cells')} of {grid.get('num_cells')} "
                  f"footprint cells empty ({100.0 * (grid.get('empty_fraction') or 0):.1f}%), "
                  f"{grid.get('thin_cells')} thin ({100.0 * (grid.get('thin_fraction') or 0):.1f}%)")
            print(f"          verdict {coverage.get('verdict')}")
            if not coverage_gate_ok(coverage):
                print(f"  WARNING  coverage verdict is {coverage.get('verdict')!r}: "
                      + "; ".join(coverage.get("verdict_reasons") or [])
                      + " — the model has thin or missing regions; see coverage_report.json")
                ok = False
        else:
            print("COVERAGE  no coverage report (run predates Phase 4 coverage reporting)")

    cloud = run_dir / "dense" / "cloud.ply"
    if cloud.exists():
        try:
            points = load_ply_points(cloud)
            print(f"CLOUD     {len(points):,} points -> {cloud}")
        except Exception as exc:  # noqa: BLE001 - reporting only
            print(f"CLOUD     could not read back {cloud}: {exc}")

    print("=" * 72)
    print("RESULT    " + ("usable" if ok else "completed WITH WARNINGS (see above)"))
    return ok


def main_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("data/city_flyover.mp4"))
    parser.add_argument("--config", type=Path, default=Path("configs/video_flyover.yaml"))
    parser.add_argument("--max-width", type=int, default=960,
                        help="Dense working width; 960 is the fast preview, 1600 the full run")
    parser.add_argument("--ref-stride", type=int, default=2)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First video frame to ingest (0-based, inclusive)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last video frame to ingest (0-based, exclusive)",
    )
    parser.add_argument("--mesh", action="store_true", help="Also build the visualisation mesh")
    parser.add_argument(
        "--focal-px",
        type=float,
        default=None,
        help=(
            "Known focal length in pixels (fx = fy). Overrides config "
            "source.focal_px and any <video-stem>.meta.json sidecar; without "
            "it the sidecar wins over the 0.8*width guess. Must be within "
            "0.2*width..5*width."
        ),
    )
    parser.add_argument("--prior", action="store_true",
                        help="Fill MVS-invalid pixels with a monocular depth prior (default off)")
    parser.add_argument("--preview", action="store_true",
                        help="Quick pass at 640 px to prove the chain works before the full run")
    parser.add_argument("--stage", choices=("keyframes", "sfm", "dense", "all"), default="all",
                        help="Stop after a given stage, or reuse earlier stages")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Reuse an existing run directory instead of creating one")
    args = parser.parse_args()

    video = _host_fs_path(args.video)
    config = _host_fs_path(args.config)
    if not video.exists():
        raise SystemExit(
            f"Video not found: {video}\n"
            "Put the file at data/city_flyover.mp4 (or pass --video PATH)."
        )
    if not config.exists():
        raise SystemExit(f"Config not found: {config}")

    if args.preview:
        args.max_width = 640
        args.ref_stride = 3

    info = probe_video(video)
    print(f"Video: {video}")
    print(f"  {info['width']}x{info['height']}, {info['frames']} frames, "
          f"{info['fps']:.2f} fps, {info['seconds']:.1f} s")
    if args.start_frame or args.end_frame is not None:
        end = args.end_frame if args.end_frame is not None else info["frames"]
        print(f"  using frame range {args.start_frame}..{end - 1} "
              f"({end - args.start_frame} frames)")
    if info["frames"] == 0 or info["width"] == 0:
        raise SystemExit("Video reports zero frames; the file may be corrupt or unsupported")

    started = time.time()
    run_dir = args.run_dir
    poses = None
    dense = None

    if args.stage in ("keyframes", "all") and run_dir is None:
        print("\n--- keyframes ---")
        run_dir = stage_keyframes(video, config, args.start_frame, args.end_frame,
                                  focal_px=args.focal_px)
    elif run_dir is None:
        run_dir = newest_run_dir()
        if run_dir is None:
            raise SystemExit("No existing run directory; run without --stage first")

    if args.stage in ("sfm", "all"):
        print("\n--- sfm ---")
        poses = stage_sfm(run_dir, config, focal_px=args.focal_px)

    if args.stage in ("dense", "all"):
        print("\n--- dense ---")
        dense = stage_dense(run_dir, config, args.max_width, args.ref_stride,
                            args.mesh, prior=args.prior)
        if poses is None:
            poses_path = run_dir / "poses.json"
            poses = json.loads(poses_path.read_text()) if poses_path.exists() else None

    summarise(video, info, run_dir, poses, dense)
    print(f"Wall time: {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
