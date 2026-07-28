#!/usr/bin/env python
"""Add observable mask+depth target xyz to schema-v2 expert sequences."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
for value in (REPO_ROOT, LIBERO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "baseline_libero_config")
)

from recovery.data.dataset import read_jsonl, write_jsonl
from recovery.data.visual_geometry import estimate_target_xyz


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-audit", type=Path, required=True)
    parser.add_argument("--min-visible-pixels", type=int, default=32)
    parser.add_argument("--max-jump-m", type=float, default=0.12)
    return parser.parse_args()


def camera_parameters() -> tuple[np.ndarray, float, float]:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(9)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
    )
    try:
        env.reset()
        sim = env.env.sim
        camera_to_world = np.linalg.inv(
            get_camera_transform_matrix(sim, "agentview", 256, 256)
        )
        extent = sim.model.stat.extent
        far = float(sim.model.vis.map.zfar * extent)
        near = float(sim.model.vis.map.znear * extent)
    finally:
        env.close()
    return camera_to_world, near, far


def main() -> None:
    args = arguments()
    calibration = json.loads(args.calibration_audit.read_text())
    bias = np.asarray(
        calibration["constant_bias_calibration"]["bias_estimate_minus_teacher_m"],
        dtype=np.float64,
    )
    camera_to_world, near, far = camera_parameters()
    source_paths = sorted(
        args.input_dir.glob("task*-episode*/transitions-v2.jsonl")
    )
    if not source_paths:
        raise ValueError(f"no schema-v2 episodes found in {args.input_dir}")

    errors = []
    unavailable = []
    episode_records = []
    teacher_errors = []
    gated_teacher_errors = []
    rejected_teacher_errors = []
    total = 0
    for source in source_paths:
        rows = read_jsonl(source)
        episode_errors = []
        episode_gated_errors = []
        last_reliable_xyz = None
        for row in rows:
            depth = row.online_inputs.get("depth")
            mask = row.online_inputs.get("target_mask")
            try:
                if depth is None or mask is None:
                    raise ValueError("depth or target mask is unavailable")
                xyz = estimate_target_xyz(
                    depth,
                    mask,
                    camera_to_world=camera_to_world,
                    near=near,
                    far=far,
                    calibration_bias=bias,
                )
                measurement = row.online_inputs.get("target_mask_measurement")
                measurement = measurement if isinstance(measurement, dict) else {}
                visible_pixels = measurement.get("visible_pixels")
                enough_pixels = (
                    visible_pixels is not None
                    and int(visible_pixels) >= args.min_visible_pixels
                )
                jump_m = (
                    float(np.linalg.norm(xyz - last_reliable_xyz))
                    if last_reliable_xyz is not None else 0.0
                )
                jump_safe = jump_m <= args.max_jump_m
                observable_valid = bool(enough_pixels and jump_safe)
                row.online_inputs["visual_target_xyz"] = xyz.tolist()
                row.online_inputs["visual_target_xyz_valid"] = observable_valid
                validity = dict(row.online_inputs.get("observation_validity") or {})
                validity["visual_target_xyz"] = observable_valid
                validity["visual_target_xyz_reason"] = (
                    "valid"
                    if observable_valid else
                    "insufficient_visible_pixels"
                    if not enough_pixels else
                    "temporal_jump"
                )
                validity["visual_target_xyz_jump_m"] = jump_m
                row.online_inputs["observation_validity"] = validity
                if observable_valid:
                    last_reliable_xyz = xyz
                teacher = (
                    row.teacher_only.get("raw_privileged_object_state") or {}
                ).get("akita_black_bowl_1_pos")
                if teacher is not None:
                    error = float(
                        np.linalg.norm(xyz - np.asarray(teacher, dtype=np.float32))
                    )
                    teacher_errors.append(error)
                    episode_errors.append(error)
                    if observable_valid:
                        gated_teacher_errors.append(error)
                        episode_gated_errors.append(error)
                    else:
                        rejected_teacher_errors.append(error)
            except Exception as error:
                message = str(error)
                record = {
                    "episode_id": row.episode_id,
                    "timestep": row.timestep,
                    "reason": message,
                }
                if (
                    message.startswith("empty target mask:")
                    or message == "depth or target mask is unavailable"
                ):
                    validity = dict(
                        row.online_inputs.get("observation_validity") or {}
                    )
                    validity["visual_target_xyz"] = False
                    validity["visual_target_xyz_reason"] = (
                        "empty_target_mask"
                        if message.startswith("empty target mask:")
                        else "missing_depth_or_mask"
                    )
                    row.online_inputs["observation_validity"] = validity
                    unavailable.append(record)
                else:
                    errors.append({**record, "error": message})
            total += 1
        relative = source.relative_to(args.input_dir)
        destination = args.output_dir / relative
        write_jsonl(destination, rows)
        episode_records.append(
            {
                "episode_id": rows[0].episode_id,
                "split": rows[0].split,
                "num_rows": len(rows),
                "num_valid_visual_xyz": sum(
                    bool(row.online_inputs.get("visual_target_xyz_valid"))
                    for row in rows
                ),
                "teacher_mean_error_cm": (
                    float(100 * np.mean(episode_errors))
                    if episode_errors else None
                ),
                "gated_teacher_mean_error_cm": (
                    float(100 * np.mean(episode_gated_errors))
                    if episode_gated_errors else None
                ),
                "output": str(destination.resolve()),
            }
        )

    report = {
        "schema_version": 1,
        "kind": "observable_visual_xyz_enrichment",
        "source": str(args.input_dir.resolve()),
        "calibration_audit": str(args.calibration_audit.resolve()),
        "estimator": "target mask point-cloud all-median minus fixed development bias",
        "observable_validity_gate": {
            "min_visible_pixels_64x64": args.min_visible_pixels,
            "max_jump_from_last_reliable_xyz_m": args.max_jump_m,
            "uses_teacher_state": False,
        },
        "camera": {
            "near": near,
            "far": far,
            "saved_to_camera_orientation": "camera_y=saved_y; camera_x=255-saved_x",
        },
        "num_rows": total,
        "num_enriched": total - len(errors) - len(unavailable),
        "num_observation_unavailable": len(unavailable),
        "observation_unavailable_sample": unavailable[:20],
        "num_errors": len(errors),
        "errors": errors[:100],
        "teacher_only_audit": {
            "num_rows": len(teacher_errors),
            "mean_error_cm": float(100 * np.mean(teacher_errors)),
            "median_error_cm": float(100 * np.median(teacher_errors)),
            "p90_error_cm": float(100 * np.quantile(teacher_errors, 0.9)),
            "max_error_cm": float(100 * np.max(teacher_errors)),
            "gated_num_rows": len(gated_teacher_errors),
            "gated_mean_error_cm": float(100 * np.mean(gated_teacher_errors)),
            "gated_median_error_cm": float(100 * np.median(gated_teacher_errors)),
            "gated_p90_error_cm": float(
                100 * np.quantile(gated_teacher_errors, 0.9)
            ),
            "gated_max_error_cm": float(100 * np.max(gated_teacher_errors)),
            "rejected_num_rows": len(rejected_teacher_errors),
            "rejected_mean_error_cm": float(
                100 * np.mean(rejected_teacher_errors)
            ),
            "claim_limit": (
                "teacher pose is used only for audit; stored xyz uses mask, depth, "
                "camera calibration, and a fixed development-set surface bias"
            ),
        },
        "episodes": episode_records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "enrichment-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
