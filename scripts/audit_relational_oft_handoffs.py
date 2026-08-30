#!/usr/bin/env python3
"""Audit one teacher phase boundary per task with one shared frozen OFT load."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_object")
    parser.add_argument(
        "--normalization-key",
        help="Defaults to task-suite (or task-suite_no_noops when that is the available checkpoint key).",
    )
    parser.add_argument(
        "--candidate-phase", default="approach_target",
        help=("Audit the state immediately after the final contiguous step of this phase. "
              "Canonical approach_target/lift_target names also match the profiled/held "
              "teacher variants."),
    )
    parser.add_argument("--max-env-steps", type=int, default=600)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--source-shard-index", type=int, default=0)
    parser.add_argument("--num-source-shards", type=int, default=1)
    parser.add_argument("--task-ids", help="Optional comma-separated task ids to audit.")
    parser.add_argument("--oft-root", type=Path, default=Path("/data/lty/code/openvla-oft-official"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata"))
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve_normalization_key(stats: dict, task_suite: str, requested: str | None) -> str:
    candidates = (requested,) if requested else (task_suite, f"{task_suite}_no_noops")
    for candidate in candidates:
        if candidate in stats:
            return candidate
    raise KeyError(f"checkpoint has no normalization key among {candidates}")


def phase_aliases(selected: str) -> tuple[str, ...]:
    return {
        "approach_target": ("approach_target", "approach_profiled_target"),
        "lift_target": ("lift_target", "lift_held_target"),
    }.get(selected, (selected,))


def boundary(phases: list[str], selected: str) -> tuple[int, str]:
    if selected == "fork":
        return 0, "fork"
    matches = phase_aliases(selected)
    present = [name for name in matches if name in phases]
    if len(present) != 1:
        raise ValueError(f"teacher has no unambiguous phase {selected}: found {present}")
    teacher_phase = present[0]
    indices = [index for index, phase in enumerate(phases) if phase == teacher_phase]
    if not indices:
        raise ValueError(f"teacher has no phase {selected}")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"teacher phase is not contiguous: {teacher_phase}")
    return indices[-1] + 1, teacher_phase


def main() -> None:
    args = parse_args()
    if min(args.max_env_steps, args.fps) < 1:
        raise ValueError("max-env-steps and fps must be positive")
    if args.num_source_shards < 1 or not 0 <= args.source_shard_index < args.num_source_shards:
        raise ValueError("source shard index must be in [0, num-source-shards)")
    all_metadata = sorted(args.teacher_root.glob("task*-episode*/metadata.json"))
    # A collection root contains both successful and failed attempts.  Only a
    # successful teacher trajectory can define a candidate OFT handoff.
    teacher_metadata = []
    for path in all_metadata:
        payload = json.loads(path.read_text())
        if payload.get("success") is not True or payload.get("task_suite") != args.task_suite:
            continue
        try:
            boundary(payload["phase_by_step"], args.candidate_phase)
        except ValueError:
            continue
        teacher_metadata.append(path)
    if args.task_ids:
        selected_ids = {int(value) for value in args.task_ids.split(",") if value.strip()}
        teacher_metadata = [
            path for path in teacher_metadata
            if int(json.loads(path.read_text())["task_id"]) in selected_ids
        ]
    if args.max_sources is not None:
        teacher_metadata = teacher_metadata[:args.max_sources]
    teacher_metadata = teacher_metadata[args.source_shard_index::args.num_source_shards]
    if not teacher_metadata:
        raise ValueError(f"no teacher sources below {args.teacher_root}")
    for path in (args.oft_root, args.checkpoint, args.libero_root):
        if not path.exists():
            raise FileNotFoundError(path)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_root / "baseline_libero_config")
    sys.path[:0] = [str(args.libero_root), str(args.oft_root), str(ROOT)]

    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "This OFT/LIBERO rollout must use .conda/envs/openvla-oft/bin/python; "
            f"current interpreter is {sys.executable}. Prefer "
            "scripts/run_relational_oft_handoff_audit.sh instead of calling Python directly."
        ) from error
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from experiments.robot.libero.libero_utils import get_libero_image
    from experiments.robot.libero.run_libero_eval import process_action
    import prismatic
    import prismatic.training
    import prismatic.vla
    from recovery.models.oft_l1_action_head import OFTL1RegressionActionHead
    from scripts.precompute_v4_oft_relation_features import (
        dynamic_module, exact_oft_policy_image, load_torch_state, normalized_proprio,
    )

    prismatic.__path__.append(str(args.oft_root / "prismatic"))
    prismatic.training.__path__.append(str(args.oft_root / "prismatic" / "training"))
    prismatic.vla.__path__.append(str(args.oft_root / "prismatic" / "vla"))
    projectors = dynamic_module("handoff_audit_projectors", args.oft_root / "prismatic/models/projectors.py")
    device = torch.device("cuda")
    action_head_path = args.checkpoint / "action_head--150000_checkpoint.pt"
    projector_path = args.checkpoint / "proprio_projector--150000_checkpoint.pt"
    stats = json.loads((args.checkpoint / "dataset_statistics.json").read_text())
    normalization_key = resolve_normalization_key(stats, args.task_suite, args.normalization_key)
    action_head_hash, projector_hash = sha256(action_head_path), sha256(projector_path)
    config_path = args.checkpoint / "config.json"
    model_index_path = args.checkpoint / "model.safetensors.index.json"
    checkpoint_contract = {
        "path": str(args.checkpoint.resolve()),
        "config_sha256": sha256(config_path) if config_path.is_file() else None,
        "model_index_sha256": sha256(model_index_path) if model_index_path.is_file() else None,
        "action_head_sha256": action_head_hash,
        "proprio_projector_sha256": projector_hash,
    }
    oft = processor = action_head = proprio_projector = None

    def ensure_policy_loaded():
        nonlocal oft, processor, action_head, proprio_projector
        if oft is None:
            oft = AutoModelForVision2Seq.from_pretrained(
                str(args.checkpoint), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(device).eval()
            oft.vision_backbone.set_num_images_in_input(2)
            oft.norm_stats = stats
            processor = AutoProcessor.from_pretrained(str(args.checkpoint), trust_remote_code=True)
            action_head = OFTL1RegressionActionHead(4096, 4096, 7)
            action_head.load_state_dict(load_torch_state(action_head_path, torch))
            action_head = action_head.to(device=device, dtype=torch.bfloat16).eval()
            proprio_projector = projectors.ProprioProjector(4096, 8)
            proprio_projector.load_state_dict(load_torch_state(projector_path, torch))
            proprio_projector = proprio_projector.to(device=device, dtype=torch.bfloat16).eval()
        return oft, processor, action_head, proprio_projector

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for metadata_path in teacher_metadata:
        teacher = json.loads(metadata_path.read_text())
        if teacher.get("success") is not True or teacher.get("task_suite") != args.task_suite:
            raise ValueError(f"invalid teacher metadata: {metadata_path}")
        task_id, episode_id = int(teacher["task_id"]), int(teacher["episode_id"])
        teacher_step, teacher_phase = boundary(teacher["phase_by_step"], args.candidate_phase)
        state_path = metadata_path.parent / "states" / f"step-{teacher_step:04d}.npy"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        output = args.output_root / f"task{task_id}-episode{episode_id}"
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "report.json"
        run_contract = {
            "protocol": "fresh_replan_oft_handoff_v2",
            "task_suite": args.task_suite,
            "task_id": task_id,
            "episode_id": episode_id,
            "candidate_phase": args.candidate_phase,
            "teacher_phase": teacher_phase,
            "teacher_relative_timestep": teacher_step,
            "teacher_metadata_sha256": sha256(metadata_path),
            "state_sha256": sha256(state_path),
            "normalization_key": normalization_key,
            "max_env_steps": args.max_env_steps,
            "checkpoint": checkpoint_contract,
        }
        run_fingerprint = fingerprint(run_contract)
        if result_path.is_file():
            prior = json.loads(result_path.read_text())
            if prior.get("complete") is True and prior.get("run_fingerprint") == run_fingerprint:
                records.append(prior)
                print(json.dumps({"task_id": task_id, "status": "skipped_complete", "success": prior["success"]}), flush=True)
                continue

        task = suite.get_task(task_id)
        if task.language != teacher["task_language"]:
            raise ValueError(f"task language mismatch for task {task_id}")
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
        )
        try:
            env.seed(0)
            env.reset()
            observation = env.regenerate_obs_from_state(np.load(state_path))
            controller = env.env.robots[0].controller
            controller.update(force=True)
            controller.reset_goal()
            env.env.timestep = 0
            env.env.done = False
            prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
            # Preserve a visual certificate even when the restored boundary
            # already satisfies the task and OFT correctly executes zero steps.
            frames, events = [get_libero_image(observation)], []
            queue: deque = deque()
            env_steps = 0
            # Placement/release boundaries may already satisfy the task.  Do
            # not let one unnecessary OFT step destroy a valid handoff state.
            while env_steps < args.max_env_steps and not env.check_success():
                if not queue:
                    oft, processor, action_head, proprio_projector = ensure_policy_loaded()
                    main_image = get_libero_image(observation)
                    wrist_image = observation["robot0_eye_in_hand_image"][::-1, ::-1]
                    inputs = processor(prompt, exact_oft_policy_image(main_image, Image, tf)).to(
                        device, dtype=torch.bfloat16,
                    )
                    wrist = processor(prompt, exact_oft_policy_image(wrist_image, Image, tf)).to(
                        device, dtype=torch.bfloat16,
                    )
                    inputs["pixel_values"] = torch.cat((inputs["pixel_values"], wrist["pixel_values"]), dim=1)
                    proprio = normalized_proprio({
                        "robot0_eef_pos": observation["robot0_eef_pos"],
                        "robot0_eef_quat": observation["robot0_eef_quat"],
                        "robot0_gripper_qpos": observation["robot0_gripper_qpos"],
                    }, oft.norm_stats[normalization_key]["proprio"], np)
                    with torch.inference_mode():
                        _, hidden = oft.predict_action(
                            **inputs, unnorm_key=normalization_key, do_sample=False,
                            proprio=proprio, proprio_projector=proprio_projector,
                            action_head=action_head, use_film=False,
                        )
                        normalized = action_head.predict_action(hidden.to(torch.bfloat16)).float().cpu().numpy()
                    raw = oft._unnormalize_actions(normalized, normalization_key)[0]
                    queue.extend(raw)
                    events.append({
                        "decision": len(events), "env_step": env_steps,
                        "first_planned_action": np.asarray(raw[0], np.float32).tolist(),
                    })
                frames.append(get_libero_image(observation))
                action = process_action(np.asarray(queue.popleft(), np.float32), "openvla")
                observation, _, done, _ = env.step(action.tolist())
                env_steps += 1
                if done or env.check_success():
                    break
            video = output / "rollout.mp4"
            imageio.mimsave(video, frames, fps=args.fps)
            result = {
                "kind": "relational_oft_fresh_replan_handoff_audit_v1",
                "run_fingerprint": run_fingerprint,
                "run_contract": run_contract,
                "complete": True,
                "success": bool(env.check_success()),
                "task_suite": args.task_suite, "task_id": task_id, "episode_id": episode_id,
                "candidate_phase": args.candidate_phase,
                "teacher_phase": teacher_phase,
                "teacher_relative_timestep": teacher_step,
                "source_fork_timestep": int(teacher["fork_timestep"]),
                "env_steps": env_steps, "policy_decisions": len(events),
                "normalization_key": normalization_key,
                "teacher": str(metadata_path.parent.resolve()),
                "state": str(state_path.resolve()), "video": str(video.resolve()),
                "events": events,
            }
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            records.append(result)
            print(json.dumps({
                "task_id": task_id, "status": "audited", "success": result["success"],
                "teacher_relative_timestep": teacher_step, "env_steps": env_steps,
                "completed": len(records), "total": len(teacher_metadata),
            }), flush=True)
        finally:
            env.close()

    aggregate = {
        "kind": "relational_oft_handoff_audit_report_v1", "complete": True,
        "candidate_phase": args.candidate_phase,
        "task_suite": args.task_suite, "normalization_key": normalization_key,
        "source_shard_index": args.source_shard_index,
        "num_source_shards": args.num_source_shards,
        "teacher_root": str(args.teacher_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_contract": checkpoint_contract,
        "action_head_sha256": action_head_hash,
        "proprio_projector_sha256": projector_hash,
        "num_sources": len(records),
        "successes": sum(bool(record["success"]) for record in records),
        "records": records,
    }
    (args.output_root / "report.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps({key: aggregate[key] for key in ("num_sources", "successes", "candidate_phase")}, indent=2))


if __name__ == "__main__":
    main()
