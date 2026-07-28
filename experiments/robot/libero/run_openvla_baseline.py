"""Minimal headless LIBERO baseline without object slots or SlotSSM.

Start with ``--policy dummy`` to validate MuJoCo/EGL, then use
``--policy openvla`` with an official LIBERO-finetuned OpenVLA checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault(
    "LIBERO_CONFIG_PATH",
    str(REPO_ROOT / "experiments" / "robot" / "libero" / "baseline_libero_config"),
)


SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("dummy", "openvla"), default="dummy")
    parser.add_argument("--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--task-suite", choices=tuple(SUITE_MAX_STEPS), default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0, help="0 uses the suite default")
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--device-map",
        default="balanced_low_0",
        help="Accelerate device map; balanced_low_0 keeps extra room on GPU 0",
    )
    parser.add_argument(
        "--gpu-memory-gib",
        type=int,
        default=20,
        help="Maximum model memory per visible GPU; leaves room for rendering and activations",
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "baseline-a.mp4")
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=None,
        help="Directory for metadata.json and crash-safe steps.jsonl; defaults beside --output.",
    )
    return parser.parse_args()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def extract_robot_state(obs: dict) -> dict[str, object]:
    """Keep compact numeric proprioception while excluding image-like observations."""
    state = {}
    for key in sorted(obs):
        if not key.startswith("robot0_") or any(
            token in key.lower() for token in ("image", "depth", "segmentation", "rgb")
        ):
            continue
        try:
            value = np.asarray(obs[key])
        except Exception:
            continue
        if value.dtype.kind not in "biuf" or value.size > 128:
            continue
        state[key] = value.reshape(-1).astype(np.float32).tolist()
    return state


def extract_privileged_object_state(obs: dict) -> dict[str, object]:
    """Simulator-only object poses for labels/evaluation, never policy input."""
    state = {}
    for key in sorted(obs):
        if key.startswith("robot0_") or "_to_robot" in key:
            continue
        if not (key.endswith("_pos") or key.endswith("_quat")):
            continue
        value = np.asarray(obs[key])
        if value.dtype.kind in "biuf" and value.size <= 8:
            state[key] = value.reshape(-1).astype(np.float32).tolist()
    return state


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_openvla(checkpoint: str, device_map: str, gpu_memory_gib: int):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("OpenVLA inference requires a visible CUDA GPU")
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    max_memory = {
        index: f"{gpu_memory_gib}GiB" for index in range(torch.cuda.device_count())
    }
    max_memory["cpu"] = "64GiB"
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,
        max_memory=max_memory,
    )
    model.eval()
    input_device = next(model.parameters()).device
    print(f"CUDA devices={torch.cuda.device_count()} input_device={input_device}")
    if hasattr(model, "hf_device_map"):
        print(f"device_map={model.hf_device_map}")
    return model, processor, input_device


def center_crop(image: np.ndarray) -> np.ndarray:
    from PIL import Image

    height, width = image.shape[:2]
    scale = np.sqrt(0.9)
    crop_h, crop_w = int(height * scale), int(width * scale)
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    cropped = image[top : top + crop_h, left : left + crop_w]
    return np.asarray(Image.fromarray(cropped).resize((width, height), Image.Resampling.LANCZOS))


def predict_action(
    model, processor, image: np.ndarray, instruction: str, suite: str, crop: bool, input_device
) -> np.ndarray:
    import torch
    from PIL import Image

    if crop:
        image = center_crop(image)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(image).convert("RGB")).to(
        input_device, dtype=torch.bfloat16
    )
    with torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key=suite, do_sample=False)
    action = np.asarray(action, dtype=np.float32)
    action[-1] = 2.0 * action[-1] - 1.0
    action[-1] = np.sign(action[-1])
    action[-1] *= -1.0
    return action


def main() -> None:
    args = parse_args()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path

    suites = benchmark.get_benchmark_dict()
    suite = suites[args.task_suite]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError(f"task-id must be in [0, {suite.n_tasks - 1}]")
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.episode_id < len(initial_states):
        raise ValueError(f"episode-id must be in [0, {len(initial_states) - 1}]")

    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256
    )
    env.seed(0)
    env.reset()
    obs = env.set_init_state(initial_states[args.episode_id])

    rollout_dir = args.rollout_dir or args.output.with_suffix("")
    rollout_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = rollout_dir / "metadata.json"
    steps_path = rollout_dir / "steps.jsonl"
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": args.policy,
        "checkpoint": args.checkpoint if args.policy == "openvla" else None,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "episode_id": args.episode_id,
        "task_language": task.language,
        "wait_steps": args.wait_steps,
        "max_steps": args.max_steps or SUITE_MAX_STEPS[args.task_suite],
        "center_crop": args.center_crop,
        "observation_keys": sorted(obs.keys()),
        "action_semantics": {
            "planned_action": "policy output before environment/safety modification",
            "executed_action": "action actually passed to env.step",
        },
        "completed": False,
        "success": False,
    }
    write_json(metadata_path, metadata)
    steps_path.write_text("", encoding="utf-8")

    model = processor = input_device = None
    if args.policy == "openvla":
        model, processor, input_device = load_openvla(
            args.checkpoint, args.device_map, args.gpu_memory_gib
        )

    frames: list[np.ndarray] = []
    no_op = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    for _ in range(args.wait_steps):
        obs, _, done, _ = env.step(no_op)
        if done:
            break

    max_steps = args.max_steps or SUITE_MAX_STEPS[args.task_suite]
    success = False
    try:
        for step in range(max_steps):
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            frames.append(image)
            planned_action = no_op.copy()
            if args.policy == "openvla":
                planned_action = predict_action(
                    model,
                    processor,
                    image,
                    task.language,
                    args.task_suite,
                    args.center_crop,
                    input_device,
                )
            # This baseline has no safety controller yet, so planned and
            # executed actions are identical.  Keep both fields for the
            # Failure-Aware data contract.
            executed_action = np.asarray(planned_action, dtype=np.float32).copy()
            robot_state_before = extract_robot_state(obs)
            next_obs, reward, done, info = env.step(executed_action.tolist())
            success = bool(done)
            record = {
                "episode_id": args.episode_id,
                "timestep": step,
                "phase": "policy",
                "planned_action": planned_action.astype(np.float32).tolist(),
                "executed_action": executed_action.tolist(),
                "robot_state_before": robot_state_before,
                "robot_state_after": extract_robot_state(next_obs),
                "privileged_object_state_before": extract_privileged_object_state(obs),
                "privileged_object_state_after": extract_privileged_object_state(next_obs),
                "reward": float(reward),
                "done": bool(done),
                "success": success,
                "info": {str(key): jsonable(value) for key, value in dict(info).items()},
            }
            with steps_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            obs = next_obs
            if success:
                break
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=20)
    metadata.update(
        {
            "completed": True,
            "success": bool(success),
            "num_policy_steps": len(frames),
            "video": str(args.output.resolve()),
            "steps_jsonl": str(steps_path.resolve()),
        }
    )
    write_json(metadata_path, metadata)
    print(f"policy={args.policy} task={task.language!r} steps={len(frames)} success={bool(success)}")
    print(f"video={args.output.resolve()}")
    print(f"rollout={rollout_dir.resolve()}")


if __name__ == "__main__":
    main()
