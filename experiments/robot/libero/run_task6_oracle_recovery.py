#!/usr/bin/env python
"""Measure the task-6 oracle stage-switching recovery ceiling.

The oracle only reads the simulator predicate for the first subgoal. Robot
actions remain OpenVLA outputs. Once the white mug is stably on the plate, the
policy prompt switches from the full long-horizon instruction to the remaining
chocolate-pudding subgoal.
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
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = REPO_ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LIBERO_ROOT))
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", str(LIBERO_ROOT / "baseline_libero_config")
)

from run_openvla_baseline import (  # noqa: E402
    extract_privileged_object_state,
    extract_robot_state,
    jsonable,
    load_openvla,
    predict_action,
    write_json,
)

TASK_SUITE = "libero_10"
TASK_ID = 6
PUDDING_SUBGOAL = "put the chocolate pudding to the right of the plate"


class StableBooleanLatch:
    """Latch true after a predicate is true for N consecutive observations."""

    def __init__(self, confirm_steps: int):
        if confirm_steps <= 0:
            raise ValueError("confirm_steps must be positive")
        self.confirm_steps = confirm_steps
        self.run_length = 0
        self.latched = False

    def update(self, value: bool) -> bool:
        if self.latched:
            return True
        self.run_length = self.run_length + 1 if value else 0
        if self.run_length >= self.confirm_steps:
            self.latched = True
        return self.latched


class ReversibleStageLatch:
    """Enter and leave a subgoal after separately confirmed evidence."""

    def __init__(self, enter_steps: int, rollback_steps: int):
        self.enter_steps = enter_steps
        self.rollback_steps = rollback_steps
        self.enter_run = 0
        self.rollback_run = 0
        self.switched = False

    def update(self, enter: bool, rollback: bool) -> bool:
        if not self.switched:
            self.enter_run = self.enter_run + 1 if enter else 0
            if self.enter_run >= self.enter_steps:
                self.switched = True
                self.rollback_run = 0
        else:
            self.rollback_run = self.rollback_run + 1 if rollback else 0
            if self.rollback_run >= self.rollback_steps:
                self.switched = False
                self.enter_run = 0
        return self.switched


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/data2/lty/models/openvla-7b-finetuned-libero-10",
    )
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--subgoal-confirm-steps", type=int, default=5)
    parser.add_argument("--subgoal-instruction", default=PUDDING_SUBGOAL)
    parser.add_argument(
        "--oracle-stage-switch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--learned-stage-checkpoint", type=Path)
    parser.add_argument("--slot-checkpoint", type=Path)
    parser.add_argument("--grounded-role-checkpoint", type=Path)
    parser.add_argument("--grounded-temporal-checkpoint", type=Path)
    parser.add_argument("--rollback-threshold", type=float, default=0.8)
    parser.add_argument("--rollback-confirm-steps", type=int, default=5)
    parser.add_argument("--temporal-window", type=int, default=128)
    parser.add_argument(
        "--center-crop", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device-map", default="balanced_low_0")
    parser.add_argument("--gpu-memory-gib", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path)
    return parser.parse_args()


def annotate(
    image: np.ndarray,
    step: int,
    phase: str,
    predicate: bool,
    switched: bool,
    success: bool,
) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    color = (35, 150, 75) if phase == "PUDDING SUBGOAL" else (30, 100, 190)
    if success:
        color = (25, 170, 70)
    draw.rectangle((0, 0, canvas.width, 31), fill=(0, 0, 0))
    draw.text((5, 3), f"{phase}  step={step}", fill=color)
    draw.text(
        (5, 17),
        f"mug_on_plate={int(predicate)} stage_switched={int(switched)}",
        fill=(235, 235, 235),
    )
    return np.asarray(canvas)


def main() -> None:
    args = arguments()
    if args.learned_stage_checkpoint and not args.slot_checkpoint:
        raise ValueError("--slot-checkpoint is required for learned stage switching")
    if args.learned_stage_checkpoint and args.oracle_stage_switch:
        raise ValueError("use --no-oracle-stage-switch with learned switching")
    if args.grounded_role_checkpoint and args.oracle_stage_switch:
        raise ValueError("use --no-oracle-stage-switch with grounded role switching")
    if args.learned_stage_checkpoint and args.grounded_role_checkpoint:
        raise ValueError("choose one learned stage checkpoint")
    if args.grounded_temporal_checkpoint and not args.grounded_role_checkpoint:
        raise ValueError("grounded temporal mode requires grounded role checkpoint")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    initial_states = suite.get_task_init_states(TASK_ID)
    if not 0 <= args.episode_id < len(initial_states):
        raise ValueError(
            f"episode-id must be in [0, {len(initial_states) - 1}]"
        )

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    env.seed(0)
    env.reset()
    obs = env.set_init_state(initial_states[args.episode_id])

    rollout = args.rollout_dir or args.output.with_suffix("")
    rollout.mkdir(parents=True, exist_ok=True)
    steps_path = rollout / "steps.jsonl"
    metadata_path = rollout / "metadata.json"
    steps_path.write_text("", encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "demo_kind": (
            "task6_grounded_role_stage_switch"
            if args.grounded_role_checkpoint
            else (
                "task6_learned_slot_temporal_stage_switch"
                if args.learned_stage_checkpoint
                else "task6_oracle_stage_switch_ceiling"
            )
        ),
        "checkpoint": args.checkpoint,
        "task_suite": TASK_SUITE,
        "task_id": TASK_ID,
        "episode_id": args.episode_id,
        "full_instruction": task.language,
        "subgoal_instruction": args.subgoal_instruction,
        "oracle_stage_switch": args.oracle_stage_switch,
        "subgoal_confirm_steps": args.subgoal_confirm_steps,
        "stage_switch_policy_input": (
            "RGB-derived mask-supervised mug/plate role slots"
            if args.grounded_role_checkpoint
            else (
                "continuous RGB-derived slots and causal temporal hidden state"
                if args.learned_stage_checkpoint
                else "simulator On(porcelain_mug_1, plate_1) predicate only"
            )
        ),
        "oracle_action_access": False,
        "learned_stage_checkpoint": (
            str(args.learned_stage_checkpoint)
            if args.learned_stage_checkpoint else None
        ),
        "grounded_role_checkpoint": (
            str(args.grounded_role_checkpoint)
            if args.grounded_role_checkpoint else None
        ),
        "grounded_temporal_checkpoint": (
            str(args.grounded_temporal_checkpoint)
            if args.grounded_temporal_checkpoint else None
        ),
        "completed": False,
        "success": False,
    }
    write_json(metadata_path, metadata)

    model, processor, input_device = load_openvla(
        args.checkpoint, args.device_map, args.gpu_memory_gib
    )
    learned_model = slot_tokenizer = vision_backbone = vision_device = None
    learned_hidden = learned_past_slots = None
    learned_threshold = None
    grounded_transform = None
    grounded_temporal = None
    grounded_role_history = []
    grounded_probability_history = []
    if args.learned_stage_checkpoint:
        from prismatic.extern.hf.modeling_prismatic import (
            OpenVLAForActionPrediction_SlotAtt,
        )
        from scripts.train_task6_slot_temporal_state import SlotTemporalClassifier

        slot_wrapper = OpenVLAForActionPrediction_SlotAtt(
            model=model, number_of_slots=16
        )
        slot_wrapper.load_state_dict(load_file(args.slot_checkpoint), strict=False)
        vision_backbone = slot_wrapper.model.vision_backbone
        vision_device = next(vision_backbone.parameters()).device
        slot_tokenizer = slot_wrapper.object_centric_tokenizer.to(
            vision_device, dtype=torch.bfloat16
        ).eval()
        learned_model = SlotTemporalClassifier().to(vision_device).eval()
        learned_state = load_file(args.learned_stage_checkpoint)
        learned_model.load_state_dict(
            {
                key.removeprefix("model."): value
                for key, value in learned_state.items()
                if key.startswith("model.")
            }
        )
        learned_threshold = float(learned_state["threshold"][0])
        metadata["learned_threshold"] = learned_threshold
        write_json(metadata_path, metadata)
    elif args.grounded_role_checkpoint:
        from torchvision import transforms
        from scripts.train_task6_grounded_role_slots import GroundedRoleSlots

        vision_device = input_device
        learned_model = GroundedRoleSlots().to(vision_device).eval()
        learned_model.load_state_dict(load_file(args.grounded_role_checkpoint))
        learned_threshold = 0.5
        grounded_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        metadata["learned_threshold"] = learned_threshold
        if args.grounded_temporal_checkpoint:
            from scripts.train_task6_grounded_role_mamba import GroundedRoleMamba

            grounded_temporal = GroundedRoleMamba(
                backend="mamba2"
            ).to(vision_device).eval()
            grounded_temporal.load_state_dict(
                load_file(args.grounded_temporal_checkpoint)
            )
            metadata["rollback_threshold"] = args.rollback_threshold
            metadata["rollback_confirm_steps"] = args.rollback_confirm_steps
        write_json(metadata_path, metadata)
    noop = np.asarray([0, 0, 0, 0, 0, 0, -1], np.float32)
    for _ in range(args.wait_steps):
        obs, _, done, _ = env.step(noop.tolist())

    latch = StableBooleanLatch(args.subgoal_confirm_steps)
    reversible_latch = ReversibleStageLatch(
        args.subgoal_confirm_steps, args.rollback_confirm_steps
    )
    switch_step = None
    rollback_steps = []
    frames: list[np.ndarray] = []
    success = False
    try:
        for step in range(args.max_steps):
            mug_on_plate = bool(
                env.env._eval_predicate(
                    ["on", "porcelain_mug_1", "plate_1"]
                )
            )
            learned_probability = None
            learned_rollback_probability = None
            if learned_model is not None:
                raw_image = Image.fromarray(
                    np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                )
                if grounded_transform is not None:
                    pixels = grounded_transform(raw_image)[None].to(vision_device)
                    with torch.inference_mode():
                        _, relation_logits, role_slots = learned_model(pixels)
                        single_probability = torch.sigmoid(relation_logits[0])
                        if grounded_temporal is None:
                            learned_probability = float(single_probability)
                        else:
                            grounded_role_history.append(role_slots[0])
                            grounded_probability_history.append(single_probability)
                            roles = torch.stack(
                                grounded_role_history[-args.temporal_window:]
                            )[None]
                            singles = torch.stack(
                                grounded_probability_history[-args.temporal_window:]
                            )[None]
                            completion_logits, rollback_logits = grounded_temporal(
                                roles, singles
                            )
                            learned_probability = float(torch.minimum(
                                completion_logits[0, -1].sigmoid(),
                                single_probability,
                            ))
                            learned_rollback_probability = float(torch.minimum(
                                rollback_logits[0, -1].sigmoid(),
                                1 - single_probability,
                            ))
                else:
                    pixels = processor.image_processor.apply_transform(
                        raw_image
                    )[None].to(vision_device, dtype=torch.bfloat16)
                    with torch.inference_mode(), torch.autocast(
                        "cuda", dtype=torch.bfloat16
                    ):
                        patch = vision_backbone(pixels).reshape(1, -1, 2176)
                        current_slots, _, _ = slot_tokenizer(
                            patch, past_slots=learned_past_slots
                        )
                        learned_past_slots = current_slots[:, None]
                        values = learned_model.slot_norm(current_slots[:, None].float())
                        weights = learned_model.slot_score(values).squeeze(-1).softmax(-1)
                        pooled = torch.einsum("bts,btsd->btd", weights, values)
                        temporal_input = learned_model.input_projection(pooled)
                        temporal_output, learned_hidden = learned_model.temporal(
                            temporal_input, learned_hidden
                        )
                        learned_probability = float(
                            torch.sigmoid(
                                learned_model.head(temporal_output)[0, 0, 0]
                            )
                        )
                if grounded_temporal is not None:
                    was_switched = reversible_latch.switched
                    switched = reversible_latch.update(
                        learned_probability >= learned_threshold,
                        learned_rollback_probability >= args.rollback_threshold,
                    )
                    if was_switched and not switched:
                        rollback_steps.append(step)
                else:
                    switched = latch.update(
                        learned_probability >= learned_threshold
                    )
            else:
                switched = (
                    latch.update(mug_on_plate)
                    if args.oracle_stage_switch else False
                )
            if switched and switch_step is None:
                switch_step = step
            instruction = (
                args.subgoal_instruction if switched else task.language
            )
            phase = "PUDDING SUBGOAL" if switched else "FULL TASK"
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            action = predict_action(
                model,
                processor,
                image,
                instruction,
                TASK_SUITE,
                args.center_crop,
                input_device,
            )
            before = obs
            obs, reward, done, info = env.step(action.tolist())
            success = bool(env.check_success())
            record = {
                "episode_id": args.episode_id,
                "timestep": step,
                "phase": phase,
                "policy_instruction": instruction,
                "mug_on_plate_oracle": mug_on_plate,
                "oracle_switched": switched,
                "oracle_switch_event": step == switch_step,
                "stage_switched": switched,
                "stage_switch_event": step == switch_step,
                "learned_stage_probability": learned_probability,
                "learned_rollback_probability": learned_rollback_probability,
                "planned_action": action.tolist(),
                "executed_action": action.tolist(),
                "robot_state_before": extract_robot_state(before),
                "robot_state_after": extract_robot_state(obs),
                "privileged_object_state_before": extract_privileged_object_state(
                    before
                ),
                "privileged_object_state_after": extract_privileged_object_state(obs),
                "reward": float(reward),
                "done": bool(done),
                "success": success,
                "info": {
                    str(key): jsonable(value) for key, value in dict(info).items()
                },
            }
            with steps_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            frames.append(
                annotate(
                    image,
                    step,
                    phase,
                    mug_on_plate,
                    switched,
                    success,
                )
            )
            if success:
                frames.extend([frames[-1]] * 10)
                break
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps)
    metadata.update(
        {
            "completed": True,
            "success": success,
            "num_policy_steps": len(frames) - (10 if success else 0),
            "oracle_switch_step": switch_step,
            "oracle_switched": switch_step is not None,
            "stage_switch_step": switch_step,
            "stage_switched": switch_step is not None,
            "rollback_steps": rollback_steps,
            "video": str(args.output.resolve()),
            "steps_jsonl": str(steps_path.resolve()),
        }
    )
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
