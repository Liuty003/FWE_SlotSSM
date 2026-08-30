#!/usr/bin/env python3
"""Run the learned OFT+SlotSSM gate and recovery head in LIBERO online.

The controller has no simulator-state failure input.  At every observation it
computes visual slots, updates causal SlotSSM history, obtains OFT action-token
features, and uses the learned three-state gate for hard routing.
"""
from __future__ import annotations

import argparse, copy, json, os, sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOCAL_LIBERO_ROOT = ROOT / "experiments" / "robot" / "libero"
sys.path.insert(0, str(LOCAL_LIBERO_ROOT))
from scripts.smoke_unified_oft_slotssm_bridge import DEFAULT_OFT_ROOT, DEFAULT_STAGE1, DEFAULT_STAGE2, center_crop_resize, dynamic_module, load_torch_state, normalized_proprio  # noqa: E402
from recovery.models.openvla_oft_slotssm_adapter import OpenVLAOFTSlotSSMAdapter, RecoveryMode  # noqa: E402
from recovery.models.neural_recovery_residual import NeuralRecoveryResidual  # noqa: E402
# Import the existing SlotSSM-side implementation before the OFT source tree
# is prepended to sys.path later in ``main``.
from prismatic.extern.hf.modeling_prismatic import SlotAttention  # noqa: E402
from scripts.train_slot_stage2_ssm import Stage2Dynamics  # noqa: E402


def quaternion_rotation_error(target_wxyz: np.ndarray, current_wxyz: np.ndarray) -> np.ndarray:
    """Return the shortest world-frame rotation vector from current to target.

    LIBERO exposes EEF quaternions in w,x,y,z order.  This small dependency-free
    conversion lets the placement macro regulate orientation from proprio,
    rather than replaying a fixed number of wrist actions from one trajectory.
    """
    target = target_wxyz / np.linalg.norm(target_wxyz)
    current = current_wxyz / np.linalg.norm(current_wxyz)
    # q_error = q_target * conjugate(q_current), all in w,x,y,z convention.
    tw, tx, ty, tz = target; cw, cx, cy, cz = current
    w = tw * cw + tx * cx + ty * cy + tz * cz
    x = -tw * cx + tx * cw - ty * cz + tz * cy
    y = -tw * cy + tx * cz + ty * cw - tz * cx
    z = -tw * cz - tx * cy + ty * cx + tz * cw
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    vector = np.asarray([x, y, z], dtype=np.float32)
    magnitude = float(np.linalg.norm(vector))
    if magnitude < 1e-7:
        return np.zeros(3, dtype=np.float32)
    return vector * (2.0 * np.arctan2(magnitude, np.clip(w, -1.0, 1.0)) / magnitude)


def args_() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-suite", default="libero_10"); p.add_argument("--task-id", type=int, default=6); p.add_argument("--episode-id", type=int, default=7)
    p.add_argument("--gpu", type=int, default=1); p.add_argument("--normal-open-loop", type=int, default=8); p.add_argument("--history", type=int, default=32)
    p.add_argument(
        "--official-normal-actions", action="store_true",
        help="Use the official OFT+ LIBERO preprocessing/action path whenever recovery is inactive.",
    )
    p.add_argument("--recovery-confirm-steps", type=int, default=2, help="Consecutive confident RECOVERY predictions required to interrupt NORMAL.")
    p.add_argument("--max-learned-recoveries", type=int, default=1,
                   help="Safety cap on learned-gate interventions per episode; oracle interventions are unaffected.")
    p.add_argument("--complete-confirm-steps", type=int, default=2, help="Consecutive confident RECOVERY_COMPLETE predictions required to leave recovery.")
    p.add_argument("--min-gate-confidence", type=float, default=0.0, help="Ignore non-NORMAL gate predictions below this softmax confidence.")
    p.add_argument(
        "--recovery-probability-threshold", type=float,
        help="Binary-gate calibration: enter RECOVERY when P(RECOVERY) reaches this value, instead of argmax at 0.5.",
    )
    p.add_argument("--max-policy-steps", type=int, default=None, help="Optional short online smoke horizon after stabilization.")
    p.add_argument("--oft-root", type=Path, default=DEFAULT_OFT_ROOT); p.add_argument("--checkpoint", type=Path, default=Path("/data2/lty/models/openvla-7b-oft-finetuned-libero-plus-mixdata"))
    p.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    p.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1); p.add_argument("--stage2-checkpoint", type=Path, default=DEFAULT_STAGE2)
    p.add_argument("--adapter-checkpoint", type=Path, default=Path("/data2/lty/naive-e-slotssm-runs/oft-slotssm-unified-smoke-v2/adapter_unified.safetensors"))
    p.add_argument("--gate-checkpoint", type=Path,
                   help="Optional gate-only checkpoint loaded after the unified adapter checkpoint.")
    p.add_argument(
        "--gate-num-classes", type=int, choices=(2, 3),
        help="Gate output classes. Defaults to 2 when --gate-checkpoint is supplied, else 3 for legacy adapters.",
    )
    p.add_argument(
        "--learned-recovery-max-steps", type=int, default=70,
        help="Shared finite recovery horizon for a two-class gate, before handing control back to OFT+.",
    )
    p.add_argument(
        "--neural-hard-phase-routing", action="store_true",
        help=("During pure neural recovery, hard-route through the argmax of the "
              "adapter's learned phase head instead of softly averaging experts."),
    )
    p.add_argument(
        "--deterministic-sdp", action="store_true",
        help=("Use PyTorch's math-only scaled-dot-product attention kernels for "
              "reproducible closed-loop diagnostics; launch with "
              "CUBLAS_WORKSPACE_CONFIG=:4096:8 as well."),
    )
    p.add_argument("--output-dir", type=Path, default=Path("/data2/lty/naive-e-slotssm-runs/oft-slotssm-unified-online-v1"))
    p.add_argument("--force-recovery-at-step", type=int, default=None, help="Oracle-only policy step at which to route through the learned recovery head.")
    p.add_argument("--force-recovery-steps", type=int, default=0, help="Oracle-only duration; zero disables forced recovery.")
    p.add_argument(
        "--recovery-open-prefix-steps", type=int, default=0,
        help="Diagnostic phase-prior ablation: force OFT gripper-open during the first N recovery actions.",
    )
    p.add_argument(
        "--recovery-phase-schedule", choices=("none", "teacher_duration"), default="none",
        help="Oracle-only diagnostic: provide a fixed local-regrasp phase indexed by recovery duration.",
    )
    p.add_argument(
        "--recovery-phase-durations", default="6,10,12,12",
        help="BACKOFF,ALIGN,DESCEND,CLOSE action counts for the fixed recovery schedule.",
    )
    p.add_argument(
        "--recovery-backoff-template", action="store_true",
        help="Oracle-only diagnostic: execute the validated six-step vertical-open safety backoff primitive.",
    )
    p.add_argument(
        "--recovery-geometry-controller", action="store_true",
        help=("Oracle-only geometry diagnostic: use the predicted target-minus-EEF vector "
              "with the fixed local regrasp phase program instead of the learned action residual."),
    )
    p.add_argument(
        "--recovery-target-anchor",
        help=("Optional task-specific learned target world coordinate x,y,z in metres. "
              "Used only by --recovery-geometry-controller; online relative geometry is anchor minus EEF proprio."),
    )
    p.add_argument("--patch-geometry-checkpoint", type=Path,
                   help="Optional patch-query visual geometry checkpoint. It replaces the fixed book target anchor, not the placement anchor.")
    p.add_argument(
        "--candidate-verifier-checkpoint", type=Path,
        help=(
            "Optional matched-FSM macro verifier.  At a learned recovery trigger it "
            "selects one discrete XY approach offset from five candidates using only "
            "OFT hidden state, SlotSSM state, proprio and previous action."
        ),
    )
    p.add_argument(
        "--neural-recovery-mode", action="store_true",
        help=("Pure-neural recovery deployment: use the candidate verifier only as a learned five-way mode "
              "selector, then route every recovery action through the mode-conditioned action head. "
              "No geometry controller, anchor, or FSM is used."),
    )
    p.add_argument(
        "--disable-recovery-phase-conditioning", action="store_true",
        help=("Pure-neural ablation: bypass offline-phase embeddings and phase experts; "
              "the recovery action head uses only visual/history/proprio features and the learned mode."),
    )
    p.add_argument(
        "--neural-geometry-conditioning", action="store_true",
        help=("Pure-neural recovery: condition the learned action head on its own SlotSSM visual "
              "target-relative-position prediction; never uses a simulator pose or target anchor."),
    )
    p.add_argument("--neural-recovery-residual-checkpoint", type=Path,
                   help="Optional bounded neural residual for learned on-policy recovery correction states.")
    p.add_argument("--neural-recovery-residual-threshold", type=float, default=0.5,
                   help="Apply the learned residual when its learned neural gate reaches this probability.")
    p.add_argument(
        "--recovery-geometry-xy-offset", default="0,0",
        help="Deployable local-program approach offset x,y in metres, applied with the geometry controller.",
    )
    p.add_argument(
        "--recovery-anchor-fsm", action="store_true",
        help=("Use online EEF-proprio feedback to hold alignment before descent from a target anchor or patch-query visual geometry. "
              "It never reads simulator object pose or contact."),
    )
    p.add_argument(
        "--recovery-placement-anchor",
        help=("Optional task-specific learned EEF placement coordinate x,y,z. Together with "
              "--recovery-anchor-fsm, complete the local repair by transporting the regrasped object "
              "to this anchor, descending, opening, and retreating using only EEF proprio feedback."),
    )
    p.add_argument(
        "--recovery-placement-quat",
        help=("Optional learned placement EEF quaternion w,x,y,z. With a placement anchor, regulate the wrist "
              "orientation from EEF proprio during transport and descent."),
    )
    p.add_argument("--save-forced-states-dir", type=Path, help="Optional teacher-only snapshots from learned forced-recovery states for on-policy correction.")
    p.add_argument("--save-forced-state-every", type=int, default=8)
    p.add_argument(
        "--save-learned-recovery-states-dir",
        type=Path,
        help=(
            "Optional teacher-only snapshot of each state where the learned gate "
            "enters recovery.  It is for DAgger-style recovery-label collection "
            "only and is never read by the online controller."
        ),
    )
    p.add_argument("--save-online-images", action="store_true",
                   help="Save dual-camera PNGs and policy-visible proprio/action records for recovery-data collection.")
    p.add_argument(
        "--save-neural-recovery-states-dir", type=Path,
        help=("Teacher-only DAgger snapshots every --save-neural-recovery-state-every steps while the "
              "pure neural recovery policy is active. Never read by deployment."),
    )
    p.add_argument("--save-neural-recovery-state-every", type=int, default=8)
    return p.parse_args()


def main() -> None:
    a = args_()
    for path in (a.oft_root, a.libero_root, a.checkpoint, a.stage1_checkpoint, a.stage2_checkpoint, a.adapter_checkpoint):
        if not path.exists(): raise FileNotFoundError(path)
    if (a.force_recovery_at_step is None) != (a.force_recovery_steps == 0):
        raise ValueError("oracle recovery requires both --force-recovery-at-step and positive --force-recovery-steps")
    if a.recovery_open_prefix_steps < 0:
        raise ValueError("recovery_open_prefix_steps must be non-negative")
    if a.max_learned_recoveries < 0:
        raise ValueError("max_learned_recoveries must be non-negative")
    if a.learned_recovery_max_steps < 1:
        raise ValueError("learned-recovery-max-steps must be positive")
    if a.save_neural_recovery_states_dir is not None and (
        not a.neural_recovery_mode or a.save_neural_recovery_state_every < 1
    ):
        raise ValueError("neural recovery snapshots require --neural-recovery-mode and positive snapshot interval")
    if a.recovery_probability_threshold is not None and not 0.0 <= a.recovery_probability_threshold <= 1.0:
        raise ValueError("recovery-probability-threshold must be in [0, 1]")
    if a.recovery_phase_schedule != "none" and a.force_recovery_at_step is None:
        raise ValueError("a recovery phase schedule is oracle-only and requires forced recovery")
    if a.recovery_backoff_template and a.force_recovery_at_step is None:
        raise ValueError("the backoff template is oracle-only and requires forced recovery")
    if a.recovery_geometry_controller and not a.recovery_anchor_fsm and (
        a.force_recovery_at_step is None or a.recovery_phase_schedule != "teacher_duration"
    ):
        raise ValueError("fixed-duration geometry control requires an oracle trigger and teacher_duration phase schedule")
    if a.recovery_target_anchor is not None and not a.recovery_geometry_controller:
        raise ValueError("a target anchor is only valid with the geometry controller")
    if a.recovery_anchor_fsm and (not a.recovery_geometry_controller or (a.recovery_target_anchor is None and a.patch_geometry_checkpoint is None)):
        raise ValueError("--recovery-anchor-fsm requires geometry controller plus a target anchor or patch geometry checkpoint")
    if a.recovery_placement_anchor is not None and not a.recovery_anchor_fsm:
        raise ValueError("--recovery-placement-anchor requires --recovery-anchor-fsm")
    if a.recovery_placement_quat is not None and a.recovery_placement_anchor is None:
        raise ValueError("--recovery-placement-quat requires --recovery-placement-anchor")
    try:
        phase_durations = tuple(int(value) for value in a.recovery_phase_durations.split(","))
    except ValueError as error:
        raise ValueError("--recovery-phase-durations must be four positive integers") from error
    if len(phase_durations) != 4 or any(value < 1 for value in phase_durations):
        raise ValueError("--recovery-phase-durations must be four positive integers")
    backoff_end, align_end, descend_end, close_end = np.cumsum(phase_durations).tolist()
    try:
        geometry_xy_offset = np.asarray([float(value) for value in a.recovery_geometry_xy_offset.split(",")], dtype=np.float32)
    except ValueError as error:
        raise ValueError("--recovery-geometry-xy-offset must be x,y") from error
    if geometry_xy_offset.shape != (2,):
        raise ValueError("--recovery-geometry-xy-offset must be x,y")
    target_anchor = None
    if a.recovery_target_anchor is not None:
        try:
            target_anchor = np.asarray([float(value) for value in a.recovery_target_anchor.split(",")], dtype=np.float32)
        except ValueError as error:
            raise ValueError("--recovery-target-anchor must be x,y,z") from error
        if target_anchor.shape != (3,):
            raise ValueError("--recovery-target-anchor must be x,y,z")
    placement_anchor = None
    if a.recovery_placement_anchor is not None:
        try:
            placement_anchor = np.asarray([float(value) for value in a.recovery_placement_anchor.split(",")], dtype=np.float32)
        except ValueError as error:
            raise ValueError("--recovery-placement-anchor must be x,y,z") from error
        if placement_anchor.shape != (3,):
            raise ValueError("--recovery-placement-anchor must be x,y,z")
    placement_quat = None
    if a.recovery_placement_quat is not None:
        try:
            placement_quat = np.asarray([float(value) for value in a.recovery_placement_quat.split(",")], dtype=np.float32)
        except ValueError as error:
            raise ValueError("--recovery-placement-quat must be w,x,y,z") from error
        if placement_quat.shape != (4,) or float(np.linalg.norm(placement_quat)) < 1e-6:
            raise ValueError("--recovery-placement-quat must be a non-zero w,x,y,z quaternion")
        placement_quat /= np.linalg.norm(placement_quat)
    if a.save_forced_states_dir is not None and (a.force_recovery_at_step is None or a.save_forced_state_every < 1):
        raise ValueError("forced-state snapshots require oracle recovery and a positive interval")
    if a.gate_checkpoint is not None and not a.gate_checkpoint.exists():
        raise FileNotFoundError(a.gate_checkpoint)
    if a.patch_geometry_checkpoint is not None and not a.patch_geometry_checkpoint.exists():
        raise FileNotFoundError(a.patch_geometry_checkpoint)
    if a.candidate_verifier_checkpoint is not None and not a.candidate_verifier_checkpoint.exists():
        raise FileNotFoundError(a.candidate_verifier_checkpoint)
    if (a.neural_recovery_residual_checkpoint is not None and
            not a.neural_recovery_residual_checkpoint.exists()):
        raise FileNotFoundError(a.neural_recovery_residual_checkpoint)
    if not 0.0 <= a.neural_recovery_residual_threshold <= 1.0:
        raise ValueError("--neural-recovery-residual-threshold must be in [0, 1]")
    if a.candidate_verifier_checkpoint is not None and not (
        (a.recovery_geometry_controller and a.recovery_anchor_fsm) or a.neural_recovery_mode
    ):
        raise ValueError("candidate verifier requires either closed-loop geometry recovery or --neural-recovery-mode")
    if a.neural_recovery_mode:
        if a.candidate_verifier_checkpoint is None:
            raise ValueError("--neural-recovery-mode requires --candidate-verifier-checkpoint")
        if a.recovery_geometry_controller or a.recovery_anchor_fsm or a.recovery_target_anchor is not None:
            raise ValueError("--neural-recovery-mode cannot use geometry-controller, anchor, or FSM options")
    elif a.disable_recovery_phase_conditioning:
        raise ValueError("--disable-recovery-phase-conditioning requires --neural-recovery-mode")
    if a.neural_geometry_conditioning and not a.neural_recovery_mode:
        raise ValueError("--neural-geometry-conditioning requires --neural-recovery-mode")
    if (a.neural_recovery_residual_checkpoint is not None and
            not a.neural_recovery_mode):
        raise ValueError("--neural-recovery-residual-checkpoint requires --neural-recovery-mode")
    if a.neural_hard_phase_routing and not a.neural_recovery_mode:
        raise ValueError("--neural-hard-phase-routing requires --neural-recovery-mode")
    if a.neural_hard_phase_routing and a.disable_recovery_phase_conditioning:
        raise ValueError("--neural-hard-phase-routing cannot combine with disabled phase conditioning")
    # SlotSSM imports torch above this function.  A launcher-specified device
    # must therefore win; changing CUDA_VISIBLE_DEVICES after those imports is
    # too late and can silently place the 7B model on GPU 0.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu)); os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    libero_root = a.libero_root; os.environ["LIBERO_CONFIG_PATH"] = str(libero_root / "baseline_libero_config")
    sys.path[:0] = [str(a.oft_root), str(libero_root)]; os.chdir(a.oft_root)
    import imageio.v2 as imageio
    import torch
    import torch.nn as nn
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import prismatic, prismatic.models, prismatic.training, prismatic.vla
    prismatic.__path__.append(str(a.oft_root / "prismatic")); prismatic.models.__path__.insert(0, str(a.oft_root / "prismatic" / "models")); prismatic.training.__path__.append(str(a.oft_root / "prismatic" / "training")); prismatic.vla.__path__.insert(0, str(a.oft_root / "prismatic" / "vla"))
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_env
    from experiments.robot.libero.run_libero_eval import prepare_observation as official_prepare_observation
    from experiments.robot.robot_utils import set_seed_everywhere
    from experiments.robot.openvla_utils import normalize_proprio as official_normalize_proprio, prepare_images_for_vla
    device = torch.device("cuda")
    # Match the official evaluator before constructing the simulator and VLA.
    set_seed_everywhere(7)
    if a.deterministic_sdp:
      # Flash and memory-efficient SDPA can use reduction orders whose last
      # bits vary across identical GPU calls.  In a contact-rich MuJoCo
      # rollout those bits can amplify after tens of actions.  This setting
      # changes no policy inputs or control logic; it only makes the neural
      # forward pass suitable for repeatability checks.
      torch.backends.cuda.enable_flash_sdp(False)
      torch.backends.cuda.enable_mem_efficient_sdp(False)
      torch.backends.cuda.enable_math_sdp(True)
    model = AutoModelForVision2Seq.from_pretrained(str(a.checkpoint), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to(device).eval()
    model.vision_backbone.set_num_images_in_input(2); model.norm_stats = json.loads((a.checkpoint / "dataset_statistics.json").read_text())
    # A second copy of the 7B model's vision tower needs another ~8.6 GB and
    # cannot coexist with OFT+ on a 24 GB 4090.  The OpenVLA vision tower is
    # stateless in eval mode, so safely reuse it for SlotSSM feature extraction.
    # (The nominal evaluator has a separate raw-action path below.)
    slot_vision_backbone = model.vision_backbone.eval()
    processor = AutoProcessor.from_pretrained(str(a.checkpoint), trust_remote_code=True)
    actions_mod = dynamic_module("oft_action_heads_online", a.oft_root / "prismatic" / "models" / "action_heads.py"); projectors_mod = dynamic_module("oft_projectors_online", a.oft_root / "prismatic" / "models" / "projectors.py")
    head = actions_mod.L1RegressionActionHead(4096,4096,7); head.load_state_dict(load_torch_state(a.checkpoint / "action_head--150000_checkpoint.pt", torch)); head = head.to(device=device,dtype=torch.bfloat16).eval()
    proprio_projector = projectors_mod.ProprioProjector(4096,8); proprio_projector.load_state_dict(load_torch_state(a.checkpoint / "proprio_projector--150000_checkpoint.pt", torch)); proprio_projector=proprio_projector.to(device=device,dtype=torch.bfloat16).eval()
    s1=load_file(str(a.stage1_checkpoint)); tokenizer=SlotAttention(n_slots=16,in_dim=2176,feature_dim=512); tokenizer.load_state_dict({k.removeprefix("object_centric_tokenizer."):v for k,v in s1.items() if k.startswith("object_centric_tokenizer.")},strict=True); tokenizer=tokenizer.to(device=device,dtype=torch.bfloat16).eval()
    ssm=Stage2Dynamics("mamba2",residual_next=True); ssm.load_state_dict(load_file(str(a.stage2_checkpoint)),strict=True); ssm=ssm.to(device=device,dtype=torch.bfloat16).eval()
    adapter=OpenVLAOFTSlotSSMAdapter(head,action_hidden_dim=4096,slot_dim=512).to(device=device,dtype=torch.bfloat16); adapter.load_state_dict(load_file(str(a.adapter_checkpoint)),strict=False); adapter.eval()
    if a.gate_checkpoint is not None:
      adapter.load_state_dict(load_file(str(a.gate_checkpoint)), strict=False); adapter.eval()
    neural_residual = None
    if a.neural_recovery_residual_checkpoint is not None:
      neural_residual = NeuralRecoveryResidual().to(device=device, dtype=torch.bfloat16)
      neural_residual.load_state_dict(load_file(str(a.neural_recovery_residual_checkpoint)), strict=True)
      neural_residual.eval()
    class MacroVerifier(nn.Module):
      """Deployment copy of the small matched-FSM candidate scorer."""
      def __init__(self):
        super().__init__()
        self.action=nn.Sequential(nn.LayerNorm(4096),nn.Linear(4096,48),nn.GELU())
        self.slots=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,32),nn.GELU())
        self.robot=nn.Sequential(nn.LayerNorm(15),nn.Linear(15,16),nn.GELU())
        self.candidate=nn.Sequential(nn.Linear(7,24),nn.GELU())
        self.value=nn.Sequential(nn.LayerNorm(120),nn.Linear(120,48),nn.GELU(),nn.Dropout(0.15),nn.Linear(48,1))
      def forward(self, action_hidden, temporal_slots, proprio, previous_action, candidates):
        context=torch.cat((self.action(action_hidden.mean(1)),self.slots(temporal_slots.mean(1)),
                           self.robot(torch.cat((proprio,previous_action),dim=-1))),dim=-1)
        count=candidates.shape[1]; context=context[:,None].expand(-1,count,-1)
        scale=torch.tensor([0.03,0.03,0.105,0.004,12.0,1.0,1.0],device=candidates.device)
        return self.value(torch.cat((context,self.candidate(candidates/scale)),dim=-1)).squeeze(-1)
    candidate_verifier = None
    if a.candidate_verifier_checkpoint is not None:
      candidate_verifier=MacroVerifier().to(device).eval()
      candidate_verifier.load_state_dict(load_file(str(a.candidate_verifier_checkpoint)),strict=True)
    gate_num_classes = a.gate_num_classes if a.gate_num_classes is not None else (2 if a.gate_checkpoint is not None else 3)
    class PatchQueryGeometry(nn.Module):
      def __init__(self, patch_dim: int = 2176, action_dim: int = 4096, patch_count: int = 512, hidden: int = 256) -> None:
        super().__init__()
        self.action=nn.Sequential(nn.LayerNorm(action_dim),nn.Linear(action_dim,hidden),nn.GELU())
        self.patch=nn.Sequential(nn.LayerNorm(patch_dim),nn.Linear(patch_dim,hidden),nn.GELU())
        self.position=nn.Parameter(torch.zeros(1,patch_count,hidden)); self.camera=nn.Parameter(torch.zeros(1,patch_count,hidden))
        self.attention=nn.MultiheadAttention(hidden,8,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(2*hidden),nn.Linear(2*hidden,hidden),nn.GELU(),nn.Linear(hidden,3))
      def forward(self, patches, action_hidden):
        query=self.action(action_hidden.mean(dim=1))[:,None]; tokens=self.patch(patches)+self.position+self.camera
        selected,_=self.attention(query,tokens,tokens,need_weights=False)
        return self.head(torch.cat((query[:,0],selected[:,0]),dim=-1))
    patch_geometry = None
    if a.patch_geometry_checkpoint is not None:
      patch_geometry=PatchQueryGeometry().to(device=device,dtype=torch.bfloat16)
      patch_geometry.load_state_dict(load_file(str(a.patch_geometry_checkpoint)),strict=True); patch_geometry.eval()
    suite=benchmark.get_benchmark_dict()[a.task_suite](); task=suite.get_task(a.task_id); init=suite.get_task_init_states(a.task_id)[a.episode_id]
    env, text = get_libero_env(task, "openvla", resolution=256); norm_key=a.task_suite if a.task_suite in model.norm_stats else f"{a.task_suite}_no_noops"
    official_cfg = SimpleNamespace(
        model_family="openvla", num_images_in_input=2, use_proprio=True,
        unnorm_key=norm_key, center_crop=True,
    )
    def official_policy_query(obs):
      """Exact upstream OFT+ action call, retaining its action-token states."""
      observation, _ = official_prepare_observation(obs, 224)
      all_images = prepare_images_for_vla(
          [observation["full_image"], observation["wrist_image"]], official_cfg,
      )
      primary = all_images.pop(0)
      prompt = f"In: What action should the robot take to {text.lower()}?\nOut:"
      policy_inputs = processor(prompt, primary).to(device, dtype=torch.bfloat16)
      wrist_inputs = [processor(prompt, image).to(device, dtype=torch.bfloat16) for image in all_images]
      if wrist_inputs:
        policy_inputs["pixel_values"] = torch.cat(
            [policy_inputs["pixel_values"]] + [item["pixel_values"] for item in wrist_inputs], dim=1,
        )
      proprio = official_normalize_proprio(
          observation["state"], model.norm_stats[norm_key]["proprio"],
      )
      # The upstream evaluator does not wrap this call in autocast.  Disable
      # the surrounding SlotSSM autocast context to keep its numerics exact.
      with torch.inference_mode(False), torch.no_grad(), torch.autocast("cuda", enabled=False):
        action, action_hidden = model.predict_action(
            **policy_inputs, unnorm_key=norm_key, do_sample=False, proprio=proprio,
            proprio_projector=proprio_projector, noisy_action_projector=None,
            action_head=head, use_film=False,
        )
      return [action[index] for index in range(len(action))], action_hidden
    def input_(obs):
        prompt=f"In: What action should the robot take to {text.lower()}?\nOut:"
        x=processor(prompt, center_crop_resize(obs["agentview_image"][::-1,::-1],Image)).to(device,dtype=torch.bfloat16); w=processor(prompt,center_crop_resize(obs["robot0_eye_in_hand_image"][::-1,::-1],Image)).to(device,dtype=torch.bfloat16); x["pixel_values"]=torch.cat((x["pixel_values"],w["pixel_values"]),dim=1); return x
    def recovery_feature_query(inputs, proprio):
      """Get the OFT hidden state in the visual convention used in training."""
      # Normal OFT+ actions may use the upstream evaluator preprocessing, but
      # recovery datasets were frozen from ``input_``.  Mixing these two
      # states changes crop/orientation underneath the learned heads.
      _, action_hidden = model.predict_action(
          **inputs, unnorm_key=norm_key, do_sample=False, proprio=proprio,
          proprio_projector=proprio_projector, action_head=head, use_film=False,
      )
      return action_hidden
    frames=[]; decisions=[]; online_steps=[]; residual_events=[]; patches=deque(maxlen=a.history); slots=deque(maxlen=a.history); previous=None; q=deque(); success=False
    stem=f"task{a.task_id}-episode{a.episode_id}"
    rgb_dir = wrist_dir = None
    if a.save_online_images:
      rgb_dir, wrist_dir = a.output_dir / "rgb", a.output_dir / "wrist_rgb"
      rgb_dir.mkdir(parents=True, exist_ok=True); wrist_dir.mkdir(parents=True, exist_ok=True)
    controller_mode = RecoveryMode.NORMAL
    recovery_votes = complete_votes = 0
    was_forced_recovery = False
    previous_executed_action = np.zeros(7, dtype=np.float32)
    last_action_hidden = None
    anchor_fsm_phase, anchor_fsm_phase_steps = 0, 0
    learned_recovery_count = 0
    learned_recovery_steps = 0
    candidate_names=("fsm_xyp0_p0","fsm_xym25_p0","fsm_xyp0_m25","fsm_xyp0_p25","fsm_xyp25_p0")
    candidate_offsets=np.asarray([[0.,0.],[0.,-0.025],[0.,0.025],[-0.025,0.],[0.025,0.]],dtype=np.float32)
    selected_candidate_name = None
    selected_candidate_scores = None
    selected_recovery_mode = None
    try:
      obs=env.reset(); obs=env.set_init_state(init)
      max_total = (a.max_policy_steps if a.max_policy_steps is not None else 520) + 10
      for t in range(max_total):
        if t<10: obs,_,_,_=env.step([0,0,0,0,0,0,-1]); continue
        frames.append(obs["agentview_image"][::-1,::-1])
        rgb_path = wrist_path = None
        if rgb_dir is not None:
          rgb_path = rgb_dir / f"step-{t - 10:04d}.png"; wrist_path = wrist_dir / f"step-{t - 10:04d}.png"
          imageio.imwrite(rgb_path, obs["agentview_image"][::-1,::-1])
          imageio.imwrite(wrist_path, obs["robot0_eye_in_hand_image"][::-1,::-1])
        # Slots and the causal SlotSSM state are updated on *every* observation,
        # including the seven actions served from a cached normal chunk.
        inputs=input_(obs)
        residual_gate_probability = None
        residual_delta = None
        residual_applied = False
        proprio_np=normalized_proprio({"robot0_eef_pos":obs["robot0_eef_pos"],"robot0_eef_quat":obs["robot0_eef_quat"],"robot0_gripper_qpos":obs["robot0_gripper_qpos"]},model.norm_stats[norm_key]["proprio"],np)
        with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
          patch=slot_vision_backbone(inputs["pixel_values"]); current,_,_=tokenizer(patch,past_slots=previous); previous=current[:,None]
          patches.append(patch); slots.append(current)
          patch_seq=torch.stack(tuple(patches),dim=1); slot_seq=torch.stack(tuple(slots),dim=1); temporal,_,_=ssm(slot_seq,patch_seq,compute_window=False)
          # The expensive OFT language/action pass remains chunk-rate during
          # NORMAL, but RECOVERY always refreshes it after its one action.
          queried_policy = last_action_hidden is None or not q
          official_actions_at_query = None
          if queried_policy:
            # Run the baseline action query *before* the SlotSSM-side hidden
            # extraction.  The upstream OpenVLA implementation carries a
            # small amount of inference state, so reversing this order made
            # otherwise identical long rollouts drift after a few chunks.
            if a.official_normal_actions:
              official_actions_at_query, _ = official_policy_query(obs)
              # Preserve exactly upstream applied actions, while keeping the
              # gate/recovery features in the frozen-training convention.
              last_action_hidden = recovery_feature_query(inputs, proprio_np)
            else:
              last_action_hidden = recovery_feature_query(inputs, proprio_np)
          policy_step = t - 10
          forced_recovery = (a.force_recovery_at_step is not None and
                             a.force_recovery_at_step <= policy_step < a.force_recovery_at_step + a.force_recovery_steps)
          phase_override = None
          if forced_recovery and a.recovery_phase_schedule == "teacher_duration":
            recovery_index = policy_step - a.force_recovery_at_step
            phase = 0 if recovery_index < backoff_end else 1 if recovery_index < align_end else 2 if recovery_index < descend_end else 3 if recovery_index < close_end else 4
            phase_override = torch.tensor([phase], device=device, dtype=torch.long)
          out=adapter(last_action_hidden,temporal[:,-1],proprio=torch.as_tensor(proprio_np[None],device=device,dtype=torch.bfloat16),previous_action=torch.as_tensor(previous_executed_action[None],device=device,dtype=torch.bfloat16),recovery_phase_override=phase_override,disable_phase_conditioning=a.disable_recovery_phase_conditioning)
          # A separately trained two-class gate leaves the adapter's legacy
          # RECOVERY_COMPLETE row unsupervised.  Do not let that arbitrary
          # logit alter a binary deployment decision.
          probabilities=out["gate_logits"][..., :gate_num_classes].float().softmax(dim=-1)[0]
          if gate_num_classes == 2 and a.recovery_probability_threshold is not None:
            predicted_mode = int(RecoveryMode.RECOVERY) if float(probabilities[RecoveryMode.RECOVERY]) >= a.recovery_probability_threshold else int(RecoveryMode.NORMAL)
          else:
            predicted_mode=int(probabilities.argmax().item())
          gate_confidence=float(probabilities[predicted_mode].item())
          if predicted_mode != int(RecoveryMode.NORMAL) and gate_confidence < a.min_gate_confidence:
            predicted_mode=int(RecoveryMode.NORMAL)
          if (forced_recovery and a.save_forced_states_dir is not None and
              (policy_step - a.force_recovery_at_step) % a.save_forced_state_every == 0):
            # This privileged state is an offline teacher seed only.  The
            # online controller above never reads it.
            snapshot_dir = a.save_forced_states_dir / f"task{a.task_id}" / f"step{policy_step:04d}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(snapshot_dir / "mujoco_state.npz", mujoco_state=env.sim.get_state().flatten())
            (snapshot_dir / "manifest.json").write_text(json.dumps({
                "kind": "learned_oracle_recovery_state_v1", "teacher_only": True,
                "task_suite": a.task_suite, "task_id": a.task_id, "episode_id": a.episode_id,
                "fork_timestep": policy_step, "source": "learned_recovery_head_oracle_trigger",
            }, indent=2) + "\n")
          if forced_recovery:
            if not was_forced_recovery:
              anchor_fsm_phase, anchor_fsm_phase_steps = 0, 0
            controller_mode=RecoveryMode.RECOVERY; q.clear(); recovery_votes=0; complete_votes=0
          elif was_forced_recovery:
            # Oracle intervention has a known finite duration.  Resume OFT+
            # from a fresh chunk afterwards; no learned gate decision is used.
            controller_mode=RecoveryMode.NORMAL; q.clear(); last_action_hidden=None; recovery_votes=0; complete_votes=0
          elif controller_mode == RecoveryMode.NORMAL:
            recovery_votes = recovery_votes + 1 if predicted_mode == int(RecoveryMode.RECOVERY) else 0
            if recovery_votes >= a.recovery_confirm_steps and learned_recovery_count < a.max_learned_recoveries:
              controller_mode=RecoveryMode.RECOVERY; q.clear(); complete_votes=0
              anchor_fsm_phase, anchor_fsm_phase_steps = 0, 0
              learned_recovery_count += 1
              learned_recovery_steps = 0
              # A normal action chunk can have been queried several frames
              # ago.  Start recovery from this observation, as in every
              # offline recovery window, not from that stale hidden state.
              last_action_hidden = recovery_feature_query(inputs, proprio_np)
              queried_policy = True
              if candidate_verifier is not None:
                candidates=torch.tensor([[0.,0.,.105,.004,12.,0.,0.],
                                         [0.,-.025,.105,.004,12.,0.,0.],
                                         [0.,.025,.105,.004,12.,0.,0.],
                                         [-.025,0.,.105,.004,12.,0.,0.],
                                         [.025,0.,.105,.004,12.,0.,0.]],device=device,dtype=torch.float32)[None]
                with torch.inference_mode():
                  values=candidate_verifier(last_action_hidden.float(),temporal[:,-1].float(),
                    torch.as_tensor(proprio_np[None],device=device,dtype=torch.float32),
                    torch.as_tensor(previous_executed_action[None],device=device,dtype=torch.float32),candidates)[0]
                selected_index=int(values.argmax().item())
                geometry_xy_offset=candidate_offsets[selected_index].copy()
                selected_candidate_name=candidate_names[selected_index]
                selected_candidate_scores=values.float().cpu().tolist()
                selected_recovery_mode = torch.tensor([selected_index], device=device, dtype=torch.long)
              if a.save_learned_recovery_states_dir is not None:
                # Save the causal policy state *before* the first recovery
                # action.  Existing recovery data is seeded from nominal OFT+
                # forks; these are the distribution actually induced by the
                # learned gate, so they are the right source for an on-policy
                # correction pass.  The simulator state is strictly teacher-
                # only and never feeds back into this loop.
                snapshot_dir = (
                    a.save_learned_recovery_states_dir
                    / f"task{a.task_id}-episode{a.episode_id}"
                    / f"step{policy_step:04d}"
                )
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    snapshot_dir / "mujoco_state.npz",
                    mujoco_state=env.sim.get_state().flatten(),
                )
                (snapshot_dir / "manifest.json").write_text(json.dumps({
                    "kind": "learned_gate_recovery_state_v1",
                    "teacher_only": True,
                    "task_suite": a.task_suite,
                    "task_id": a.task_id,
                    "episode_id": a.episode_id,
                    "fork_timestep": policy_step,
                    "source": "learned_gate_recovery_trigger",
                    "gate_prediction": RecoveryMode(predicted_mode).name,
                    "gate_confidence": gate_confidence,
                    # This path is populated at the end of the rollout when
                    # --save-online-images is set.  Keeping it in the fork
                    # manifest lets feature extraction replay the exact
                    # causal visual/action prefix rather than resetting the
                    # SlotSSM state at a DAgger fork.
                    "source_steps": str((a.output_dir / "online_trajectory.jsonl").resolve()),
                }, indent=2) + "\n")
          else:
            complete_votes = (complete_votes + 1 if predicted_mode == int(RecoveryMode.RECOVERY_COMPLETE) else 0) if gate_num_classes == 3 else 0
            if (gate_num_classes == 3 and complete_votes >= a.complete_confirm_steps) or (
                gate_num_classes == 2 and learned_recovery_steps >= a.learned_recovery_max_steps
            ):
              controller_mode=RecoveryMode.NORMAL; q.clear(); last_action_hidden=None; recovery_votes=0
          # On a recovery-to-normal handoff, generate the new normal chunk
          # from this observation rather than reusing the last recovery query.
          if controller_mode == RecoveryMode.NORMAL and last_action_hidden is None:
            last_action_hidden = recovery_feature_query(inputs, proprio_np)
            out=adapter(last_action_hidden,temporal[:,-1],proprio=torch.as_tensor(proprio_np[None],device=device,dtype=torch.bfloat16),previous_action=torch.as_tensor(previous_executed_action[None],device=device,dtype=torch.bfloat16),disable_phase_conditioning=a.disable_recovery_phase_conditioning)
            queried_policy=True
            if a.official_normal_actions:
              official_actions_at_query, _ = official_policy_query(obs)
          if controller_mode == RecoveryMode.RECOVERY:
            if a.neural_recovery_mode:
              if selected_recovery_mode is None:
                raise RuntimeError("neural recovery entered without a selected recovery mode")
              # This is the deployment boundary: frozen visual/OFT/SlotSSM
              # features and a learned mode feed a neural action head.  No
              # task anchor, phase schedule, or FSM action is consulted.
              learned_phase_override = (
                  out["recovery_phase_logits"].argmax(dim=-1)
                  if a.neural_hard_phase_routing else None
              )
              out=adapter(last_action_hidden,temporal[:,-1],
                  proprio=torch.as_tensor(proprio_np[None],device=device,dtype=torch.bfloat16),
                  previous_action=torch.as_tensor(previous_executed_action[None],device=device,dtype=torch.bfloat16),
                  recovery_phase_override=learned_phase_override,
                  recovery_mode_override=selected_recovery_mode,
                  disable_phase_conditioning=a.disable_recovery_phase_conditioning,
                  geometry_conditioning=a.neural_geometry_conditioning)
            if (a.save_neural_recovery_states_dir is not None and not forced_recovery and
                    learned_recovery_steps % a.save_neural_recovery_state_every == 0):
              # Offline DAgger teacher seed.  This writes only simulator
              # state after the neural policy's own preceding actions; none
              # of it is an input to the policy above.
              snapshot_dir = (a.save_neural_recovery_states_dir / f"task{a.task_id}-episode{a.episode_id}" /
                              f"step{policy_step:04d}")
              snapshot_dir.mkdir(parents=True, exist_ok=True)
              np.savez_compressed(snapshot_dir / "mujoco_state.npz", mujoco_state=env.sim.get_state().flatten())
              (snapshot_dir / "manifest.json").write_text(json.dumps({
                  "kind": "neural_recovery_dagger_state_v1", "teacher_only": True,
                  "task_suite": a.task_suite, "task_id": a.task_id, "episode_id": a.episode_id,
                  "fork_timestep": policy_step, "recovery_step": learned_recovery_steps,
                  "source": "mode_conditioned_neural_recovery_policy",
                  "source_steps": str((a.output_dir / "online_trajectory.jsonl").resolve()),
              }, indent=2) + "\n")
            q.clear(); selected=out["recovery_actions"][0,:1]; selected_is_official_raw=False; cached_normal=False
            if neural_residual is not None and not forced_recovery:
              # The residual sees exactly the frozen online feature contract
              # and the learned candidate mode.  It has no access to teacher
              # state, anchors, phases, or simulator geometry.
              with torch.inference_mode():
                residual_out = neural_residual(
                    last_action_hidden,
                    temporal[:, -1],
                    torch.as_tensor(proprio_np[None], device=device, dtype=torch.bfloat16),
                    torch.as_tensor(previous_executed_action[None], device=device, dtype=torch.bfloat16),
                    selected_recovery_mode,
                )
              residual_gate_probability = float(torch.sigmoid(residual_out["gate_logit"])[0].float().cpu())
              residual_delta = residual_out["delta"][0].float().cpu().tolist()
              if residual_gate_probability >= a.neural_recovery_residual_threshold:
                selected = (selected + residual_out["delta"]).clamp(-1.0, 1.0)
                residual_applied = True
              residual_events.append({
                  "step": policy_step,
                  "mode": int(selected_recovery_mode[0].item()),
                  "gate_probability": residual_gate_probability,
                  "delta": residual_delta,
                  "applied": residual_applied,
              })
            recovery_index = (policy_step - a.force_recovery_at_step
                              if forced_recovery else learned_recovery_steps)
            if recovery_index < a.recovery_open_prefix_steps:
              # OFT gripper values above 0.5 decode to environment-open. This
              # is an explicit diagnostic phase prior, not a learned gate or
              # a simulator-state rule; reports retain the intervention tag.
              selected = selected.clone()
              selected[..., 6] = 1.0
          else:
            cached_normal=not queried_policy and bool(q)
            if not q:
              if a.official_normal_actions:
                if official_actions_at_query is None:
                  raise RuntimeError("official action chunk was not prepared at the policy query")
                # Keep upstream float64 action precision through env.step.
                # Downcasting here creates a tiny but accumulating MuJoCo
                # state drift in otherwise identical long rollouts.
                q.extend((np.asarray(action).copy(), True)
                         for action in official_actions_at_query[:a.normal_open_loop])
              else:
                q.extend((np.asarray(action, dtype=np.float32), False)
                         for action in out["normal_actions"][0,:a.normal_open_loop].float().cpu().numpy())
            selected_value, selected_is_official_raw = q.popleft()
            selected = (None if selected_is_official_raw
                        else torch.as_tensor(selected_value[None], device=device))
            cached_normal=not queried_policy
          if selected_is_official_raw:
            raw = np.asarray(selected_value).copy()
          else:
            normalized=selected.float().cpu().numpy(); raw=model._unnormalize_actions(normalized,norm_key)[0]
        backoff_template = bool(
            controller_mode == RecoveryMode.RECOVERY
            and forced_recovery
            and a.recovery_backoff_template
            and recovery_index < 6
        )
        geometry_controller = bool(
            controller_mode == RecoveryMode.RECOVERY
            and a.recovery_geometry_controller
            and (forced_recovery or a.recovery_anchor_fsm)
        )
        geometry_relative = None
        if geometry_controller:
          # This controller never accesses simulator object pose.  It receives
          # only the adapter's visual/proprio/history prediction, while the
          # fixed phase duration and oracle trigger are deliberately exposed
          # as an ablation.  Constants match ``local_regrasp_candidates``.
          geometry_relative = out["target_relative_position"][0].float().cpu().numpy()
          if patch_geometry is not None:
            # ``patch`` and ``last_action_hidden`` originate in the main
            # policy's inference-mode block.  Keep this auxiliary visual
            # query in inference mode as well; otherwise LayerNorm attempts
            # to save an inference tensor for backward at the first recovery
            # step.
            with torch.inference_mode():
              geometry_relative = patch_geometry(patch, last_action_hidden)[0].float().cpu().numpy()
          elif target_anchor is not None:
            # Task-specific anchor is an offline-trained parameter, not a
            # simulator query.  At deployment the only variable it consumes
            # is the robot's observed EEF position.
            geometry_relative = target_anchor - np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
          if a.recovery_anchor_fsm:
            # Task-specific target coordinates are learned offline.  This
            # state machine consumes only that parameter and robot proprio;
            # unlike the old duration-only program it cannot descend until
            # the observed EEF is actually aligned with the desired approach.
            phase_before = anchor_fsm_phase
            eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
            # Patch geometry is already expressed as target-minus-EEF.  Keep
            # the small learned grasp offset in that same relative frame, so
            # this branch has no hidden dependency on a task target anchor.
            desired_xy = (None if patch_geometry is not None
                          else target_anchor[:2] + geometry_xy_offset)
            if anchor_fsm_phase == 0:
              raw = np.asarray([0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
              transition = (anchor_fsm_phase_steps >= 4 if patch_geometry is not None
                            else anchor_fsm_phase_steps >= 4 and eef[2] >= target_anchor[2] + 0.10)
            elif anchor_fsm_phase == 1:
              xy_error = (geometry_relative[:2] + geometry_xy_offset
                          if patch_geometry is not None else desired_xy - eef[:2])
              raw = np.asarray([
                  np.clip(xy_error[0] / 0.0105, -0.65, 0.65),
                  np.clip(xy_error[1] / 0.0105, -0.65, 0.65),
                  0.0, 0.0, 0.0, 0.0, 1.0,
              ], dtype=np.float32)
              transition = anchor_fsm_phase_steps >= 2 and float(np.linalg.norm(xy_error)) <= 0.006
            elif anchor_fsm_phase == 2:
              z_error = geometry_relative[2] + 0.105 if patch_geometry is not None else target_anchor[2] + 0.105 - eef[2]
              raw = np.asarray([0.0, 0.0, np.clip(z_error / 0.0105, -0.65, 0.65),
                                0.0, 0.0, 0.0, 1.0], dtype=np.float32)
              transition = anchor_fsm_phase_steps >= 2 and abs(float(z_error)) <= 0.008
            elif anchor_fsm_phase == 3:
              raw = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
              transition = anchor_fsm_phase_steps >= 11
            elif anchor_fsm_phase == 4:
              raw = np.asarray([0.0, 0.0, 0.28, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
              # In the original lift-only ablation this terminal phase is
              # deliberately unbounded.  A placement anchor turns it into a
              # complete task macro: first lift high enough to safely clear
              # the caddy, then transport with closed fingers.
              transition = placement_anchor is not None and anchor_fsm_phase_steps >= 4 and eef[2] >= placement_anchor[2] + 0.105
            elif anchor_fsm_phase == 5:
              desired_xy = placement_anchor[:2]
              rotation_error = (quaternion_rotation_error(placement_quat, np.asarray(obs["robot0_eef_quat"], dtype=np.float32))
                                if placement_quat is not None else np.zeros(3, dtype=np.float32))
              raw = np.asarray([
                  np.clip((desired_xy[0] - eef[0]) / 0.0105, -0.65, 0.65),
                  np.clip((desired_xy[1] - eef[1]) / 0.0105, -0.65, 0.65),
                  0.0,
                  *np.clip(rotation_error / 0.10, -0.35, 0.35), 0.0,
              ], dtype=np.float32)
              transition = (anchor_fsm_phase_steps >= 2 and float(np.linalg.norm(desired_xy - eef[:2])) <= 0.006
                            and float(np.linalg.norm(rotation_error)) <= 0.10)
            elif anchor_fsm_phase == 6:
              rotation_error = (quaternion_rotation_error(placement_quat, np.asarray(obs["robot0_eef_quat"], dtype=np.float32))
                                if placement_quat is not None else np.zeros(3, dtype=np.float32))
              raw = np.asarray([0.0, 0.0, np.clip((placement_anchor[2] - eef[2]) / 0.0105, -0.65, 0.65),
                                *np.clip(rotation_error / 0.10, -0.35, 0.35), 0.0], dtype=np.float32)
              transition = (anchor_fsm_phase_steps >= 2 and eef[2] <= placement_anchor[2] + 0.006
                            and float(np.linalg.norm(rotation_error)) <= 0.10)
            elif anchor_fsm_phase == 7:
              # ``raw=1`` decodes to the environment's open command.  Keep
              # this phase explicit so that release is not delegated back to
              # an OFT chunk generated from an out-of-distribution recovery
              # state.
              raw = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
              transition = anchor_fsm_phase_steps >= 7
            else:
              raw = np.asarray([0.0, 0.0, 0.28, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
              # Completion is observable from the finite macro itself; until
              # RECOVERY_COMPLETE has dense labels, this avoids keeping the
              # robot in recovery forever after a successful release.
              transition = placement_anchor is not None and anchor_fsm_phase_steps >= 6
            if transition and anchor_fsm_phase < (8 if placement_anchor is not None else 4):
              anchor_fsm_phase += 1; anchor_fsm_phase_steps = 0
            else:
              anchor_fsm_phase_steps += 1
            if anchor_fsm_phase == 8 and transition:
              controller_mode = RecoveryMode.NORMAL; q.clear(); last_action_hidden = None
          elif recovery_index < backoff_end:
            raw = np.asarray([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            phase_before = None
          elif recovery_index < align_end:
            raw = np.asarray([
                np.clip((geometry_relative[0] + geometry_xy_offset[0]) / (0.0105 * 10), -0.85, 0.85),
                np.clip((geometry_relative[1] + geometry_xy_offset[1]) / (0.0105 * 10), -0.85, 0.85),
                0.0, 0.0, 0.0, 0.0, 1.0,
            ], dtype=np.float32)
            phase_before = None
          elif recovery_index < descend_end:
            raw = np.asarray([
                0.0, 0.0,
                np.clip((geometry_relative[2] - (0.0105 * 6 * 0.30)) / (0.0105 * 12), -0.70, -0.10),
                0.0, 0.0, 0.0, 1.0,
            ], dtype=np.float32)
            phase_before = None
          elif recovery_index < close_end:
            raw = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            phase_before = None
          else:
            raw = np.asarray([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            phase_before = None
        if backoff_template:
          # Every recorded teacher branch starts with this collision-avoiding
          # primitive.  It is a transparent diagnostic safety action, not a
          # learned prediction or a simulator-state query.
          raw = np.asarray([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        phase_probabilities = out["recovery_phase_logits"].float().softmax(dim=-1)[0]
        decisions.append({"step":t-10,"mode":controller_mode.name,"forced_recovery":forced_recovery,"neural_recovery_mode":a.neural_recovery_mode,"phase_open_prior":bool(controller_mode == RecoveryMode.RECOVERY and recovery_index < a.recovery_open_prefix_steps),"backoff_template":backoff_template,"geometry_controller":geometry_controller,"patch_geometry_controller":patch_geometry is not None,"anchor_fsm_phase":phase_before if geometry_controller and a.recovery_anchor_fsm else None,"task_specific_target_anchor":None if target_anchor is None else target_anchor.tolist(),"task_specific_placement_anchor":None if placement_anchor is None else placement_anchor.tolist(),"task_specific_placement_quat":None if placement_quat is None else placement_quat.tolist(),"online_observation":None if rgb_path is None else {"rgb":str(rgb_path),"wrist_rgb":str(wrist_path),"proprio":{"robot0_eef_pos":np.asarray(obs["robot0_eef_pos"]).tolist(),"robot0_eef_quat":np.asarray(obs["robot0_eef_quat"]).tolist(),"robot0_gripper_qpos":np.asarray(obs["robot0_gripper_qpos"]).tolist()}},"geometry_xy_offset":geometry_xy_offset.tolist(),"selected_candidate_name":selected_candidate_name,"selected_candidate_scores":selected_candidate_scores,"recovery_phase_durations":phase_durations,"predicted_target_relative_position":(out["spatial_target_relative_position"][0].float().cpu().tolist() if a.neural_geometry_conditioning else (None if geometry_relative is None else geometry_relative.tolist())),"phase_schedule":a.recovery_phase_schedule if forced_recovery else "none","gate_num_classes":gate_num_classes,"gate_prediction":RecoveryMode(predicted_mode).name,"gate_confidence":gate_confidence,"gate_logits":out["gate_logits"][0].float().cpu().tolist(),"recovery_phase_prediction":int(phase_probabilities.argmax().item()),"recovery_phase_confidence":float(phase_probabilities.max().item()),"action":raw.tolist(),"cached_normal_chunk":cached_normal,"policy_queried":queried_policy,"recovery_votes":recovery_votes,"complete_votes":complete_votes,"learned_recovery_count":learned_recovery_count,"learned_recovery_steps":learned_recovery_steps})
        if controller_mode == RecoveryMode.RECOVERY and not forced_recovery:
          learned_recovery_steps += 1
        # Preserve upstream OFT+ action precision on the normal path.  A
        # float32 cast here is enough to make MuJoCo diverge after tens of
        # otherwise bit-identical actions.  Learned recovery tensors are
        # already float32, so this leaves their behaviour unchanged.
        env_action=np.asarray(raw).copy(); env_action[-1] = -np.sign(2.0 * env_action[-1] - 1.0)
        if rgb_path is not None:
          online_steps.append({
              "online_observation": {
                  "rgb": str(rgb_path.resolve()),
                  "wrist_rgb": str(wrist_path.resolve()),
                  "proprio": {
                      "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"]).tolist(),
                      "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"]).tolist(),
                      "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).tolist(),
                  },
              },
              "executed_action": env_action.tolist(),
          })
        obs,_,done,_=env.step(env_action.tolist())
        previous_executed_action=env_action
        was_forced_recovery = forced_recovery
        if done: success=True; break
    finally: env.close()
    a.output_dir.mkdir(parents=True,exist_ok=True); imageio.mimsave(a.output_dir/f"{stem}.mp4",frames,fps=30)
    if online_steps:
      (a.output_dir / "online_trajectory.jsonl").write_text(
          "".join(json.dumps(row) + "\n" for row in online_steps)
      )
    report={"success":success,"task":text,"frames":len(frames),"mode_counts":{m:sum(d["mode"]==m for d in decisions) for m in [x.name for x in RecoveryMode]},"residual_events":residual_events,"decisions":decisions,"video":str(a.output_dir/f"{stem}.mp4")}; (a.output_dir/f"{stem}.json").write_text(json.dumps(report,indent=2)); print(json.dumps({k:v for k,v in report.items() if k!="decisions"},indent=2))
if __name__=="__main__": main()
