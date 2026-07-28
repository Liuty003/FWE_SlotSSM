# Recovery Expert Dataset Schema

## Purpose

The dataset supervises a learned short-horizon recovery controller. Simulator
object state is retained only as a teacher/debug field and must not be exposed
to the learned policy input.

## Episode layout

```text
task{task_id}-episode{episode_id}/
├── transitions.jsonl
├── rgb/step-NNNN.png
├── depth/step-NNNN.npz
└── target_masks/step-NNNN.npz
```

Each JSONL row is aligned with the observation immediately before
`expert_or_executed_action`.

Collector schema version 2 always writes RGB and depth when expert collection
is enabled, and automatically enables instance segmentation so a 64×64 target
mask is available even when the runtime failure detector uses privileged
height. Stage-B collection continues to save its tracked policy-visible mask.

## Required fields

- `task_suite`, `task_id`, `episode_id`, `timestep`
- `phase`
- `is_failure_injection`
- `is_failure_detection_event`
- `is_recovery_expert_action`
- `recovery_completed`
- `rgb`: oriented 256×256 agent-view RGB
- `depth`: oriented 256×256 MuJoCo depth buffer, stored as float32 `depth`
- `target_mask_64`: Stage-B predicted and temporally tracked mask, stored as
  uint8 `mask`
- `target_mask_measurement`: centroid, visible area, proposal index, tracking
  mode, and selector margin
- `robot_state`: policy-observable proprioception
- `nominal_action`: OpenVLA action from the same observation
- `expert_or_executed_action`: environment-convention action actually executed
- `success_after`

## Teacher-only field

`teacher_privileged_object_state` may be used to generate oracle actions,
validate depth calibration, and measure label quality. It must be removed from
the model input pipeline and from any claim of vision-only recovery.

`teacher_contact_state` records the exact MuJoCo target/gripper contact geom
pairs before the action. `teacher_attached_truth` is a conservative bilateral
finger-contact proxy. Both are simulator-only labels and must never enter
policy inputs.

## Initial training target

Train only rows with `is_recovery_expert_action=true` for recovery behavior
cloning. Use the preceding nominal/failure rows as temporal context and as
negative examples for mode selection. Split by episode, never by adjacent
frames from the same episode.

The action target can initially be:

```text
delta_action = expert_or_executed_action - nominal_action
```

This preserves OpenVLA as the base policy and learns a bounded recovery
residual. A later SlotSSM version should consume a short history of target
slots/masks, robot state, nominal actions, and recovery mode.
