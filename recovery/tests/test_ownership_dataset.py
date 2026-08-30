import json
from pathlib import Path

from scripts.build_ownership_dataset import build, recovery_chunk


def test_recovery_chunk_requires_complete_actions():
    rows = [{"online_observation": {"recovery_action": [0] * 7}} for _ in range(8)]
    assert len(recovery_chunk(rows, 0, 8)) == 8
    assert recovery_chunk(rows, 1, 8) is None


def test_builder_emits_gate_boundary_and_non_crossing_actions(tmp_path: Path):
    factory, task = tmp_path / "factory", tmp_path / "factory/tasks/task538-episode0"
    teacher = task / "feedback-teacher-v1"
    teacher.mkdir(parents=True)
    raw = []
    for step in range(12):
        raw.append({
            "online_observation": {
                "rgb": f"rgb-{step}", "wrist_rgb": f"wrist-{step}",
                "depth": f"depth-{step}", "wrist_depth": f"wdepth-{step}",
                "proprio": {}, "recovery_action": [0, 0, 0, 0, 0, 0, -1],
            },
            "training_only": {"instance_segmentation": f"instance-{step}"},
        })
    (teacher / "raw-transitions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw))
    (teacher / "metadata.json").write_text(json.dumps({
        "success": True, "failure_timestep": 50, "source_fork": "fork",
        "task_suite": "libero_10", "task_id": 538, "episode_id": 0,
        "task_language": "task", "agentview_calibration": {}, "open_loop_steps": 8,
    }))
    (task / "handoff-manifest.json").write_text(json.dumps({"certified_handoff": 9}))
    factory.mkdir(exist_ok=True)
    (factory / "factory-report.json").write_text(json.dumps({
        "records": [{"task_id": 538, "episode": 0, "status": "certified"}]}))
    output = tmp_path / "dataset"
    report = build(factory, output, action_horizon=8)
    states = [json.loads(line) for line in (output / "state_dataset.jsonl").read_text().splitlines()]
    assert report["state_rows"] == 12
    assert states[8]["gate_label"] == 1 and states[8]["action_mask"] == 0
    assert states[9]["gate_label"] == 0 and states[9]["action_mask"] == 0
    assert states[0]["action_mask"] == 1
    assert json.loads((output / "handoff_manifest.json").read_text())["records"][0]["certified_local_handoff"] == 9
