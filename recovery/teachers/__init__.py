"""Simulator-assisted labelers; outputs are supervision, never policy inputs."""

from recovery.teachers.recovery_labeler import label_episode

__all__ = ["label_episode"]
from recovery.teachers.sim_contact import extract_target_gripper_contact

__all__ = ["extract_target_gripper_contact"]
