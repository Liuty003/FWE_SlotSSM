"""Simulator-only contact labels for recovery data collection.

Nothing in this module is a legal online policy input.  The labels are derived
from MuJoCo contacts and are intentionally kept on the teacher side of the data
contract.
"""

from __future__ import annotations

from typing import Any


def _geom_name(model: Any, geom_id: int) -> str:
    name = model.geom_id2name(int(geom_id))
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return "" if name is None else str(name)


def _finger_side(name: str) -> str | None:
    lowered = name.lower()
    if "gripper" not in lowered and "finger" not in lowered:
        return None
    if "left" in lowered or "finger1" in lowered:
        return "left"
    if "right" in lowered or "finger2" in lowered:
        return "right"
    return "unknown"


def extract_target_gripper_contact(
    sim: Any,
    *,
    target_name: str,
) -> tuple[dict[str, Any], bool]:
    """Return rich target/gripper contacts and a conservative pinch label.

    ``attached_truth`` is true only when the target contacts both identifiable
    finger sides in the current simulator state.  It is a pinch/contact proxy,
    not a statement that MuJoCo welded the object to the gripper.
    """

    target_token = target_name.lower()
    pairs: list[dict[str, Any]] = []
    finger_sides: set[str] = set()
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        first = _geom_name(sim.model, contact.geom1)
        second = _geom_name(sim.model, contact.geom2)
        first_target = target_token in first.lower()
        second_target = target_token in second.lower()
        first_finger = _finger_side(first)
        second_finger = _finger_side(second)
        if first_target and second_finger is not None:
            target_geom, gripper_geom, side = first, second, second_finger
        elif second_target and first_finger is not None:
            target_geom, gripper_geom, side = second, first, first_finger
        else:
            continue
        finger_sides.add(side)
        pairs.append(
            {
                "target_geom": target_geom,
                "gripper_geom": gripper_geom,
                "finger_side": side,
            }
        )
    identifiable = {side for side in finger_sides if side in {"left", "right"}}
    attached = identifiable == {"left", "right"}
    state = {
        "target_name": target_name,
        "target_gripper_contact": bool(pairs),
        "bilateral_finger_contact": attached,
        "finger_sides": sorted(finger_sides),
        "pairs": pairs,
        "num_pairs": len(pairs),
    }
    return state, attached
