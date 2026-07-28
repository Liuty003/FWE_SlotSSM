"""Causal observable evidence used by the recovery data collector."""

from __future__ import annotations


def visual_lift_drop_evidence(
    *,
    peak_visual_rise: float,
    current_visual_rise: float | None,
    lift_threshold_px: float,
    drop_return_px: float,
    injection_was_visual_lift_triggered: bool,
) -> tuple[bool, bool]:
    """Return whether a lift happened and the target has subsequently dropped.

    A visual-lift-triggered intervention is itself proof that the configured
    injection threshold was crossed. Requiring a second, higher lift threshold
    after that event creates an impossible detector band.
    """

    had_lifted = bool(
        peak_visual_rise >= lift_threshold_px
        or injection_was_visual_lift_triggered
    )
    has_dropped = bool(
        current_visual_rise is not None
        and current_visual_rise <= drop_return_px
    )
    return had_lifted, has_dropped
