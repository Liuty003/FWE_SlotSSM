from __future__ import annotations

import unittest

from recovery.detection.failure_trigger import visual_lift_drop_evidence


class FailureTriggerTest(unittest.TestCase):
    def test_visual_injection_is_causal_lift_evidence(self) -> None:
        had_lifted, dropped = visual_lift_drop_evidence(
            peak_visual_rise=17.1,
            current_visual_rise=10.0,
            lift_threshold_px=20.0,
            drop_return_px=16.0,
            injection_was_visual_lift_triggered=True,
        )
        self.assertTrue(had_lifted)
        self.assertTrue(dropped)

    def test_fixed_step_injection_does_not_invent_a_lift(self) -> None:
        had_lifted, dropped = visual_lift_drop_evidence(
            peak_visual_rise=2.0,
            current_visual_rise=1.0,
            lift_threshold_px=20.0,
            drop_return_px=16.0,
            injection_was_visual_lift_triggered=False,
        )
        self.assertFalse(had_lifted)
        self.assertTrue(dropped)

    def test_missing_current_measurement_cannot_confirm_drop(self) -> None:
        had_lifted, dropped = visual_lift_drop_evidence(
            peak_visual_rise=25.0,
            current_visual_rise=None,
            lift_threshold_px=20.0,
            drop_return_px=16.0,
            injection_was_visual_lift_triggered=False,
        )
        self.assertTrue(had_lifted)
        self.assertFalse(dropped)


if __name__ == "__main__":
    unittest.main()
