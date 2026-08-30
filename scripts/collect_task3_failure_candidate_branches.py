#!/usr/bin/env python
"""Task-3 adapter for the shared relative-regrasp branch collector.

Task 3 has the same recovery primitive as task 9: the target is a single
rigid object and recovery starts from a recorded natural-failure state.  Keep
the branch format identical so unified SlotSSM training can consume both.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import collect_task9_failure_candidate_branches as shared


shared.TASK_ID = 3
shared.TARGET_KEY = "akita_black_bowl_1_pos"
shared.TARGET_NAME = "akita_black_bowl_1"
shared.TARGET_REGION = None


if __name__ == "__main__":
    shared.main()
