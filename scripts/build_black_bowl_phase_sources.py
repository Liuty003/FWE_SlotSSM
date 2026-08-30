#!/usr/bin/env python3
"""Split successful black-bowl teachers into phase-pure V4 action sources."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PHASE = {
    "release_distractor": "clear_gripper", "release_and_clear_bowl": "clear_gripper",
    "approach_bowl": "approach_grasp", "descend_bowl": "approach_grasp",
    "close_bowl": "grasp_lift", "lift_bowl": "grasp_lift",
    "transfer_bowl": "transfer_place", "lower_bowl": "transfer_place",
    "release_bowl": "transfer_place", "retreat_from_bowl": "transfer_place",
    "approach_drawer": "close_drawer", "contact_drawer": "close_drawer",
    "feedback_push_drawer_closed": "close_drawer", "reacquire_drawer": "close_drawer",
    "recontact_drawer": "close_drawer",
}
DETAILED_PHASE_NAMES = [
    "release_distractor", "release_and_clear_bowl", "approach_bowl", "descend_bowl",
    "close_bowl", "lift_bowl", "transfer_bowl", "lower_bowl", "release_bowl",
    "retreat_from_bowl", "approach_drawer", "contact_drawer",
    "feedback_push_drawer_closed", "reacquire_drawer", "recontact_drawer",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-roots", nargs="+", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--detailed", action="store_true")
    args = ap.parse_args()
    counts, sources, rejected = Counter(), [], []
    for root in args.teacher_roots:
        for mp in sorted(root.rglob("metadata.json")):
            meta=json.loads(mp.read_text())
            if meta.get("success") is not True: continue
            rawp=mp.parent/"raw-transitions.jsonl"
            # The output root may live below a teacher root.  Generated phase
            # sources contain steps.jsonl, not raw teacher transitions, and
            # must not be recursively treated as teachers on a rebuild.
            if not rawp.is_file(): continue
            raw=[json.loads(x) for x in rawp.read_text().splitlines() if x.strip()]
            phases=meta.get("phase_by_step",[])
            if len(raw)!=len(phases):
                rejected.append({"source":str(mp.parent),"reason":"phase_length_mismatch"});continue
            mapped=phases if args.detailed else [PHASE.get(x) for x in phases]
            if any(x is None for x in mapped):
                rejected.append({"source":str(mp.parent),"reason":"unknown_phase"});continue
            start=0
            while start<len(raw):
                end=start+1
                while end<len(raw) and mapped[end]==mapped[start]: end+=1
                phase=mapped[start]
                if end-start>=args.horizon:
                    identity=f"task{meta['task_id']}-episode0-{mp.parent.name}-s{start:04d}-{phase}"
                    out=args.output_root/phase/identity;out.mkdir(parents=True,exist_ok=True)
                    rows=[]
                    for local,row in enumerate(raw[start:end]):
                        action=row["online_observation"]["recovery_action"]
                        rows.append({"kind":"black_bowl_phase_transition","timestep":local,
                                     "phase":phase,"phase_id":((DETAILED_PHASE_NAMES if args.detailed else list(sorted(set(PHASE.values())))).index(phase)),
                                     "gate_label":"RECOVERY","online_observation":row["online_observation"],
                                     "training_only":row.get("training_only",{}),"executed_action":action})
                    (out/"steps.jsonl").write_text("".join(json.dumps(x,separators=(",",":"))+"\n" for x in rows))
                    sm={**meta,"kind":"black_bowl_phase_pure_recovery_source_v1","schema":"v4_rgbd_rollout_v2",
                        "success":True,"fork_timestep":0,"wait_steps":0,"open_loop_steps":8,
                        "transitions":len(rows),"recovery_phase":phase,
                        "source_teacher_start":start,"source_teacher_end":end,
                        "rollout_id":f"black-bowl-phase:{identity}","source_teacher":str(mp.parent.resolve())}
                    (out/"metadata.json").write_text(json.dumps(sm,indent=2)+"\n")
                    counts[phase]+=len(rows)-args.horizon+1
                    sources.append({"phase":phase,"source":str(out),"steps":len(rows)})
                start=end
    report={"kind":("black_bowl_detailed_phase_sources_v1" if args.detailed else "black_bowl_phase_sources_v1"),"horizon":args.horizon,
            "phase_names":(DETAILED_PHASE_NAMES if args.detailed else sorted(set(PHASE.values()))),
            "sources":len(sources),"complete_chunks":dict(counts),"rejected":rejected}
    args.output_root.mkdir(parents=True,exist_ok=True)
    (args.output_root/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
