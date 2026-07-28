#!/usr/bin/env python
"""Evaluate matched task-aware advantages on a held-out task."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from scripts.train_task_aware_action_scorer import load_split
from scripts.train_task_aware_advantage_scorer import TaskAwareAdvantageScorer, evaluate


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",required=True); p.add_argument("--progress-dir",required=True)
    p.add_argument("--checkpoint",required=True); p.add_argument("--task-id",type=int,default=9)
    p.add_argument("--episode-ids",default="0,1,2,3,4"); p.add_argument("--gate-margin-m",type=float,default=.001)
    p.add_argument("--output",required=True); args=p.parse_args(); episodes=[int(x) for x in args.episode_ids.split(",") if x]
    data=load_split(Path(args.data_dir),Path(args.progress_dir),[args.task_id],episodes)
    model=TaskAwareAdvantageScorer(torch.zeros(7),torch.ones(7),torch.zeros(7),torch.ones(7),torch.ones(7))
    model.load_state_dict(load_file(args.checkpoint),strict=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    metrics=evaluate(model,data,device,args.gate_margin_m)
    result={"task_id":args.task_id,"episode_ids":episodes,"num_forks":len(data["before_slots"]),"gate_margin_m":args.gate_margin_m,
            "metrics":metrics,"units":{"progress_regret_advantage":"millimeters","accuracy":"percent","auc":"unitless"}}
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); print(f"saved={output}")


if __name__=="__main__": main()
