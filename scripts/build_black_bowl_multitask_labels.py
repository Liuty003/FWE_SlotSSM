#!/usr/bin/env python3
"""Build row-aligned privileged auxiliary labels for black-bowl recovery."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from safetensors.torch import save_file

PHASE_GROUPS={
 "release_distractor":0,"release_and_clear_bowl":0,
 "approach_bowl":1,"descend_bowl":1,
 "close_bowl":2,"lift_bowl":2,
 "transfer_bowl":3,"lower_bowl":3,"release_bowl":3,"retreat_from_bowl":3,
 "approach_drawer":4,"contact_drawer":4,"feedback_push_drawer_closed":4,
 "reacquire_drawer":4,"recontact_drawer":4}
PHASE_NAMES=["clear_gripper","approach_grasp","grasp_lift","transfer_place","close_drawer"]
DETAILED_PHASE_NAMES=["release_distractor","release_and_clear_bowl","approach_bowl","descend_bowl","close_bowl","lift_bowl","transfer_bowl","lower_bowl","release_bowl","retreat_from_bowl","approach_drawer","contact_drawer","feedback_push_drawer_closed","reacquire_drawer","recontact_drawer"]

def rows(path,detailed=False):
 report=json.loads(path.with_suffix('.json').read_text());stride=int(report.get('recovery_sampling_stride') or report.get('decision_stride',4));out=[]
 for rec in report.get('source_records',report.get('sources')):
  sm=json.loads((Path(rec['source'])/'metadata.json').read_text());teacher=Path(sm['source_teacher']);tm=json.loads((teacher/'metadata.json').read_text())
  raw=[json.loads(x) for x in (teacher/'raw-transitions.jsonl').read_text().splitlines() if x.strip()];phases=tm['phase_by_step'];drawer={int(x['transition']):float(x['qpos']) for x in tm.get('drawer_joint_history',[])}
  source_start=int(sm.get('source_teacher_start',0));source_end=int(sm.get('source_teacher_end',len(raw)))
  starts=list(range(source_start,source_end-8+1,stride))
  if len(starts)!=int(rec['rows']):raise ValueError((teacher,len(starts),rec['rows']))
  for i in starts:
   r=raw[i];q=r['teacher_only'];c=q.get('contact_before',{});target=q['target_position_before'];eef=r['online_observation']['proprio']['robot0_eef_pos']
   phase_id=DETAILED_PHASE_NAMES.index(phases[i]) if detailed else PHASE_GROUPS[phases[i]]
   completion=any(x.get('teacher_only',{}).get('task_success_after',False) for x in raw[i:i+8])
   out.append({'phase':phase_id,'attached':float(q.get('attached_before',False)),
    'contact':float(c.get('target_gripper_contact',False)),'bilateral':float(c.get('bilateral_finger_contact',False)),
    'delta':[target[j]-eef[j] for j in range(3)],'drawer_qpos':drawer.get(i,float('nan')),
    'completion':float(completion)})
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--features',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--detailed',action='store_true');a=p.parse_args();rr=sum((rows(x,a.detailed) for x in a.features),[]);phase_names=DETAILED_PHASE_NAMES if a.detailed else PHASE_NAMES
 q=torch.tensor([x['drawer_qpos'] for x in rr],dtype=torch.float32);valid=torch.isfinite(q);q=torch.nan_to_num(q)
 payload={'phase_id':torch.tensor([x['phase'] for x in rr],dtype=torch.int64),'attached':torch.tensor([x['attached'] for x in rr]),
  'contact':torch.tensor([x['contact'] for x in rr]),'bilateral_contact':torch.tensor([x['bilateral'] for x in rr]),
  'target_delta_world':torch.tensor([x['delta'] for x in rr]),'drawer_qpos':q,'drawer_qpos_valid':valid,
  'completion':torch.tensor([x['completion'] for x in rr])}
 a.output.parent.mkdir(parents=True,exist_ok=True);save_file(payload,str(a.output));report={'kind':('black_bowl_detailed_labels_v1' if a.detailed else 'black_bowl_multitask_labels_v2'),'rows':len(rr),'phase_names':phase_names,'phase_counts':torch.bincount(payload['phase_id'],minlength=len(phase_names)).tolist(),'attached_positive':int(payload['attached'].sum()),'contact_positive':int(payload['contact'].sum()),'bilateral_positive':int(payload['bilateral_contact'].sum()),'completion_positive':int(payload['completion'].sum()),'drawer_qpos_rows':int(valid.sum()),'inputs':[str(x) for x in a.features]};a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
