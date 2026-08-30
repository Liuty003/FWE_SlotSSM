#!/usr/bin/env python3
"""Measure V4.3 action imitation on exact teacher states by recovery phase."""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recovery.data.v4_relation_features import load
from recovery.models.v4_relational_recovery import V4ModelConfig
from recovery.models.v43_local_recovery import V43LocalRecovery
from recovery.models.v43_geometry_aux_recovery import V43GeometryAuxRecovery

PHASE={
 "release_distractor":"clear_gripper","release_and_clear_bowl":"clear_gripper",
 "approach_bowl":"approach_grasp","descend_bowl":"approach_grasp",
 "close_bowl":"grasp_lift","lift_bowl":"grasp_lift",
 "transfer_bowl":"transfer_place","lower_bowl":"transfer_place","release_bowl":"transfer_place",
 "retreat_from_bowl":"transfer_place","approach_drawer":"close_drawer",
 "contact_drawer":"close_drawer","feedback_push_drawer_closed":"close_drawer",
 "reacquire_drawer":"close_drawer","recontact_drawer":"close_drawer"}

def metadata(feature):
 r=json.loads(feature.with_suffix('.json').read_text());stride=int(r.get('recovery_sampling_stride') or r.get('decision_stride',4));out=[]
 for rec in r.get('source_records',r.get('sources')):
  source=Path(rec['source']);m=json.loads((source/'metadata.json').read_text());teacher=Path(m['source_teacher']);tm=json.loads((teacher/'metadata.json').read_text())
  raw=[json.loads(x) for x in (teacher/'raw-transitions.jsonl').read_text().splitlines() if x.strip()]; phases=tm.get('phase_by_step',[])
  starts=list(range(0,len(raw)-8+1,stride))
  if len(starts)!=int(rec['rows']):raise ValueError((source,len(starts),rec['rows']))
  for i in starts:
   q=raw[i].get('teacher_only',{}); phase=PHASE.get(phases[i],phases[i] if phases else 'unknown')
   out.append((phase,bool(q.get('attached_before',False)),int(m.get('task_id',-1))))
 return out

def inp(b):return {k:getattr(b,k) for k in ('visual_tokens','action_hidden','token_points','token_valid','token_image_xy','gripper_position','gripper_image_xy')}

def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,nargs='+',required=True);p.add_argument('--v1',type=Path,required=True);p.add_argument('--aux',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=32);a=p.parse_args()
 import torch
 import torch.nn.functional as F
 from safetensors.torch import load_file
 fs=[load(x) for x in a.features];meta=sum((metadata(x) for x in a.features),[])
 cfg=V4ModelConfig(visual_dim=fs[0].visual_tokens.shape[-1],action_hidden_dim=fs[0].action_hidden.shape[-1],action_chunk=8,action_dim=7,num_slots=16,slot_dim=512,relation_dim=512,num_heads=8,representation='relation_3d')
 models={'v1':V43LocalRecovery(cfg,direct_head_version='pooled_chunk_mlp_v3').to(a.device).eval(),'aux':V43GeometryAuxRecovery(cfg).to(a.device).eval()}
 models['v1'].load_state_dict(load_file(str(a.v1),device=a.device));models['aux'].load_state_dict(load_file(str(a.aux),device=a.device))
 stats={name:defaultdict(lambda:torch.zeros(7,dtype=torch.float64)) for name in models};offset=0
 with torch.inference_mode():
  for f in fs:
   for start in range(0,f.num_rows,a.batch_size):
    ix=torch.arange(start,min(start+a.batch_size,f.num_rows));b=f.index(ix).to(a.device);target=b.recovery_targets.float()
    for name,model in models.items():
     pred=model(**inp(b))['recovery_actions'].float(); first_cos=F.cosine_similarity(pred[:,0,:3],target[:,0,:3],dim=-1); grip=(pred[...,6]>=0.5).eq(target[...,6]>=0.5).float().mean(1)
     l1=(pred-target).abs(); trans=l1[...,:3].mean((1,2));rot=l1[...,3:6].mean((1,2));all_l1=l1.mean((1,2))
     for j,row in enumerate(ix.tolist()):
      phase,attached,_=meta[offset+row]; key=f'{phase}|attached={int(attached)}';stats[name][key]+=torch.tensor([1,all_l1[j],trans[j],rot[j],grip[j],first_cos[j],float(target[j,:,6].ge(0).any())],dtype=torch.float64)
   offset+=f.num_rows
 report={'rows':len(meta),'phase_counts':{},'models':{}}
 for key in sorted(set(k for d in stats.values() for k in d)):
  report['phase_counts'][key]=int(stats['v1'][key][0])
 for name,d in stats.items():
  report['models'][name]={}
  for key,v in sorted(d.items()):
   n=v[0].item();report['models'][name][key]={'rows':int(n),'action_l1':v[1].item()/n,'translation_l1':v[2].item()/n,'rotation_l1':v[3].item()/n,'gripper_sign_accuracy':v[4].item()/n,'first_translation_cosine':v[5].item()/n,'rows_with_open_target_fraction':v[6].item()/n}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
