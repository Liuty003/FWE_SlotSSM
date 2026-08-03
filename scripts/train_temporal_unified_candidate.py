#!/usr/bin/env python
"""Train a candidate-future head from causal shared SlotSSM state only."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from recovery.models.shared_risk_slotssm import SharedRiskSlotSSM
from recovery.models.temporal_candidate_head import TemporalCandidateHead
from scripts.train_shared_failure_risk_slotssm import observable_inputs

def args():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--risk-checkpoint',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--validation-groups',required=True); p.add_argument('--steps',type=int,default=1200); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--window',type=int,default=32); return p.parse_args()
def history_path(task, ep):
 base=Path('/data2/lty/naive-e-slotssm-runs')
 if task==6:
  root=base/('task6-natural-baseline-role-sequences-v2-selected' if ep in (10,12,13) else 'task6-natural-baseline-role-sequences-all-v1')
 else: root=base/('task9-natural-baseline-role-sequences-v2-selected' if ep in (10,13) else 'task9-natural-baseline-role-sequences-all-v1')
 return root/f'task{task}-episode{ep}-execution-roles.safetensors'
class Data(Dataset):
 def __init__(self, rows, window):
  cache={}; self.items=[]
  for r in rows:
   task,ep,t=int(r['task_id']),int(r['source_episode_id']),int(r['failure_timestep']); key=(task,ep)
   if key not in cache:
    x=load_file(history_path(task,ep)); cache[key]=observable_inputs(task,x)
   h=cache[key]; end=min(t,len(h)-1); start=max(0,end-window+1); context=h[start:end+1]
   if len(context)<window: context=torch.cat((context[:1].expand(window-len(context),-1),context))
   x=load_file(r['slot_file'])['recovery_actions'].float(); summary=torch.cat((x.mean(0),x.std(0),x[-1],x.abs().max(0).values))
   self.items.append((context,summary,0 if task==6 else 1,float(r['label_teacher_only']['regrasp_verified']),r['source_group_id']))
 def __len__(self): return len(self.items)
 def __getitem__(self,i): return self.items[i]
def collate(b):
 return {'context':torch.stack([x[0] for x in b]),'action':torch.stack([x[1] for x in b]),'task':torch.tensor([x[2] for x in b]),'label':torch.tensor([x[3] for x in b]),'group':[x[4] for x in b]}
@torch.inference_mode()
def ev(risk,head,loader,dev):
 risk.eval(); head.eval(); groups={}
 for b in loader:
  state=risk(b['context'].to(dev),b['task'].to(dev))['temporal_state'][:,-1]
  score=head(state,b['action'].to(dev),b['task'].to(dev)).sigmoid().cpu().tolist()
  for g,y,s in zip(b['group'],b['label'].tolist(),score): groups.setdefault(g,[]).append((y,s))
 correct={}
 for g,v in groups.items():
  y,s=max(v,key=lambda z:z[1]); anygood=any(x[0] for x in v); correct[g]=(bool(y) if s>=.5 else not anygood)
 return {'group_policy_accuracy_with_abstain':sum(correct.values())/len(correct),'groups':correct}
def main():
 a=args(); torch.manual_seed(20260803); rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]; val=set(filter(None,a.validation_groups.split(',')))
 train=Data([r for r in rows if r['source_group_id'] not in val],a.window); valid=Data([r for r in rows if r['source_group_id'] in val],a.window)
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); risk=SharedRiskSlotSSM().to(dev); risk.load_state_dict(load_file(a.risk_checkpoint)); risk.eval()
 for p in risk.parameters(): p.requires_grad_(False)
 head=TemporalCandidateHead().to(dev); opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-3); pos=sum(x[3] for x in train.items); weight=torch.tensor([(len(train)-pos)/max(pos,1)],device=dev)
 tl=DataLoader(train,batch_size=a.batch_size,shuffle=True,collate_fn=collate); vl=DataLoader(valid,batch_size=a.batch_size,shuffle=False,collate_fn=collate); it=iter(tl); best=-1.; hist=[]; a.output_dir.mkdir(parents=True,exist_ok=True)
 for step in range(1,a.steps+1):
  try:b=next(it)
  except StopIteration: it=iter(tl); b=next(it)
  with torch.no_grad(): state=risk(b['context'].to(dev),b['task'].to(dev))['temporal_state'][:,-1]
  logits=head(state,b['action'].to(dev),b['task'].to(dev)); loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,b['label'].to(dev),pos_weight=weight); opt.zero_grad(); loss.backward(); opt.step()
  if step==1 or step%100==0 or step==a.steps:
   report=ev(risk,head,vl,dev); row={'step':step,'train_bce':float(loss),'validation':report}; hist.append(row); print(json.dumps(row),flush=True)
   if report['group_policy_accuracy_with_abstain']>best: best=report['group_policy_accuracy_with_abstain']; save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},a.output_dir/'temporal-candidate-best.safetensors')
 (a.output_dir/'report.json').write_text(json.dumps({'kind':'frozen_shared_risk_slotssm_temporal_candidate_head','input_contract':'32 causal pre-failure RGB-slot/action windows plus candidate action block; no post-candidate slots','validation_groups':sorted(val),'history':hist},indent=2)+'\n')
if __name__=='__main__': main()
