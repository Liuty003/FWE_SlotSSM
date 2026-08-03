#!/usr/bin/env python
"""Train canonical candidate futures from frozen causal SlotSSM state."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from torch.utils.data import DataLoader,Dataset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recovery.models.shared_risk_slotssm import SharedRiskSlotSSM
from recovery.models.canonical_future_dynamics import CanonicalFutureDynamics
from scripts.train_temporal_unified_candidate import Data
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--risk-checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--validation-groups',required=True);p.add_argument('--steps',type=int,default=1600);a=p.parse_args();torch.manual_seed(20260803)
 rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]; val=set(a.validation_groups.split(','))
 def build(rs):
  base=Data(rs,32); out=[]
  for item,r in zip(base.items,rs):
   x=load_file(r['slot_file']);perm=torch.tensor(r['source_role_permutation']); geo=x['role_geometry'][:,perm]; target=geo[-1]; attached=float(r['label_teacher_only']['attached_after']);lift=float(r['label_teacher_only']['regrasp_verified']);out.append((*item,target,attached,lift))
  return out
 tr,va=build([r for r in rows if r['source_group_id'] not in val]),build([r for r in rows if r['source_group_id'] in val])
 def col(b):return {'ctx':torch.stack([x[0] for x in b]),'act':torch.stack([x[1] for x in b]),'task':torch.tensor([x[2] for x in b]),'y':torch.tensor([x[3] for x in b]),'g':[x[4] for x in b],'geo':torch.stack([x[5] for x in b]),'att':torch.tensor([x[6] for x in b]),'lift':torch.tensor([x[7] for x in b])}
 dev=torch.device('cuda');risk=SharedRiskSlotSSM().to(dev);risk.load_state_dict(load_file(a.risk_checkpoint));risk.eval();[q.requires_grad_(False) for q in risk.parameters()];m=CanonicalFutureDynamics().to(dev);o=torch.optim.AdamW(m.parameters(),lr=3e-4);tl=DataLoader(tr,16,shuffle=True,collate_fn=col);vl=DataLoader(va,16,collate_fn=col);it=iter(tl);best=-1.;hist=[];a.output_dir.mkdir(parents=True,exist_ok=True)
 for s in range(1,a.steps+1):
  try:b=next(it)
  except StopIteration:it=iter(tl);b=next(it)
  with torch.no_grad():st=risk(b['ctx'].to(dev),b['task'].to(dev))['temporal_state'][:,-1]
  cur=b['ctx'][:,-1,288:297].to(dev).view(-1,3,3);z=m(st,b['act'].to(dev),b['task'].to(dev),cur);loss=torch.nn.functional.smooth_l1_loss(z['future_geometry'],b['geo'].to(dev))+torch.nn.functional.binary_cross_entropy_with_logits(z['attached_logits'],b['att'].to(dev))+torch.nn.functional.binary_cross_entropy_with_logits(z['lifted_logits'],b['lift'].to(dev))+torch.nn.functional.binary_cross_entropy_with_logits(z['candidate_value'],b['y'].to(dev));o.zero_grad();loss.backward();o.step()
  if s==1 or s%100==0 or s==a.steps:
   groups={}
   with torch.no_grad():
    for b in vl:
     st=risk(b['ctx'].to(dev),b['task'].to(dev))['temporal_state'][:,-1];cur=b['ctx'][:,-1,288:297].to(dev).view(-1,3,3);score=m(st,b['act'].to(dev),b['task'].to(dev),cur)['candidate_value'].sigmoid().cpu().tolist()
     for g,y,v in zip(b['g'],b['y'].tolist(),score):groups.setdefault(g,[]).append((y,v))
   ok={g:bool(max(v,key=lambda x:x[1])[0]) for g,v in groups.items()};acc=sum(ok.values())/len(ok);hist.append({'step':s,'loss':float(loss),'group_accuracy':acc});print(json.dumps(hist[-1]),flush=True)
   if acc>best:best=acc;save_file({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},a.output_dir/'canonical-future-best.safetensors')
 (a.output_dir/'report.json').write_text(json.dumps({'kind':'canonical_action_conditioned_future_dynamics','validation_groups':sorted(val),'history':hist},indent=2)+'\n')
if __name__=='__main__':main()
