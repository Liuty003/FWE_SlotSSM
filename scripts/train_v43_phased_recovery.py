#!/usr/bin/env python3
"""Train phase classifier and phase-specific action heads on pure chunks."""

import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--features-root',type=Path,required=True);ap.add_argument('--init-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--steps',type=int,default=10000);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--device',default='cuda');a=ap.parse_args()
 import torch
 from safetensors.torch import load_file,save_file
 from recovery.data.v4_relation_features import load
 from recovery.models.v4_relational_recovery import V4ModelConfig
 from recovery.models.v43_phased_recovery import V43PhasedRecovery,PHASE_NAMES
 cfg=json.loads((a.init_dir/'config.json').read_text());model=V43PhasedRecovery(V4ModelConfig(**cfg['model']),direct_head_version=cfg.get('direct_head_version','context_action_query_v2')).to(a.device)
 missing,unexpected=model.load_state_dict(load_file(str(a.init_dir/'best.safetensors')),strict=False)
 expected_missing={f'phase_classifier.{x}' for x in ('0.weight','0.bias','1.weight','1.bias','3.weight','3.bias')}|{f'phase_actions.{i}.{k}' for i in range(5) for k in model.direct_action.state_dict()}
 if unexpected or set(missing)!=expected_missing: raise RuntimeError((missing,unexpected))
 for head in model.phase_actions: head.load_state_dict(model.direct_action.state_dict())
 for p in model.parameters(): p.requires_grad=False
 for module in (model.phase_classifier,model.phase_actions):
  for p in module.parameters():p.requires_grad=True
 data=[load(a.features_root/f'{name}.safetensors') for name in PHASE_NAMES]
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr);gen=torch.Generator().manual_seed(20260821)
 history=[];best=float('inf');a.output_dir.mkdir(parents=True,exist_ok=True)
 keys=('visual_tokens','action_hidden','token_points','token_valid','token_image_xy','gripper_position','gripper_image_xy')
 for step in range(1,a.steps+1):
  labels=torch.randint(len(data),(a.batch_size,),generator=gen);parts=[]
  for label in labels.tolist(): parts.append(data[label].index(torch.randint(data[label].num_rows,(1,),generator=gen)))
  inputs={k:torch.cat([getattr(x,k) for x in parts]).to(a.device) for k in keys};targets=torch.cat([x.recovery_targets for x in parts]).to(a.device);labels=labels.to(a.device)
  model.train();out=model(**inputs);loss,pl,al=model.phased_loss(out,targets,labels);opt.zero_grad();loss.backward();opt.step()
  if step==1 or step%250==0:
   model.eval();correct=0;total=0;action=0
   with torch.inference_mode():
    for label,d in enumerate(data):
     idx=torch.linspace(0,d.num_rows-1,min(128,d.num_rows)).long();b=d.index(idx).to(a.device);o=model(**{k:getattr(b,k) for k in keys});correct+=int((o['predicted_phase']==label).sum());total+=len(idx);action+=float(torch.nn.functional.l1_loss(o['phase_recovery_candidates'][:,label],b.recovery_targets))*len(idx)
   metric=action/total;row={'step':step,'train_total':float(loss),'train_phase':float(pl),'train_action':float(al),'phase_accuracy':correct/total,'action_l1':metric};history.append(row);print(json.dumps(row),flush=True)
   if metric<best:best=metric;save_file({k:v.detach().cpu() for k,v in model.state_dict().items()},str(a.output_dir/'best.safetensors'))
 save_file({k:v.detach().cpu() for k,v in model.state_dict().items()},str(a.output_dir/'last.safetensors'))
 (a.output_dir/'config.json').write_text(json.dumps({**cfg,'kind':'v43_phased_recovery_v1','phase_names':PHASE_NAMES},indent=2)+'\n');(a.output_dir/'report.json').write_text(json.dumps({'best_action_l1':best,'history':history,'rows':{n:d.num_rows for n,d in zip(PHASE_NAMES,data)}},indent=2)+'\n')
if __name__=='__main__':main()
