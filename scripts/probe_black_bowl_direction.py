#!/usr/bin/env python3
"""Probe whether frozen V4.3 context localizes the black bowl from policy inputs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def labels(report_path, stride=4):
 report=json.loads(report_path.read_text()); values=[]; first=[]; offset=0
 for rec in report['sources']:
  source=Path(rec['source']);meta=json.loads((source/'metadata.json').read_text())
  teacher=Path(meta['source_teacher']);raw=[json.loads(x) for x in (teacher/'raw-transitions.jsonl').read_text().splitlines() if x.strip()]
  starts=list(range(0,len(raw)-8+1,stride))
  if len(starts)!=rec['rows']:raise ValueError((source,len(starts),rec['rows']))
  first.append(offset)
  for i in starts:
   row=raw[i];target=row['teacher_only']['target_position_before'];eef=row['online_observation']['proprio']['robot0_eef_pos']
   values.append([target[j]-eef[j] for j in range(3)])
  offset+=rec['rows']
 return values,first

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train-features',type=Path,required=True);ap.add_argument('--test-features',type=Path,required=True);ap.add_argument('--adapter-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda');a=ap.parse_args()
 import torch
 from safetensors.torch import load_file
 from recovery.data.v4_relation_features import load
 from recovery.models.v43_local_recovery import V43LocalRecovery
 from recovery.models.v4_relational_recovery import V4ModelConfig
 cfg=json.loads((a.adapter_dir/'config.json').read_text());model=V43LocalRecovery(V4ModelConfig(**cfg['model']),direct_head_version=cfg['direct_head_version']).to(a.device).eval();model.load_state_dict(load_file(str(a.adapter_dir/'best.safetensors'),device=a.device))
 keys=('visual_tokens','action_hidden','token_points','token_valid','token_image_xy','gripper_position','gripper_image_xy')
 def context(path):
  d=load(path);out=[]
  with torch.inference_mode():
   for s in range(0,d.num_rows,32):
    b=d.index(torch.arange(s,min(s+32,d.num_rows))).to(a.device);o=model(**{k:getattr(b,k) for k in keys});out.append(o['relational_context'].mean(1).cpu())
  return torch.cat(out)
 train=context(a.train_features);test=context(a.test_features)
 yl,_=labels(a.train_features.with_suffix('.json'));yt,first=labels(a.test_features.with_suffix('.json'));y=torch.tensor(yl);ty=torch.tensor(yt)
 mean=train.mean(0);std=train.std(0).clamp_min(.01);ym=y.mean(0);ys=y.std(0).clamp_min(.01)
 probe=torch.nn.Sequential(torch.nn.Linear(train.shape[1],256),torch.nn.GELU(),torch.nn.Linear(256,3)).to(a.device);opt=torch.optim.AdamW(probe.parameters(),lr=1e-3)
 x=((train-mean)/std).to(a.device);yn=((y-ym)/ys).to(a.device)
 for _ in range(3000):
  idx=torch.randint(len(x),(128,),device=a.device);pred=probe(x[idx]);loss=torch.nn.functional.mse_loss(pred,yn[idx]);opt.zero_grad();loss.backward();opt.step()
 def metrics(feat,target,indices=None):
  if indices is not None:feat=feat[indices];target=target[indices]
  with torch.inference_mode():pred=(probe(((feat-mean)/std).to(a.device)).cpu()*ys+ym)
  err=(pred-target).norm(dim=1);cos=torch.nn.functional.cosine_similarity(pred,target,dim=1)
  return {'rows':len(err),'mean_position_error_m':float(err.mean()),'median_position_error_m':float(err.median()),'direction_cosine_mean':float(cos.mean()),'wrong_hemisphere_rate':float((cos<0).float().mean()),'within_3cm_rate':float((err<.03).float().mean())}
 result={'kind':'black_bowl_direction_probe_v1','train_in_sample':metrics(train,y),'dagger_all':metrics(test,ty),'dagger_activation_states':metrics(test,ty,torch.tensor(first))}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
