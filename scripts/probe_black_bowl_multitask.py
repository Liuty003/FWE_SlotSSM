#!/usr/bin/env python3
"""In-sample probes for policy-visible recovery state in frozen V4.3 features."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recovery.data.v4_relation_features import load
from recovery.models.v4_relational_recovery import V4ModelConfig
from recovery.models.v43_geometry_aux_recovery import V43GeometryAuxRecovery

def inp(b):return {k:getattr(b,k) for k in ('visual_tokens','action_hidden','token_points','token_valid','token_image_xy','gripper_position','gripper_image_xy')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,nargs='+',required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--adapter-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=3000);a=p.parse_args()
 import torch
 import torch.nn.functional as F
 from safetensors.torch import load_file
 fs=[load(x) for x in a.features];lab=load_file(str(a.labels));cfgj=json.loads((a.adapter_dir/'config.json').read_text());model=V43GeometryAuxRecovery(V4ModelConfig(**cfgj['model'])).to(a.device).eval();model.load_state_dict(load_file(str(a.adapter_dir/'best.safetensors'),device=a.device));xx=[]
 with torch.inference_mode():
  for f in fs:
   for s in range(0,f.num_rows,32):
    b=f.index(torch.arange(s,min(s+32,f.num_rows))).to(a.device);o=model(**inp(b));xx.append(torch.cat((o['relational_context'].mean(1),o['recovery_action_query'].mean(1)),1).cpu())
 x=torch.cat(xx);mean=x.mean(0);std=x.std(0).clamp_min(.01);x=((x-mean)/std).to(a.device);n=len(x)
 class Probe(torch.nn.Module):
  def __init__(self):super().__init__();self.body=torch.nn.Sequential(torch.nn.Linear(x.shape[1],512),torch.nn.GELU(),torch.nn.Linear(512,256),torch.nn.GELU());self.phase=torch.nn.Linear(256,5);self.binary=torch.nn.Linear(256,3);self.delta=torch.nn.Linear(256,3);self.drawer=torch.nn.Linear(256,1)
  def forward(self,z):h=self.body(z);return self.phase(h),self.binary(h),self.delta(h),self.drawer(h).squeeze(1)
 q=Probe().to(a.device);opt=torch.optim.AdamW(q.parameters(),lr=1e-3);phase=lab['phase_id'].to(a.device);binary=torch.stack((lab['attached'],lab['contact'],lab['bilateral_contact']),1).to(a.device);delta=lab['target_delta_world'].to(a.device);drawer=lab['drawer_qpos'].to(a.device);valid=lab['drawer_qpos_valid'].to(a.device);counts=torch.bincount(phase,minlength=5).float();weights=(counts.sum()/counts).clamp(max=10);weights/=weights.mean()
 for _ in range(a.steps):
  ix=torch.randint(n,(128,),device=a.device);ph,bi,de,dr=q(x[ix]);loss=F.cross_entropy(ph,phase[ix],weight=weights)+F.binary_cross_entropy_with_logits(bi,binary[ix])+4*F.smooth_l1_loss(de,delta[ix]);dv=valid[ix];loss+=0 if not dv.any() else F.smooth_l1_loss(dr[dv],drawer[ix][dv]);opt.zero_grad();loss.backward();opt.step()
 with torch.inference_mode():
  ph,bi,de,dr=q(x);pc=ph.argmax(1);predb=bi.sigmoid()>=.5;targetb=binary>=.5;err=(de-delta).norm(dim=1);dv=valid
  result={'kind':'black_bowl_multitask_probe_v1','rows':n,'in_sample_only':True,'phase_accuracy':float(pc.eq(phase).float().mean()),'phase_recall':[float(pc[phase.eq(i)].eq(i).float().mean()) for i in range(5)],'attached_accuracy':float(predb[:,0].eq(targetb[:,0]).float().mean()),'contact_accuracy':float(predb[:,1].eq(targetb[:,1]).float().mean()),'bilateral_accuracy':float(predb[:,2].eq(targetb[:,2]).float().mean()),'delta_mean_l1_m':float(err.mean()),'drawer_qpos_mae':float((dr[dv]-drawer[dv]).abs().mean()),'drawer_rows':int(dv.sum())}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
