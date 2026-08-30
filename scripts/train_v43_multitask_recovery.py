#!/usr/bin/env python3
"""Train phase/state auxiliaries and phase-conditioned recovery actions."""
from __future__ import annotations
import argparse,json,sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recovery.data.v4_relation_features import REQUIRED_KEYS,V4RelationFeatures,load
from recovery.models.v4_relational_recovery import V4ModelConfig
from recovery.models.v43_multitask_recovery import V43MultitaskRecovery
from recovery.models.v43_auxiliary_state_recovery import V43AuxiliaryStateRecovery
from recovery.models.v43_auxiliary_conditioned_recovery import V43AuxiliaryConditionedRecovery
from recovery.models.v43_completion_recovery import V43CompletionRecovery
from recovery.models.v43_detailed_phase_recovery import V43DetailedPhaseRecovery,DETAILED_PHASE_NAMES
def cat(parts,torch):return V4RelationFeatures(**{k:torch.cat([getattr(x,k) for x in parts]) for k in REQUIRED_KEYS})
def inp(b):return {k:getattr(b,k) for k in ('visual_tokens','action_hidden','token_points','token_valid','token_image_xy','gripper_position','gripper_image_xy')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,nargs='+',required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--init-checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--action-routing',choices=('phase_experts','unified','aux_conditioned','completion_conditioned','detailed'),default='phase_experts');p.add_argument('--steps',type=int,default=5000);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--eval-every',type=int,default=100);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=20260821);a=p.parse_args()
 import torch
 import torch.nn.functional as F
 from safetensors.torch import load_file,save_file
 torch.manual_seed(a.seed);f=cat([load(x) for x in a.features],torch);lab=load_file(str(a.labels));cfg=V4ModelConfig(visual_dim=f.visual_tokens.shape[-1],action_hidden_dim=f.action_hidden.shape[-1],action_chunk=8,action_dim=7,num_slots=16,slot_dim=512,relation_dim=512,num_heads=8,representation='relation_3d');classes={'phase_experts':V43MultitaskRecovery,'unified':V43AuxiliaryStateRecovery,'aux_conditioned':V43AuxiliaryConditionedRecovery,'completion_conditioned':V43CompletionRecovery,'detailed':V43DetailedPhaseRecovery};m=classes[a.action_routing](cfg).to(a.device);res=m.load_state_dict(load_file(str(a.init_checkpoint),device=a.device),strict=False);new_prefix=(('phase_head.','state_head.','drawer_head.','phase_action_experts.','gripper_head.') if a.action_routing=='phase_experts' else ('phase_head.','state_head.','drawer_head.','unified_action_residual.','gripper_head.','target_delta_head.','task_trunk.','aux_to_action.','completion_head.'));expected={n for n,_ in m.named_parameters() if n.startswith(new_prefix)}
 # Support both bootstrapping from the geometry model (all task heads are new)
 # and DAgger fine-tuning from a complete multitask checkpoint.
 if not set(res.missing_keys).issubset(expected) or res.unexpected_keys:raise ValueError(res)
 for n,x in m.named_parameters():x.requires_grad_(n in expected)
 train=[x for x in m.parameters() if x.requires_grad];opt=torch.optim.AdamW(train,lr=a.lr,weight_decay=1e-4);phase=lab['phase_id'];attached=lab['attached'];sample_table=(torch.tensor([2.,2.,3.,4.,8.,6.,2.,4.,8.,3.,2.,3.,2.,3.,3.]) if a.action_routing=='detailed' else torch.tensor([2.,3.,5.,1.,1.]));sampling=sample_table[phase];sampling*=torch.where(attached<.5,torch.tensor(1.5),torch.tensor(1.));sampling/=sampling.sum();phase_counts=torch.bincount(phase,minlength=len(sample_table)).float();phase_weights=(phase_counts.sum()/phase_counts.clamp_min(1)).clamp(max=10);phase_weights/=phase_weights.mean();completion_target=lab.get('completion',torch.zeros_like(attached));completion_positive=completion_target.sum().clamp_min(1);completion_pos_weight=((len(attached)-completion_positive)/completion_positive).clamp(max=30).to(a.device);a.output_dir.mkdir(parents=True,exist_ok=True);history=[];best=float('inf')
 def losses(ix):
  b=f.index(ix).to(a.device);o=m(**inp(b));ph=phase[ix].to(a.device);states=torch.stack((lab['attached'][ix],lab['contact'][ix],lab['bilateral_contact'][ix]),1).to(a.device);delta=lab['target_delta_world'][ix].to(a.device);dv=lab['drawer_qpos_valid'][ix].to(a.device);drawer=lab['drawer_qpos'][ix].to(a.device);target=b.recovery_targets.float();continuous=F.l1_loss(o['recovery_actions'][...,:6],target[...,:6]);gripper=F.binary_cross_entropy_with_logits(o['gripper_logits'],target[...,6]);pl=F.cross_entropy(o['phase_logits'],ph,weight=phase_weights.to(a.device));state=F.binary_cross_entropy_with_logits(o['state_logits'],states);geo=F.smooth_l1_loss(o['predicted_target_delta'],delta);dl=o['predicted_drawer_qpos'].sum()*0 if not dv.any() else F.smooth_l1_loss(o['predicted_drawer_qpos'][dv],drawer[dv]);lo=torch.tensor([-0.68839288,-0.80892855,-0.9375],device=a.device);hi=torch.tensor([0.9375,0.87053573,0.9375],device=a.device);pred_raw=.5*(o['recovery_actions'][:,0,:3]+1)*(hi-lo)+lo;teacher_raw=.5*(target[:,0,:3]+1)*(hi-lo)+lo;teacher_cos=F.cosine_similarity(teacher_raw,delta);direction_phase=((ph==2)|(ph==3)) if a.action_routing=='detailed' else ph.eq(1);direction_mask=direction_phase&states[:,0].lt(.5)&teacher_cos.gt(.2);direction=pred_raw.sum()*0 if not direction_mask.any() else F.relu(.2-F.cosine_similarity(pred_raw[direction_mask],delta[direction_mask])).mean();completion=continuous*0 if 'completion_logit' not in o else F.binary_cross_entropy_with_logits(o['completion_logit'],completion_target[ix].to(a.device),pos_weight=completion_pos_weight);total=continuous+2*gripper+pl+state+4*geo+.25*dl+.5*direction+completion;return total,dict(action=continuous,gripper=gripper,phase=pl,state=state,geometry=geo,drawer=dl,direction=direction,completion=completion)
 def evaluate():
  m.eval();s={k:0. for k in ('action','gripper','phase','state','geometry','drawer','direction','completion')};count=0
  with torch.inference_mode():
   for st in range(0,f.num_rows,a.batch_size):
    ix=torch.arange(st,min(st+a.batch_size,f.num_rows));_,z=losses(ix);n=len(ix);count+=n
    for k,v in z.items():s[k]+=float(v)*n
  m.train();return {k:v/count for k,v in s.items()}
 m.train()
 for step in range(1,a.steps+1):
  ix=torch.multinomial(sampling,a.batch_size,replacement=True);total,z=losses(ix);opt.zero_grad(set_to_none=True);total.backward();torch.nn.utils.clip_grad_norm_(train,1.);opt.step()
  if step==1 or step%a.eval_every==0:
   e=evaluate();score=e['action']+2*e['gripper']+.25*e['phase'];row={'step':step,'eval':e,'score':score};history.append(row);print(json.dumps(row),flush=True)
   if score<best:best=score;save_file({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},str(a.output_dir/'best.safetensors'))
 save_file({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},str(a.output_dir/'last.safetensors'))
 def serial(v):return str(v) if isinstance(v,Path) else [serial(x) for x in v] if isinstance(v,list) else v
 fr=json.loads(a.features[0].with_suffix('.json').read_text());kind={'phase_experts':'v43_multitask_recovery_v1','unified':'v43_auxiliary_state_recovery_v1','aux_conditioned':'v43_auxiliary_conditioned_recovery_v1','completion_conditioned':'v43_completion_recovery_v1','detailed':'v43_detailed_phase_recovery_v1'}[a.action_routing];phase_names=(DETAILED_PHASE_NAMES if a.action_routing=='detailed' else ['clear_gripper','approach_grasp','grasp_lift','transfer_place','close_drawer']);config={'kind':kind,'model':asdict(cfg),'direct_head_version':'pooled_chunk_mlp_v3','phase_names':phase_names,'gate_preservation':'exact frozen base model','args':{k:serial(v) for k,v in vars(a).items()},'feature_provenance':{k:fr.get(k) for k in ('oft_checkpoint','action_head_sha256','proprio_projector_sha256','action_normalization_key','recovery_target_coordinates','decoder_action_contract','image_preprocessing','decision_grid_contract','recovery_target_contract')}};(a.output_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n');(a.output_dir/'report.json').write_text(json.dumps({'best_score':best,'history':history},indent=2)+'\n')
if __name__=='__main__':main()
