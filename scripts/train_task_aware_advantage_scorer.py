#!/usr/bin/env python
"""Train matched candidate-vs-nominal task-progress advantages."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from scripts.train_failure_physical_scorer import binary_auc, normalization, targets
from scripts.train_task_aware_action_scorer import ids, load_split


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--progress-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-tasks", default="0,1,2,3,4,5,6,7"); p.add_argument("--train-episodes", default="0,1,2,3")
    p.add_argument("--val-task", type=int, default=8); p.add_argument("--val-episodes", default="0,1,2,3,4")
    p.add_argument("--steps", type=int, default=5000); p.add_argument("--batch-forks", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--ranking-temperature-m", type=float, default=0.005)
    p.add_argument("--physical-weight", type=float, default=0.25); p.add_argument("--classification-weight", type=float, default=0.2)
    p.add_argument("--advantage-weight", type=float, default=1.0); p.add_argument("--beat-nominal-weight", type=float, default=0.5)
    p.add_argument("--ranking-weight", type=float, default=1.0); p.add_argument("--beat-margin-m", type=float, default=0.001)
    p.add_argument("--eval-every", type=int, default=200); p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20); p.add_argument("--seed", type=int, default=20260721)
    return p.parse_args()


def delta_std(data):
    actions = data["executed_actions"].float()
    return (actions - actions[:, :1]).reshape(-1, 7).std(0).clamp_min(1e-3)


class TaskAwareAdvantageScorer(torch.nn.Module):
    def __init__(self, action_mean, action_std, joint_mean, joint_std, action_delta_std, horizon=4, dim=512):
        super().__init__(); self.horizon = horizon
        for name, value in (("action_mean", action_mean), ("action_std", action_std), ("joint_mean", joint_mean),
                            ("joint_std", joint_std), ("action_delta_std", action_delta_std)):
            self.register_buffer(name, value)
        self.slot_norm = torch.nn.LayerNorm(dim)
        self.action_encoder = torch.nn.Sequential(torch.nn.Linear(7,256),torch.nn.GELU(),torch.nn.Linear(256,dim))
        self.delta_encoder = torch.nn.Sequential(torch.nn.Linear(7,256,bias=False),torch.nn.GELU(),torch.nn.Linear(256,dim,bias=False))
        self.joint_encoder = torch.nn.Sequential(torch.nn.Linear(7,256),torch.nn.GELU(),torch.nn.Linear(256,dim))
        self.language_embedding = torch.nn.Embedding(4096,128,padding_idx=0)
        self.language_projection = torch.nn.Sequential(torch.nn.Linear(128,256),torch.nn.GELU(),torch.nn.Linear(256,dim))
        self.step_embedding = torch.nn.Embedding(horizon,dim); self.film = torch.nn.Linear(dim,dim*2)
        self.attention = torch.nn.MultiheadAttention(dim,8,batch_first=True); self.norm = torch.nn.LayerNorm(dim)
        # physical(7), interaction, failure, rank advantage, advantage(m), beats-nominal logit
        self.head = torch.nn.Sequential(torch.nn.Linear(dim,512),torch.nn.GELU(),torch.nn.Linear(512,12))
        torch.nn.init.zeros_(self.head[-1].weight); torch.nn.init.zeros_(self.head[-1].bias)

    def forward(self, slots, actions, joints, language):
        batch, branches = actions.shape[:2]
        delta = (actions - actions[:, :1]) / self.action_delta_std
        gate = delta.norm(dim=-1).tanh()
        slot_values = self.slot_norm(slots)[:,None].expand(-1,branches,-1,-1).reshape(batch*branches,slots.shape[1],slots.shape[2])
        language_mask = language.ne(0)[...,None]
        language_value = (self.language_embedding(language)*language_mask).sum(1)/language_mask.sum(1).clamp_min(1)
        condition = self.action_encoder((actions-self.action_mean)/self.action_std) + self.delta_encoder(delta)
        condition = condition + self.joint_encoder((joints-self.joint_mean)/self.joint_std)[:,None]
        condition = condition + self.language_projection(language_value)[:,None]
        outputs=[]
        for step in range(self.horizon):
            scale,shift=self.film(condition+self.step_embedding.weight[step]).chunk(2,-1)
            value=slot_values*(1+0.1*scale.reshape(-1,1,512).tanh())+shift.reshape(-1,1,512)
            attended,_=self.attention(value,value,value,need_weights=False); raw=self.head(self.norm(value+attended).mean(1)).reshape(batch,branches,12)
            # Nominal is the exact reference: rank and advantage are identically zero.
            raw[...,9]=raw[...,9]*gate; raw[...,10]=raw[...,10]*gate
            outputs.append(raw)
        return torch.stack(outputs,dim=2)


def loss_terms(output, physical, interaction, failure, progress, valid, args):
    mask=valid[...,None].float(); advantage=progress-progress[:,:1]
    physical_loss=(F.smooth_l1_loss(output[...,:7]*100,physical*100,reduction="none")*mask).sum()/(mask.sum()*7).clamp_min(1)
    interaction_loss=(F.binary_cross_entropy_with_logits(output[...,7],interaction,reduction="none")*valid).sum()/valid.sum().clamp_min(1)
    failure_loss=(F.binary_cross_entropy_with_logits(output[...,8],failure,reduction="none")*valid).sum()/valid.sum().clamp_min(1)
    advantage_loss=(F.smooth_l1_loss(output[...,10]*1000,advantage*1000,reduction="none")*valid).sum()/valid.sum().clamp_min(1)
    non_nominal=valid.clone(); non_nominal[:,0]=False; beat=advantage.gt(args.beat_margin_m).float()
    beat_loss=(F.binary_cross_entropy_with_logits(output[...,11],beat,reduction="none")*non_nominal).sum()/non_nominal.sum().clamp_min(1)
    logits=output[...,9].masked_fill(~valid,-1e4); target=(advantage/args.ranking_temperature_m).masked_fill(~valid,-1e4).softmax(1)
    ranking=-(target*logits.log_softmax(1)).sum(1); informative=(advantage.max(1).values-advantage.min(1).values>0.001)&valid.any(1)
    ranking_loss=(ranking*informative).sum()/informative.sum().clamp_min(1)
    total=(args.physical_weight*physical_loss+args.classification_weight*(interaction_loss+failure_loss)
           +args.advantage_weight*advantage_loss+args.beat_nominal_weight*beat_loss+args.ranking_weight*ranking_loss)
    return total,{"physical_cm":physical_loss,"advantage_mm":advantage_loss,"beat_bce":beat_loss,"ranking":ranking_loss}


@torch.inference_mode()
def evaluate(model,data,device,gate_margin_m=0.001):
    model.eval(); n=len(data["before_slots"])
    output=model(data["before_slots"].to(device).float(),data["executed_actions"].to(device).float(),
                 data["joint_positions"].to(device).float(),data["language_tokens"].to(device))
    true=data["task_progress_m"][:,:,-1].to(device).float(); advantage=true-true[:,:1]
    rank=output[:,:,-1,9]; predicted_advantage=output[:,:,-1,10]; beat_score=output[:,:,-1,11].sigmoid()
    best=rank.argmax(1); row=torch.arange(n,device=device); predicted_gain=predicted_advantage[row,best]
    intervene=(predicted_gain>gate_margin_m)&(beat_score[row,best]>0.5); chosen=torch.where(intervene,best,torch.zeros_like(best))
    selected=true.gather(1,chosen[:,None]).squeeze(1); oracle=true.max(1).values; nominal=true[:,0]
    top3=true.gather(1,rank.topk(3,1).indices).max(1).values; combos=torch.combinations(torch.arange(10,device=device),r=3)
    random_top3=true[:,combos].max(-1).values.mean(1)
    non_nominal=torch.ones_like(advantage,dtype=torch.bool); non_nominal[:,0]=False
    result={
      "advantage_mae_mm":float((predicted_advantage[non_nominal]-advantage[non_nominal]).abs().mean()*1000),
      "beat_nominal_auc":binary_auc(beat_score[non_nominal],advantage[non_nominal].gt(0.001)),
      "best_branch_accuracy_percent":float(100*best.eq(true.argmax(1)).float().mean()),
      "top3_regret_mm":float((oracle-top3).mean()*1000),"random_top3_regret_mm":float((oracle-random_top3).mean()*1000),
      "nominal_progress_mm":float(nominal.mean()*1000),"always_scorer_progress_mm":float(true.gather(1,best[:,None]).mean()*1000),
      "gated_progress_mm":float(selected.mean()*1000),"oracle_progress_mm":float(oracle.mean()*1000),
      "gated_intervention_percent":float(100*intervene.float().mean()),"gated_positive_progress_percent":float(100*selected.gt(0).float().mean()),
    }; model.train(); return result


def save(model,directory,step,label=None):
    directory.mkdir(parents=True,exist_ok=True); name=f"advantage-best-{label}.safetensors" if label else f"advantage-step-{step:06d}.safetensors"
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(directory/name))


def main():
    args=arguments(); torch.manual_seed(args.seed); data_dir=Path(args.data_dir); progress_dir=Path(args.progress_dir); output=Path(args.output_dir)
    train=load_split(data_dir,progress_dir,ids(args.train_tasks),ids(args.train_episodes)); val=load_split(data_dir,progress_dir,[args.val_task],ids(args.val_episodes))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=TaskAwareAdvantageScorer(*normalization(train),delta_std(train)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01); best_gate=-float("inf"); best_rank=float("inf")
    print(f"device={device} train_forks={len(train['before_slots'])} val_forks={len(val['before_slots'])} params={sum(p.numel() for p in model.parameters())}",flush=True)
    for step in range(1,args.steps+1):
        idx=torch.randint(len(train["before_slots"]),(args.batch_forks,)); valid=train["after_valid"][idx].to(device).bool(); progress=train["task_progress_m"][idx].to(device).float()
        physical,interaction,failure=targets(train,idx,device)
        result=model(train["before_slots"][idx].to(device).float(),train["executed_actions"][idx].to(device).float(),train["joint_positions"][idx].to(device).float(),train["language_tokens"][idx].to(device))
        loss,terms=loss_terms(result,physical,interaction,failure,progress,valid,args); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if step==1 or step%args.log_every==0: print(f"step={step}/{args.steps} loss={float(loss):.6f} "+" ".join(f"{k}={float(v):.6f}" for k,v in terms.items())+f" finite={finite}",flush=True)
        if step%args.eval_every==0 or step==args.steps:
            metrics=evaluate(model,val,device); print(f"validation step={step} {json.dumps(metrics)}",flush=True)
            if metrics["gated_progress_mm"]>best_gate: best_gate=metrics["gated_progress_mm"]; save(model,output,step,"gated")
            if metrics["top3_regret_mm"]<best_rank: best_rank=metrics["top3_regret_mm"]; save(model,output,step,"ranking")
        if step%args.save_every==0: save(model,output,step)
    output.mkdir(parents=True,exist_ok=True); (output/"config.json").write_text(json.dumps(vars(args),indent=2),encoding="utf-8")
    print(f"saved={output} best_gated_progress_mm={best_gate:.6f} best_top3_regret_mm={best_rank:.6f}")


if __name__=="__main__": main()
