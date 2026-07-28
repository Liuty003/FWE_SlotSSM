#!/usr/bin/env python
"""Headless LIBERO rollout with failure-aware candidate-vs-nominal gating."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

REPO_ROOT=Path(__file__).resolve().parents[3]; LIBERO_ROOT=REPO_ROOT/"experiments"/"robot"/"libero"
sys.path.insert(0,str(REPO_ROOT)); sys.path.insert(0,str(LIBERO_ROOT))
os.environ.setdefault("LIBERO_CONFIG_PATH",str(LIBERO_ROOT/"baseline_libero_config"))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction,OpenVLAForActionPrediction_SlotAtt
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor,PrismaticProcessor
from scripts.train_slot_stage3_openvla import select_auxiliary_device
from scripts.train_task_aware_action_scorer import language_tokens
from scripts.train_task_aware_advantage_scorer import TaskAwareAdvantageScorer
from scripts.train_target_state_estimator import TargetStateEstimator
from collect_counterfactual_transitions import intervention_actions
from run_openvla_baseline import SUITE_MAX_STEPS,center_crop,extract_privileged_object_state,extract_robot_state,jsonable,write_json


def arguments():
    p=argparse.ArgumentParser(); p.add_argument("--model",default="artifacts/checkpoints/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--slot-checkpoint",required=True); p.add_argument("--advantage-checkpoint",required=True)
    p.add_argument("--target-state-checkpoint",default=None,
        help="Optional read-only target grounding/phase estimator checkpoint")
    p.add_argument("--enable-target-recovery",action="store_true")
    p.add_argument("--recovery-mode",choices=("replan","vector"),default="replan")
    p.add_argument("--recovery-max-steps",type=int,default=80)
    p.add_argument("--lift-threshold",type=float,default=.8)
    p.add_argument("--lift-confirm-steps",type=int,default=3)
    p.add_argument("--target-approach-m",type=float,default=.10)
    p.add_argument("--target-departure-m",type=float,default=.16)
    p.add_argument("--recovery-gain",type=float,default=2.0)
    p.add_argument("--recovery-grasp-steps",type=int,default=10)
    p.add_argument("--recovery-lift-steps",type=int,default=15)
    p.add_argument("--max-recovery-attempts",type=int,default=2)
    p.add_argument("--task-suite",choices=tuple(SUITE_MAX_STEPS),default="libero_spatial"); p.add_argument("--task-id",type=int,default=9)
    p.add_argument("--episode-id",type=int,default=0); p.add_argument("--wait-steps",type=int,default=20); p.add_argument("--max-steps",type=int,default=5)
    p.add_argument("--translation-delta",type=float,default=.05); p.add_argument("--rotation-delta",type=float,default=.05)
    p.add_argument("--gate-margin-m",type=float,default=.005); p.add_argument("--beat-threshold",type=float,default=.8)
    p.add_argument("--intervention-warmup",type=int,default=40,
        help="Run nominal OpenVLA for this many policy steps before allowing corrections")
    p.add_argument("--intervention-cooldown",type=int,default=8,
        help="Minimum number of nominal steps after each accepted correction")
    p.add_argument("--min-direction-ratio",type=float,default=.5,
        help="A correction may not reverse an active axis or reduce it below this fraction")
    p.add_argument("--allow-gripper-intervention",action="store_true",
        help="Disabled by default: a one-step scorer must not change grasp state")
    p.add_argument("--center-crop",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--gpu-memory-gib",type=int,default=7)
    p.add_argument("--device-map",default="balanced",
        help="Match the verified OpenVLA baseline placement by default")
    p.add_argument("--output",type=Path,required=True); p.add_argument("--rollout-dir",type=Path,default=None)
    p.add_argument("--save-slot-trace",action="store_true",
        help="Save per-step Stage-1 slots for on-policy failure retraining")
    return p.parse_args()


def register():
    AutoConfig.register("openvla",OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig,PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig,PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig,OpenVLAForActionPrediction)


class AdvantageOpenVLAPolicy:
    def __init__(self,args):
        # The cached upstream OpenVLA implementation is the stable inference
        # path used by the verified baseline and counterfactual collector.
        # Build the EGL environment before reaching this constructor, then
        # load the model so MuJoCo keeps its rendering context.
        register(); self.args=args; print("[init] processor",flush=True)
        # Use the exact cached upstream processor used by the verified
        # baseline.  Mixing a locally registered processor with the upstream
        # model produced different discrete action tokens from step zero.
        self.processor=AutoProcessor.from_pretrained(args.model,trust_remote_code=True)
        max_memory={index:f"{args.gpu_memory_gib}GiB" for index in range(torch.cuda.device_count())}
        max_memory["cpu"]="64GiB"
        print(f"[init] OpenVLA max_memory={max_memory}",flush=True)
        self.base=AutoModelForVision2Seq.from_pretrained(args.model,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,
            trust_remote_code=True,device_map=args.device_map,max_memory=max_memory).eval()
        self.base.requires_grad_(False); self.vision_device=next(self.base.vision_backbone.parameters()).device
        self.input_device=next(self.base.parameters()).device; self.auxiliary_device=select_auxiliary_device()
        print(f"[init] Slot tokenizer auxiliary={self.auxiliary_device}",flush=True)
        self.slot_model=OpenVLAForActionPrediction_SlotAtt(model=self.base,number_of_slots=16)
        slot_result=self.slot_model.load_state_dict(load_file(args.slot_checkpoint),strict=False); self.slot_model.requires_grad_(False)
        self.slot_model.object_centric_tokenizer.to(self.auxiliary_device,dtype=torch.bfloat16).eval()
        print("[init] Advantage scorer",flush=True)
        self.scorer=TaskAwareAdvantageScorer(torch.zeros(7),torch.ones(7),torch.zeros(7),torch.ones(7),torch.ones(7))
        self.scorer.load_state_dict(load_file(args.advantage_checkpoint),strict=True); self.scorer.to(self.auxiliary_device).eval()
        self.target_state=None
        if args.target_state_checkpoint:
            self.target_state=TargetStateEstimator()
            self.target_state.load_state_dict(load_file(args.target_state_checkpoint),strict=True)
            self.target_state.to(self.auxiliary_device).eval()
        self.control_step=0; self.last_intervention_step=-10**9; self.slot_trace=[]
        self.target_approached=False; self.lift_streak=0; self.lift_confirmed=False
        self.recovery_phase=None; self.recovery_phase_step=0; self.recovery_attempts=0
        print(f"vision={self.vision_device} input={self.input_device} auxiliary={self.auxiliary_device} stage1_missing={len(slot_result.missing_keys)}",flush=True)

    @torch.inference_mode()
    def slots(self,image):
        processed=center_crop(image) if self.args.center_crop else image
        pixels=self.processor.image_processor.apply_transform(Image.fromarray(processed).convert("RGB"))[None]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            patches=self.base.vision_backbone(pixels.to(self.vision_device)).reshape(1,-1,2176).to(self.auxiliary_device)
            slots,_,_=self.slot_model.object_centric_tokenizer(patches,past_slots=None)
        return slots.float()

    @torch.inference_mode()
    def nominal(self,image,instruction,suite):
        processed=center_crop(image) if self.args.center_crop else image
        prompt=f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs=self.processor(prompt,Image.fromarray(processed).convert("RGB")).to(self.input_device,dtype=torch.bfloat16)
        physical=np.asarray(self.base.predict_action(**inputs,unnorm_key=suite,do_sample=False),dtype=np.float32)
        executed=physical.copy(); executed[-1]=np.sign(2*executed[-1]-1)*-1
        return executed

    @torch.inference_mode()
    def act(self,image,instruction,suite,joints):
        nominal=self.nominal(image,instruction,suite)
        variants=intervention_actions(nominal,self.args.translation_delta,self.args.rotation_delta)
        actions=np.stack([value[1] for value in variants]).astype(np.float32)
        slots=self.slots(image)
        if self.args.save_slot_trace:
            self.slot_trace.append(slots[0].to(dtype=torch.bfloat16,device="cpu").contiguous())
        action_tensor=torch.from_numpy(actions)[None].to(self.auxiliary_device)
        joint_tensor=torch.from_numpy(np.asarray(joints,dtype=np.float32))[None].to(self.auxiliary_device)
        language=language_tokens(instruction)[None].to(self.auxiliary_device)
        output=self.scorer(slots,action_tensor,joint_tensor,language)
        target_prediction=None
        if self.target_state is not None:
            target_prediction=self.target_state(slots,joint_tensor,language)[0]
        rank=output[0,:,-1,9]; advantage=output[0,:,-1,10]; beat=output[0,:,-1,11].sigmoid()
        best=int(rank.argmax())
        candidate=actions[best]
        changed=np.flatnonzero(np.abs(candidate-nominal)>1e-6)
        direction_safe=True
        for axis in changed:
            # Preserve OpenVLA's qualitative intent on active continuous axes.
            # In particular, z-minus may attenuate ascent but cannot cancel or
            # reverse it as happened in the original online rollout.
            if axis < 6 and abs(float(nominal[axis]))>1e-3:
                direction_safe &= bool(candidate[axis]*nominal[axis]>0)
                direction_safe &= bool(abs(float(candidate[axis])) >= self.args.min_direction_ratio*abs(float(nominal[axis])))
        gripper_safe=bool(self.args.allow_gripper_intervention or best!=len(variants)-1)
        warm=bool(self.control_step>=self.args.intervention_warmup)
        cooled=bool(self.control_step-self.last_intervention_step>self.args.intervention_cooldown)
        confident=bool(advantage[best]>self.args.gate_margin_m and beat[best]>self.args.beat_threshold)
        intervene=bool(best!=0 and warm and cooled and confident and direction_safe and gripper_safe)
        chosen=best if intervene else 0
        gate_reason="accepted" if intervene else (
            "nominal_best" if best==0 else "warmup" if not warm else "cooldown" if not cooled else
            "low_confidence" if not confident else "direction_guard" if not direction_safe else "gripper_guard")
        if intervene: self.last_intervention_step=self.control_step
        self.control_step+=1
        executed=actions[chosen].copy()
        decision={"nominal_action":nominal.tolist(),"candidate_names":[v[0] for v in variants],"candidate_actions":actions.tolist(),
            "rank_scores":rank.cpu().tolist(),"predicted_advantage_m":advantage.cpu().tolist(),"beat_probability":beat.cpu().tolist(),
            "best_candidate":best,"chosen_candidate":chosen,"intervened":intervene,"gate_reason":gate_reason,
            "direction_safe":direction_safe}
        if target_prediction is not None:
            target_vector=target_prediction[:3].float().cpu().numpy()
            target_distance=float(np.linalg.norm(target_vector))
            lifted_probability=float(target_prediction[6].sigmoid().cpu())
            self.lift_streak=self.lift_streak+1 if lifted_probability>=self.args.lift_threshold else 0
            self.lift_confirmed=self.lift_confirmed or self.lift_streak>=self.args.lift_confirm_steps
            self.target_approached=self.target_approached or target_distance<=self.args.target_approach_m
            if (self.args.enable_target_recovery and self.recovery_phase is None and
                self.target_approached and not self.lift_confirmed and
                target_distance>=self.args.target_departure_m and
                self.recovery_attempts<self.args.max_recovery_attempts):
                self.recovery_phase="replan" if self.args.recovery_mode=="replan" else "return"
                self.recovery_phase_step=0; self.recovery_attempts+=1
            phase_before=self.recovery_phase
            if self.recovery_phase is not None:
                correction=np.clip(self.args.recovery_gain*target_vector,-.4,.4).astype(np.float32)
                if self.recovery_phase=="replan":
                    recovery_instruction=(
                        "find the black bowl on the wooden cabinet again and pick it up; "
                        "do not move to the plate until the bowl is securely lifted"
                    )
                    executed=self.nominal(image,recovery_instruction,suite)
                    self.recovery_phase_step+=1
                    if self.lift_confirmed:
                        self.recovery_phase=None; self.target_approached=False
                    elif self.recovery_phase_step>=self.args.recovery_max_steps:
                        self.recovery_phase="hold"; self.recovery_phase_step=0
                elif self.recovery_phase=="hold":
                    # Fail closed: bounded recovery must never chase a stale
                    # target estimate indefinitely or resume a false place.
                    executed[:6]=0.0; executed[-1]=1.0
                elif self.recovery_phase=="return":
                    # Experimental vector controller retained for ablations;
                    # unlike v1 it is bounded and fails closed.
                    executed[:3]=correction; executed[-1]=-1.0
                    self.recovery_phase_step+=1
                    if target_distance<=self.args.target_approach_m:
                        self.recovery_phase="grasp"; self.recovery_phase_step=0
                    elif self.recovery_phase_step>=self.args.recovery_max_steps:
                        self.recovery_phase="hold"; self.recovery_phase_step=0
                elif self.recovery_phase=="grasp":
                    executed[:3]=np.clip(correction,-.12,.12); executed[-1]=1.0
                    self.recovery_phase_step+=1
                    if self.recovery_phase_step>=self.args.recovery_grasp_steps:
                        self.recovery_phase="lift"; self.recovery_phase_step=0
                elif self.recovery_phase=="lift":
                    executed[:2]=np.clip(correction[:2],-.12,.12); executed[2]=.30; executed[-1]=1.0
                    self.recovery_phase_step+=1
                    if self.lift_confirmed:
                        self.recovery_phase=None; self.target_approached=False
                    elif self.recovery_phase_step>=self.args.recovery_lift_steps:
                        if self.recovery_attempts>=self.args.max_recovery_attempts:
                            self.recovery_phase="hold"; self.recovery_phase_step=0
                        else:
                            self.recovery_phase="return"; self.recovery_phase_step=0
                            self.recovery_attempts+=1
            decision.update({
                "predicted_target_minus_eef_m":target_prediction[:3].cpu().tolist(),
                "predicted_goal_minus_target_m":target_prediction[3:6].cpu().tolist(),
                "predicted_target_distance_m":target_distance,
                "predicted_lifted_probability":lifted_probability,
                "lift_streak":self.lift_streak,"lift_confirmed":self.lift_confirmed,
                "target_approached":self.target_approached,"recovery_phase":phase_before,
                "recovery_phase_next":self.recovery_phase,"recovery_attempts":self.recovery_attempts,
            })
        return executed,decision


def main():
    args=arguments(); from libero.libero import benchmark,get_libero_path; from libero.libero.envs import OffScreenRenderEnv
    suite=benchmark.get_benchmark_dict()[args.task_suite](); task=suite.get_task(args.task_id); initial=suite.get_task_init_states(args.task_id)
    bddl=Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256); env.seed(0); env.reset(); obs=env.set_init_state(initial[args.episode_id])
    rollout=args.rollout_dir or args.output.with_suffix(""); rollout.mkdir(parents=True,exist_ok=True); steps_path=rollout/"steps.jsonl"; steps_path.write_text("",encoding="utf-8")
    metadata_path=rollout/"metadata.json"; metadata={"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),
      "policy":"openvla-advantage-gated","task_suite":args.task_suite,"task_id":args.task_id,"episode_id":args.episode_id,
      "task_language":task.language,"translation_delta":args.translation_delta,"rotation_delta":args.rotation_delta,
      "gate_margin_m":args.gate_margin_m,"beat_threshold":args.beat_threshold,
      "intervention_warmup":args.intervention_warmup,"intervention_cooldown":args.intervention_cooldown,
      "min_direction_ratio":args.min_direction_ratio,"allow_gripper_intervention":args.allow_gripper_intervention,
      "completed":False,"success":False}
    write_json(metadata_path,metadata); policy=AdvantageOpenVLAPolicy(args); frames=[]; noop=np.asarray([0,0,0,0,0,0,-1],np.float32)
    for _ in range(args.wait_steps): obs,_,done,_=env.step(noop)
    success=False; interventions=0
    try:
        for step in range(args.max_steps):
            image=np.ascontiguousarray(obs["agentview_image"][::-1,::-1]); frames.append(image); before=obs
            action,decision=policy.act(image,task.language,args.task_suite,np.asarray(obs["robot0_joint_pos"],np.float32))
            obs,reward,done,info=env.step(action.tolist()); success=bool(env.check_success()); interventions+=int(decision["intervened"])
            record={"timestep":step,"executed_action":action.tolist(),**decision,"robot_state_before":extract_robot_state(before),
              "robot_state_after":extract_robot_state(obs),"privileged_object_state_before":extract_privileged_object_state(before),
              "privileged_object_state_after":extract_privileged_object_state(obs),"reward":float(reward),"done":bool(done),"success":success,
              "info":{str(k):jsonable(v) for k,v in dict(info).items()}}
            with steps_path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(record,ensure_ascii=False)+"\n")
            if success: break
    finally: env.close()
    if args.save_slot_trace and policy.slot_trace:
        save_file({"slots":torch.stack(policy.slot_trace).contiguous()},str(rollout/"slots.safetensors"))
    args.output.parent.mkdir(parents=True,exist_ok=True); imageio.mimsave(args.output,frames,fps=20)
    metadata.update({"completed":True,"success":success,"num_steps":len(frames),"num_interventions":interventions,"video":str(args.output.resolve()),"steps_jsonl":str(steps_path.resolve()),
      "slot_trace":str((rollout/"slots.safetensors").resolve()) if args.save_slot_trace else None})
    write_json(metadata_path,metadata); print(f"task={args.task_id} episode={args.episode_id} steps={len(frames)} interventions={interventions} success={success}")
    print(f"rollout={rollout.resolve()}")


if __name__=="__main__": main()
