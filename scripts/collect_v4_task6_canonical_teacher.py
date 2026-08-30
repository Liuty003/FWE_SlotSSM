#!/usr/bin/env python3
"""Re-record an audited task-6 canonical recovery teacher as V4 RGB-D data.

This is intentionally separate from the V3 collector: it preserves its exact
prefix/recovery/OFT-completion replay and admission test, while recording only
policy-visible RGB-D/calibration plus training-only instance masks.
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_v3_task6_canonical_teacher import read_jsonl, proprio
from scripts.collect_v4_oft_rgbd_rollouts import calibration, depth_and_segmentation

def main():
    p=argparse.ArgumentParser(); p.add_argument('--canonical-metadata',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0); p.add_argument('--wait-steps',type=int,default=20); p.add_argument('--max-completion-steps',type=int,default=280); p.add_argument('--verify-fork-only',action='store_true')
    p.add_argument('--oft-root',type=Path,default=Path('/data/lty/code/openvla-oft-official')); p.add_argument('--libero-root',type=Path,default=ROOT/'experiments'/'robot'/'libero'); p.add_argument('--checkpoint',type=Path,default=Path('/data2/lty/models/openvla-7b-oft-finetuned-libero-10')); p.add_argument('--skip-checkpoint-config-sync',action='store_true')
    a=p.parse_args(); meta=json.loads(a.canonical_metadata.read_text()); source=Path(meta['source_rollout']); branch=Path(meta['recovery_branch'])
    if meta.get('success') is not True or meta.get('task_id') != 6: raise ValueError('requires an admitted task-6 canonical teacher')
    for x in (source/'steps.jsonl',branch,a.oft_root,a.libero_root/'libero'/'libero',a.checkpoint):
        if not x.exists(): raise FileNotFoundError(x)
    if not os.environ.get('CUDA_VISIBLE_DEVICES'): os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    os.environ.setdefault('MUJOCO_GL','egl'); os.environ.setdefault('PYOPENGL_PLATFORM','egl'); os.environ['LIBERO_CONFIG_PATH']=str(a.libero_root/'baseline_libero_config'); sys.path[:0]=[str(a.libero_root),str(a.oft_root)]; os.chdir(a.oft_root)
    import imageio.v2 as imageio, numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    import experiments.robot.openvla_utils as ou
    from experiments.robot.libero.libero_utils import get_libero_dummy_action
    from experiments.robot.libero.run_libero_eval import GenerateConfig,get_image_resize_size,initialize_model,prepare_observation,process_action,set_seed_everywhere,validate_config
    from experiments.robot.robot_utils import get_action
    source_rows,branch_rows=read_jsonl(source/'steps.jsonl'),read_jsonl(branch); first=branch_rows[0]; fork=int(first['failure_timestep']); episode=int(first['source_episode_id'])
    actions=[]
    for row in branch_rows:
        v=row.get('online_observation',{}).get('recovery_action')
        if not isinstance(v,list) or len(v)!=7: raise ValueError('branch lacks recovery actions')
        actions.append(v)
    if a.skip_checkpoint_config_sync: ou.update_auto_map=lambda _:None
    cfg=GenerateConfig(pretrained_checkpoint=str(a.checkpoint),task_suite_name='libero_10',num_trials_per_task=1,num_open_loop_steps=8,use_wandb=False); validate_config(cfg); set_seed_everywhere(cfg.seed)
    suite=benchmark.get_benchmark_dict()['libero_10'](); task=suite.get_task(6); states=suite.get_task_init_states(6); bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    # Match the admitted V3 teacher's LIBERO checkout and seed before replaying
    # any action.  A different checkout changes fixed init states substantially.
    env=SegmentationRenderEnv(bddl_file_name=str(bddl),camera_heights=cfg.env_img_res,camera_widths=cfg.env_img_res,camera_depths=True,camera_segmentations='instance'); env.seed(0)
    dirs={k:a.output_dir/k for k in ('rgb','wrist_rgb','depth','wrist_depth','instance')}; [d.mkdir(parents=True,exist_ok=True) for d in dirs.values()]; records=[]; video=[]
    def record(obs,act,phase):
        i=len(records); stem=f'step-{i:04d}'; rgb=obs['agentview_image'][::-1,::-1]; wrist=obs['robot0_eye_in_hand_image'][::-1,::-1]; dep,seg=depth_and_segmentation(obs,'agentview',np); wdep,_=depth_and_segmentation(obs,'robot0_eye_in_hand',np)
        paths={}; imageio.imwrite(dirs['rgb']/f'{stem}.png',rgb); imageio.imwrite(dirs['wrist_rgb']/f'{stem}.png',wrist); np.savez_compressed(dirs['depth']/f'{stem}.npz',depth=dep); np.savez_compressed(dirs['wrist_depth']/f'{stem}.npz',depth=wdep); ip=None
        if seg is not None: ip=dirs['instance']/f'{stem}.npz'; np.savez_compressed(ip,instance=seg.astype(np.int32))
        records.append({'kind':'v4_task6_canonical_transition','timestep':i,'phase':phase,'gate_label':'RECOVERY' if phase=='TEACHER_RECOVERY' else 'NORMAL','online_observation':{'rgb':str((dirs['rgb']/f'{stem}.png').resolve()),'wrist_rgb':str((dirs['wrist_rgb']/f'{stem}.png').resolve()),'depth':str((dirs['depth']/f'{stem}.npz').resolve()),'wrist_depth':str((dirs['wrist_depth']/f'{stem}.npz').resolve()),'proprio':proprio(obs,np)},'training_only':{'instance_segmentation':None if ip is None else str(ip.resolve())},'executed_action':np.asarray(act,np.float32).tolist()}); video.append(rgb)
    try:
        obs=env.reset(); obs=env.set_init_state(states[episode]); cal=calibration(env,'agentview',cfg.env_img_res,cfg.env_img_res,np)
        for _ in range(a.wait_steps): obs,_,_,_=env.step(get_libero_dummy_action(cfg.model_family))
        for row in source_rows[:fork]: obs,_,_,_=env.step(row['executed_action'])
        err=float(np.linalg.norm(np.asarray(obs['robot0_eef_pos'])-np.asarray(first['online_observation']['proprio']['robot0_eef_pos'])))
        if err>0.03: raise RuntimeError(f'fork mismatch {err}')
        if a.verify_fork_only:
            print(json.dumps({'task_id':6,'episode_id':episode,'fork_timestep':fork,'fork_eef_error_m':err,'verified':True}),flush=True); return
        model,head,pp,npj,processor=initialize_model(cfg); resize=get_image_resize_size(cfg)
        done=False
        for act in actions:
            record(obs,act,'TEACHER_RECOVERY'); obs,_,done,_=env.step(act)
            if done: break
        q=deque(); success=bool(env.check_success())
        for _ in range(a.max_completion_steps):
            if success or done: break
            if not q: po,_=prepare_observation(obs,resize); q.extend(get_action(cfg,model,po,task.language,processor,head,pp,npj,cfg.use_film))
            act=process_action(np.asarray(q.popleft(),np.float32),cfg.model_family); record(obs,act,'OFT_COMPLETION'); obs,_,done,_=env.step(act); success=bool(env.check_success())
    finally: env.close()
    for i in range(0,len(records)-7,8):
        if all(r['phase']=='TEACHER_RECOVERY' for r in records[i:i+8]): records[i]['recovery_target_action_chunk']=[r['executed_action'] for r in records[i:i+8]]
    out={'kind':'v4_task6_canonical_teacher','schema':'v4_rgbd_rollout_v2','task_suite':'libero_10','task_id':6,'episode_id':episode,'task_language':task.language,'success':bool(success),'transitions':len(records),'fork_timestep':fork,'fork_eef_error_m':err,'geometry_view':'agentview','agentview_calibration':cal,'source_rollout':str(source),'recovery_branch':str(branch),'libero_root':str(a.libero_root.resolve()),'oft_root':str(a.oft_root.resolve()),'oft_checkpoint':str(a.checkpoint.resolve()),'env_seed':0,'wait_steps':a.wait_steps,'open_loop_steps':cfg.num_open_loop_steps,'rollout_id':f'task6:libero_10:6:{episode}:fork{fork}','teacher_only_branch_selection':True,'online_inputs_permitted':['main_rgb','wrist_rgb','main_depth','wrist_depth','camera_calibration','language','proprio'],'training_only_inputs':['instance_segmentation']}
    (a.output_dir/'metadata.json').write_text(json.dumps(out,indent=2)+'\n'); (a.output_dir/'steps.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records));
    if success: (a.output_dir/'transitions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    imageio.mimsave(a.output_dir/'rollout.mp4',video,fps=30); print(json.dumps(out),flush=True)
if __name__=='__main__': main()
