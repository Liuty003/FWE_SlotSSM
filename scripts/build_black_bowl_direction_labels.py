#!/usr/bin/env python3
"""Build row-aligned privileged bowl-minus-gripper labels for frozen V4 features."""
import argparse,json
from pathlib import Path
import torch
from safetensors.torch import save_file

def labels(feature):
 report=json.loads(feature.with_suffix('.json').read_text());values=[];sources=[]
 stride=int(report.get('recovery_sampling_stride') or report.get('decision_stride',4))
 for sid,rec in enumerate(report.get('source_records',report.get('sources'))):
  source=Path(rec['source']);meta=json.loads((source/'metadata.json').read_text());teacher=Path(meta['source_teacher'])
  raw=[json.loads(x) for x in (teacher/'raw-transitions.jsonl').read_text().splitlines() if x.strip()]
  starts=list(range(0,len(raw)-8+1,stride))
  if len(starts)!=int(rec['rows']):raise ValueError((source,len(starts),rec['rows'],stride))
  for i in starts:
   r=raw[i];p=r['teacher_only']['target_position_before'];e=r['online_observation']['proprio']['robot0_eef_pos'];values.append([p[j]-e[j] for j in range(3)]);sources.append(sid)
 return torch.tensor(values,dtype=torch.float32),torch.tensor(sources,dtype=torch.int64)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--features',type=Path,nargs='+',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();ys=[];ss=[];offset=0
 for p in a.features:
  y,s=labels(p);ys.append(y);ss.append(s+offset);offset+=int(s.max())+1
 y=torch.cat(ys);s=torch.cat(ss);a.output.parent.mkdir(parents=True,exist_ok=True);save_file({'target_delta_world':y,'label_source_index':s},str(a.output));a.output.with_suffix('.json').write_text(json.dumps({'kind':'black_bowl_direction_labels_v1','rows':len(y),'inputs':[str(x) for x in a.features]},indent=2)+'\n');print(json.dumps({'rows':len(y),'mean_norm':float(y.norm(dim=1).mean())}))
if __name__=='__main__':main()
