"""Single-frame, phase-conditioned V4.3 recovery policy."""
from __future__ import annotations
import torch
from torch import Tensor,nn
from recovery.models.v4_relational_recovery import V4ModelConfig
from recovery.models.v43_geometry_aux_recovery import V43GeometryAuxRecovery

class V43MultitaskRecovery(V43GeometryAuxRecovery):
    phase_count=5
    def __init__(self,config:V4ModelConfig=V4ModelConfig()):
        super().__init__(config,direct_head_version='pooled_chunk_mlp_v3');d=2*config.relation_dim
        self.phase_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,256),nn.GELU(),nn.Linear(256,self.phase_count))
        self.state_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.GELU(),nn.Linear(128,3))
        self.drawer_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.GELU(),nn.Linear(128,1))
        condition=d+3+3+1
        self.phase_action_experts=nn.ModuleList([nn.Sequential(nn.LayerNorm(condition),nn.Linear(condition,512),nn.GELU(),nn.Linear(512,config.action_chunk*6)) for _ in range(self.phase_count)])
        self.gripper_head=nn.Sequential(nn.LayerNorm(condition+self.phase_count),nn.Linear(condition+self.phase_count,256),nn.GELU(),nn.Linear(256,config.action_chunk))
        for expert in self.phase_action_experts:nn.init.zeros_(expert[-1].weight);nn.init.zeros_(expert[-1].bias)
    def forward(self,**inputs:Tensor)->dict[str,Tensor]:
        o=super().forward(**inputs);pooled=torch.cat((o['relational_context'].mean(1),o['recovery_action_query'].mean(1)),1)
        phase_logits=self.phase_head(pooled);phase_prob=phase_logits.softmax(1);state_logits=self.state_head(pooled);drawer=self.drawer_head(pooled)
        condition=torch.cat((pooled,o['predicted_target_delta'],state_logits.sigmoid(),drawer),1)
        experts=torch.stack([x(condition).reshape(-1,self.config.action_chunk,6) for x in self.phase_action_experts],1)
        residual=(experts*phase_prob[:,:,None,None]).sum(1);gripper_logits=self.gripper_head(torch.cat((condition,phase_prob),1))
        base=o['recovery_actions'];o.update({'phase_logits':phase_logits,'phase_probability':phase_prob,'state_logits':state_logits,'predicted_drawer_qpos':drawer.squeeze(1),'phase_action_residual':residual,'gripper_logits':gripper_logits})
        o['recovery_actions']=torch.cat((base[...,:6]+residual,gripper_logits.sigmoid().unsqueeze(-1)),2)
        return o
