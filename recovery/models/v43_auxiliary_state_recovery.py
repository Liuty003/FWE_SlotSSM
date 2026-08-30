"""Unified V4.3 action policy with state heads used only as auxiliary outputs."""
from __future__ import annotations
import torch
from torch import Tensor,nn
from recovery.models.v4_relational_recovery import V4ModelConfig
from recovery.models.v43_geometry_aux_recovery import V43GeometryAuxRecovery

class V43AuxiliaryStateRecovery(V43GeometryAuxRecovery):
    def __init__(self,config:V4ModelConfig=V4ModelConfig()):
        super().__init__(config,direct_head_version='pooled_chunk_mlp_v3');d=2*config.relation_dim
        self.phase_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,256),nn.GELU(),nn.Linear(256,5))
        self.state_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.GELU(),nn.Linear(128,3))
        self.drawer_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.GELU(),nn.Linear(128,1))
        # The action head sees geometry, but never phase/contact predictions.
        self.unified_action_residual=nn.Sequential(nn.LayerNorm(d+3),nn.Linear(d+3,512),nn.GELU(),nn.Linear(512,config.action_chunk*6))
        self.gripper_head=nn.Sequential(nn.LayerNorm(d+3),nn.Linear(d+3,256),nn.GELU(),nn.Linear(256,config.action_chunk))
        nn.init.zeros_(self.unified_action_residual[-1].weight);nn.init.zeros_(self.unified_action_residual[-1].bias)
    def forward(self,**inputs:Tensor)->dict[str,Tensor]:
        o=super().forward(**inputs);pooled=torch.cat((o['relational_context'].mean(1),o['recovery_action_query'].mean(1)),1);condition=torch.cat((pooled,o['predicted_target_delta']),1)
        residual=self.unified_action_residual(condition).reshape(-1,self.config.action_chunk,6);gripper=self.gripper_head(condition)
        phase_logits=self.phase_head(pooled)
        base=o['recovery_actions'];o.update({'phase_logits':phase_logits,'phase_probability':phase_logits.softmax(1),'state_logits':self.state_head(pooled),'predicted_drawer_qpos':self.drawer_head(pooled).squeeze(1),'unified_action_residual':residual,'gripper_logits':gripper})
        o['recovery_actions']=torch.cat((base[...,:6]+residual,gripper.sigmoid().unsqueeze(-1)),2);return o
