"""Cross-task candidate future predictor in canonical role-geometry space."""
from __future__ import annotations
import torch
from torch import nn

class CanonicalFutureDynamics(nn.Module):
    """Predict a candidate's observable future without post-action inputs.

    Geometry is canonical `(target, context, distractor)` image geometry
    `[3,3]`; attach/lift are teacher-supervised training targets only.
    """
    def __init__(self, state_dim: int = 128, action_dim: int = 28):
        super().__init__()
        self.action = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim,96),nn.GELU())
        self.task = nn.Embedding(2,8)
        self.trunk = nn.Sequential(nn.LayerNorm(state_dim+96+8),nn.Linear(state_dim+104,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        self.geometry_delta = nn.Linear(128,9)
        self.attached = nn.Linear(128,1)
        self.lifted = nn.Linear(128,1)
        self.value = nn.Linear(128,1)
    def forward(self,state,action_summary,task_index,current_geometry):
        if current_geometry.shape != (len(state),3,3): raise ValueError('current_geometry must be [B,3,3]')
        h=self.trunk(torch.cat((state,self.action(action_summary),self.task(task_index)),1))
        return {'future_geometry':current_geometry+self.geometry_delta(h).view(-1,3,3), 'attached_logits':self.attached(h).squeeze(1), 'lifted_logits':self.lifted(h).squeeze(1), 'candidate_value':self.value(h).squeeze(1)}
