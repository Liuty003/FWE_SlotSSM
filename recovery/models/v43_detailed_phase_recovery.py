"""Auxiliary-conditioned recovery with controller-boundary-preserving phases."""
from __future__ import annotations
from torch import nn
from recovery.models.v4_relational_recovery import V4ModelConfig
from recovery.models.v43_auxiliary_conditioned_recovery import V43AuxiliaryConditionedRecovery

DETAILED_PHASE_NAMES=["release_distractor","release_and_clear_bowl","approach_bowl","descend_bowl","close_bowl","lift_bowl","transfer_bowl","lower_bowl","release_bowl","retreat_from_bowl","approach_drawer","contact_drawer","feedback_push_drawer_closed","reacquire_drawer","recontact_drawer"]

class V43DetailedPhaseRecovery(V43AuxiliaryConditionedRecovery):
    def __init__(self,config:V4ModelConfig=V4ModelConfig()):
        super().__init__(config);d=2*config.relation_dim
        self.phase_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,256),nn.GELU(),nn.Linear(256,len(DETAILED_PHASE_NAMES)))
        self.aux_to_action=nn.Sequential(nn.Linear(len(DETAILED_PHASE_NAMES)+4,128),nn.GELU(),nn.Linear(128,d+3))
        nn.init.zeros_(self.aux_to_action[-1].weight);nn.init.zeros_(self.aux_to_action[-1].bias)
