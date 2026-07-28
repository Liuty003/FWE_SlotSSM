#!/usr/bin/env python
"""Unified inference module for the verified Stage-B spatial selector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from safetensors.torch import load_file

from scripts.train_stage_b_bowl_proposals import BowlProposalSegmenter
from scripts.train_stage_b_relation_queries import RelationQuerySegmenter


RELATION_ORDER = ("between", "in", "next_to", "on", "table_center")


def mask_stats(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool().cpu()
    points = mask.nonzero().float()
    if not len(points):
        return torch.zeros(7)
    y, x = points[:, 0] / 63, points[:, 1] / 63
    return torch.tensor([
        x.mean(), y.mean(), x.min(), y.min(), x.max(), y.max(), mask.float().mean()
    ])


def geometry_features(candidate: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    candidate = candidate.bool().cpu()
    candidate_stats = mask_stats(candidate)
    values = list(candidate_stats)
    reference_stats = []
    for reference in references:
        reference = reference.bool().cpu()
        stats = mask_stats(reference)
        reference_stats.append(stats)
        intersection = (candidate & reference).float().mean()
        union = (candidate | reference).float().mean().clamp_min(1e-6)
        values.extend((stats, torch.tensor([
            candidate_stats[0] - stats[0], candidate_stats[1] - stats[1],
            torch.linalg.vector_norm(candidate_stats[:2] - stats[:2]),
            intersection, intersection / union,
        ])))
    midpoint = (reference_stats[0][:2] + reference_stats[1][:2]) / 2
    values.extend((midpoint, torch.tensor([
        torch.linalg.vector_norm(candidate_stats[:2] - midpoint),
        torch.linalg.vector_norm(candidate_stats[:2] - torch.tensor([0.5, 0.5])),
    ])))
    return torch.cat([value.flatten() for value in values])


def component_candidates(masks: torch.Tensor) -> torch.Tensor:
    """Turn the proposal union into its two largest non-overlapping components."""
    masks = masks.bool().cpu()
    labels, count = ndimage.label(masks.any(0).numpy())
    components = sorted(
        ((labels == component, int((labels == component).sum()))
         for component in range(1, count + 1)),
        key=lambda item: item[1], reverse=True,
    )
    if len(components) < 2:
        return masks
    return torch.from_numpy(np.stack((components[0][0], components[1][0])))


class GeometryScorer(nn.Module):
    def __init__(self, dimension: int = 40):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class StageBSpatialSelector(nn.Module):
    """Select a target instance through explicit proposals and mask geometry."""

    def __init__(
        self, bowl_model: BowlProposalSegmenter,
        reference_model: RelationQuerySegmenter, geometry_model: GeometryScorer,
        mask_threshold: float = 0.5, use_components: bool = True,
    ):
        super().__init__()
        self.bowl_model = bowl_model
        self.reference_model = reference_model
        self.geometry_model = geometry_model
        self.mask_threshold = mask_threshold
        self.use_components = use_components

    @classmethod
    def from_config(cls, path: Union[Path, str], device: Union[torch.device, str] = "cpu"):
        path = Path(path)
        config = json.loads(path.read_text())
        if tuple(config["relation_order"]) != RELATION_ORDER:
            raise RuntimeError(f"Unexpected relation order: {config['relation_order']}")
        hidden_dim, heads = int(config["hidden_dim"]), int(config["heads"])
        bowl = BowlProposalSegmenter(hidden_dim, heads)
        reference = RelationQuerySegmenter(
            hidden_dim, heads, int(config["language_dim"])
        )
        geometry = GeometryScorer(int(config.get("geometry_dimension", 40)))
        bowl.load_state_dict(load_file(config["bowl_checkpoint"]), strict=True)
        reference.load_state_dict(load_file(config["reference_checkpoint"]), strict=True)
        geometry.load_state_dict(load_file(config["geometry_checkpoint"]), strict=True)
        selector = cls(
            bowl, reference, geometry,
            float(config.get("mask_threshold", 0.5)),
            bool(config.get("component_proposals", True)),
        ).to(device).eval()
        selector.config = config
        return selector

    @torch.inference_mode()
    def forward(
        self, patches: torch.Tensor, relation_ids: torch.Tensor,
        role_presence: torch.Tensor, instruction_features: torch.Tensor,
        relation_features: torch.Tensor, reference_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = patches.device
        bowl_logits = self.bowl_model(patches)
        reference_logits, _ = self.reference_model(
            patches, relation_ids, role_presence, instruction_features,
            relation_features, reference_features,
        )
        raw_candidates = bowl_logits.sigmoid() > self.mask_threshold
        references = reference_logits[:, 3:5].sigmoid() > self.mask_threshold
        references = references & role_presence[:, 3:5, None, None].bool()
        candidates, feature_rows = [], []
        for batch_index in range(len(patches)):
            sample_candidates = raw_candidates[batch_index].cpu()
            if self.use_components:
                sample_candidates = component_candidates(sample_candidates)
            candidates.append(sample_candidates)
            rows = []
            for candidate in sample_candidates:
                geometry = geometry_features(candidate, references[batch_index].cpu())
                relation = F.one_hot(
                    relation_ids[batch_index].cpu(), len(RELATION_ORDER)
                ).float()
                rows.append(torch.cat((geometry, relation)))
            feature_rows.append(torch.stack(rows))
        candidates = torch.stack(candidates).to(device)
        geometry_features_batch = torch.stack(feature_rows).to(device)
        # Geometry margins are small for a few occluded states; BF16 can turn a
        # valid ordering into an exact tie. Keep this tiny control head in FP32.
        with torch.autocast(device_type=device.type, enabled=False):
            scores = self.geometry_model(geometry_features_batch.float())
        selected_indices = scores.argmax(-1)
        target_masks = candidates[
            torch.arange(len(candidates), device=device), selected_indices
        ]
        margins = scores.gather(1, selected_indices[:, None]).squeeze(1) - scores.gather(
            1, (1 - selected_indices)[:, None]
        ).squeeze(1)
        return {
            "target_masks": target_masks,
            "candidate_masks": candidates,
            "raw_candidate_masks": raw_candidates,
            "reference_masks": references,
            "candidate_scores": scores,
            "selected_indices": selected_indices,
            "selection_margin": margins,
            "geometry_features": geometry_features_batch,
        }
