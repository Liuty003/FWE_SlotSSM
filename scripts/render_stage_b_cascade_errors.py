#!/usr/bin/env python
"""Render RGB/oracle/predicted-mask panels for Stage-B cascade errors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.probe_stage_b_oracle_geometry import predict_masks


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--relation-checkpoint", type=Path, required=True)
    parser.add_argument("--bowl-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-patches"))
    parser.add_argument("--raw-manifest", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-data/manifest.json"))
    parser.add_argument("--language-features", type=Path, default=Path(
        "/data2/lty/naive-e-slotssm-runs/stage-b-spatial-counterfactual-patches/"
        "siglip_language_features.safetensors"))
    return parser.parse_args()


def overlay(rgb, masks, colors):
    result = np.asarray(rgb).astype(np.float32).copy()
    for mask, color in zip(masks, colors):
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            rgb.size, Image.Resampling.NEAREST
        )
        selected = np.asarray(mask_image) > 0
        result[selected] = 0.55 * result[selected] + 0.45 * np.asarray(color)
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


def titled(image, title):
    panel = Image.new("RGB", (image.width, image.height + 24), "white")
    panel.paste(image, (0, 24))
    ImageDraw.Draw(panel).text((5, 5), title, fill="black")
    return panel


def main():
    args = arguments()
    report = json.loads(args.report.read_text())
    errors = report["cascade"]["predicted_bowls__predicted_references"]["test"]["errors"]
    keys = [(int(error["task_id"]), int(error["state_id"])) for error in errors]
    prediction_args = SimpleNamespace(
        data_dir=args.data_dir, language_features=args.language_features,
        relation_checkpoint=args.relation_checkpoint, bowl_checkpoint=args.bowl_checkpoint,
        hidden_dim=256, heads=8, mask_threshold=0.5,
    )
    predictions = predict_masks(prediction_args, keys)
    raw = json.loads(args.raw_manifest.read_text())
    by_key = {(int(row["task_id"]), int(row["state_id"])): row for row in raw["samples"]}
    rows, diagnostics = [], []
    for error, key in zip(errors, keys):
        sample = by_key[key]
        rgb = Image.open(sample["rgb"]).convert("RGB")
        stored = np.load(sample["masks"])
        oracle = overlay(rgb, [stored["target"], stored["counterfactual_target"],
                               stored["reference_0"]],
                         [(255, 50, 50), (50, 100, 255), (50, 220, 80)])
        predicted = predictions[key]
        oracle_candidates = [stored["target"].astype(bool),
                             stored["counterfactual_target"].astype(bool)]
        proposal_masks = [np.asarray(Image.fromarray(mask.numpy().astype(np.uint8)).resize(
            rgb.size, Image.Resampling.NEAREST)).astype(bool)
            for mask in predicted["candidates"]]
        iou_matrix = []
        for proposal in proposal_masks:
            iou_matrix.append([
                float(np.logical_and(proposal, target).sum() /
                      max(np.logical_or(proposal, target).sum(), 1))
                for target in oracle_candidates
            ])
        first, second = proposal_masks
        predicted_reference = np.asarray(Image.fromarray(
            predicted["references"][0, 0].numpy().astype(np.uint8)
        ).resize(rgb.size, Image.Resampling.NEAREST)).astype(bool)
        oracle_reference = stored["reference_0"].astype(bool)
        diagnostics.append({
            **error, "proposal_target_iou_matrix": iou_matrix,
            "proposal_pair_iou": float(np.logical_and(first, second).sum() /
                                       max(np.logical_or(first, second).sum(), 1)),
            "primary_reference_iou": float(
                np.logical_and(predicted_reference, oracle_reference).sum() /
                max(np.logical_or(predicted_reference, oracle_reference).sum(), 1)
            ),
        })
        proposals = overlay(rgb, [mask.numpy() for mask in predicted["candidates"]],
                            [(255, 50, 50), (50, 100, 255)])
        reference = overlay(rgb, [predicted["references"][0, 0].numpy()], [(50, 220, 80)])
        panels = [
            titled(rgb, f"state {key[1]} RGB margin {error['correct_margin']:.2f}"),
            titled(oracle, "oracle: bowl1 red / bowl2 blue / cookie green"),
            titled(proposals, "predicted proposals"),
            titled(reference, "predicted primary reference"),
        ]
        row = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0)); x += panel.width
        rows.append(row)
    canvas = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y)); y += row.height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    diagnostics_path = args.output.with_suffix(".json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "states": [key[1] for key in keys],
                      "size": list(canvas.size), "diagnostics": str(diagnostics_path)}, indent=2))


if __name__ == "__main__":
    main()
