#!/usr/bin/env python3
"""Split the exhaustive LIBERO-10 classification into frozen semantic families."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-index", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--onsets", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def counter(records: list[dict], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get(key)) for record in records).items()))


def main() -> None:
    args = arguments()
    index = json.loads(args.family_index.read_text())
    classification = json.loads(args.classification.read_text())
    if index.get("kind") != "libero_10_semantic_family_index_v1":
        raise ValueError("unsupported family index")
    if classification.get("kind") != "libero_plus_automatic_failure_manifest_v1":
        raise ValueError("unsupported failure classification")
    classified = {record["sample_id"]: record for record in classification["records"]}
    if len(classified) != 2519:
        raise RuntimeError(f"expected 2519 unique classified records, found {len(classified)}")

    onset_payload = None if args.onsets is None else json.loads(args.onsets.read_text())
    onsets = {} if onset_payload is None else {
        record["sample_id"]: record for record in onset_payload["records"]
    }
    if onset_payload is not None and len(onsets) != 559:
        raise RuntimeError(f"expected 559 unique onset records, found {len(onsets)}")

    assigned: set[str] = set()
    summaries = []
    onset_summaries = []
    onset_review = []
    review_candidates = []
    for family in index["families"]:
        manifest_path = Path(family["family_manifest"])
        manifest = json.loads(manifest_path.read_text())
        sample_ids = [record["sample_id"] for record in manifest["records"]]
        overlap = assigned.intersection(sample_ids)
        if overlap:
            raise RuntimeError(f"duplicate family assignments: {sorted(overlap)[:3]}")
        assigned.update(sample_ids)
        missing = [sample_id for sample_id in sample_ids if sample_id not in classified]
        if missing:
            raise RuntimeError(f"family {family['family_name']} lacks classifications: {missing[:3]}")
        records = [classified[sample_id] for sample_id in sample_ids]
        failures = [record for record in records if record["failure_type"] != "success"]
        payload = {
            "kind": "libero_10_family_failure_classification_v1",
            "schema_version": 1,
            "family_name": family["family_name"],
            "family_manifest": str(manifest_path.resolve()),
            "source_classification": str(args.classification.resolve()),
            "rollouts": len(records),
            "failures": len(failures),
            "failure_type_counts": counter(failures, "failure_type"),
            "confidence_counts": counter(failures, "confidence"),
            "perturbation_counts": counter(failures, "perturbation_category"),
            "records": records,
        }
        output = args.output_root / "families" / family["family_name"] / "failure-classification-v1.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        summaries.append({key: payload[key] for key in (
            "family_name", "rollouts", "failures", "failure_type_counts",
            "confidence_counts", "perturbation_counts",
        )})

        if onset_payload is not None:
            family_onsets = [onsets[sample_id] for sample_id in sample_ids if sample_id in onsets]
            family_onset_payload = {
                "kind": "libero_10_family_failure_onset_manifest_v2",
                "schema_version": 2,
                "family_name": family["family_name"],
                "family_manifest": str(manifest_path.resolve()),
                "source_onsets": str(args.onsets.resolve()),
                "num_records": len(family_onsets),
                "status_counts": counter(family_onsets, "onset_status"),
                "failure_type_counts": counter(family_onsets, "failure_type"),
                "records": family_onsets,
            }
            (output.parent / "failure-onsets-v2.json").write_text(
                json.dumps(family_onset_payload, indent=2, ensure_ascii=False) + "\n"
            )
            onset_summaries.append({key: family_onset_payload[key] for key in (
                "family_name", "num_records", "status_counts", "failure_type_counts",
            )})
            onset_review.extend({
                **record, "family_name": family["family_name"],
                "review_reason": f"onset_{record['onset_status']}",
            } for record in family_onsets if record["onset_status"] != "located")

        # Keep human review sparse and deterministic: all ambiguous failures,
        # plus one representative per automatic family/type cluster.
        selected_ids = set()
        for record in failures:
            if record["confidence"] in {"low", "none"}:
                selected_ids.add(record["sample_id"])
                review_candidates.append({
                    **record, "family_name": family["family_name"],
                    "review_reason": "low_or_no_confidence",
                    "video": str(Path(record["metadata"]).with_name("rollout.mp4")),
                })
        for failure_type in sorted({record["failure_type"] for record in failures}):
            representative = next(
                record for record in failures if record["failure_type"] == failure_type
            )
            if representative["sample_id"] not in selected_ids:
                review_candidates.append({
                    **representative, "family_name": family["family_name"],
                    "review_reason": "one_per_family_failure_type",
                    "video": str(Path(representative["metadata"]).with_name("rollout.mp4")),
                })

    if assigned != set(classified):
        raise RuntimeError(f"family split does not cover classification: {len(set(classified) - assigned)} missing")
    summary = {
        "kind": "libero_10_family_failure_summary_v1",
        "schema_version": 1,
        "family_index": str(args.family_index.resolve()),
        "source_classification": str(args.classification.resolve()),
        "rollouts": sum(record["rollouts"] for record in summaries),
        "failures": sum(record["failures"] for record in summaries),
        "families": summaries,
    }
    review = {
        "kind": "libero_10_sparse_failure_review_manifest_v1",
        "schema_version": 1,
        "selection": "all low/none confidence plus one representative per family/failure-type cluster",
        "num_records": len(review_candidates),
        "records": review_candidates,
    }
    (args.output_root / "failure-summary-v1.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_root / "manual-review-v1.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n"
    )
    if onset_payload is not None:
        onset_summary_payload = {
            "kind": "libero_10_family_failure_onset_summary_v2",
            "schema_version": 2,
            "source_onsets": str(args.onsets.resolve()),
            "num_records": sum(record["num_records"] for record in onset_summaries),
            "families": onset_summaries,
        }
        onset_review_payload = {
            "kind": "libero_10_sparse_onset_review_manifest_v2",
            "schema_version": 2,
            "selection": "all non-located onsets",
            "num_records": len(onset_review),
            "records": onset_review,
        }
        (args.output_root / "onset-summary-v2.json").write_text(
            json.dumps(onset_summary_payload, indent=2, ensure_ascii=False) + "\n"
        )
        (args.output_root / "onset-review-v2.json").write_text(
            json.dumps(onset_review_payload, indent=2, ensure_ascii=False) + "\n"
        )
    print(json.dumps({
        "rollouts": summary["rollouts"], "failures": summary["failures"],
        "manual_review_records": review["num_records"],
        "onset_review_records": len(onset_review), "families": summaries,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
