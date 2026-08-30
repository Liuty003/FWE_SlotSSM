#!/usr/bin/env python3
"""Build the exhaustive ten-family LIBERO-10 recovery index and baseline summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


FAMILIES = (
    ("stove-moka", "turn_on_the_stove_and_put_the_moka_pot_on_it", 281),
    ("mug-microwave", "put_the_yellow_and_white_mug_in_the_microwave_and_close_it", 278),
    ("two-moka-stove", "put_both_moka_pots_on_the_stove", 233),
    ("black-bowl-drawer", "put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", 304),
    ("two-mugs-two-plates", "put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate", 287),
    ("mug-pudding", "put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate", 283),
    ("cream-cheese-butter-basket", "put_both_the_cream_cheese_box_and_the_butter_in_the_basket", 210),
    ("soup-tomato-basket", "put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket", 212),
    ("soup-cream-cheese-basket", "put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket", 194),
    ("book-caddy", "pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy", 237),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-manifest", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def counts(records: list[dict], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[key]) for record in records).items()))


def baseline_artifact(root: Path, task_id: int, episode: int) -> dict:
    episode_root = root / f"task{task_id}-episode{episode}"
    metadata_path = episode_root / "metadata.json"
    steps_path = episode_root / "steps.jsonl"
    video_path = episode_root / "rollout.mp4"
    for path in (metadata_path, steps_path, video_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty baseline artifact: {path}")
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get("complete") is not True or metadata.get("task_suite") != "libero_10"
            or int(metadata.get("task_id", -1)) != task_id
            or int(metadata.get("episode_id", -1)) != episode):
        raise ValueError(f"baseline metadata identity/complete mismatch: {metadata_path}")
    dense_bundle = Path(metadata.get("dense_bundle", ""))
    if not dense_bundle.is_file() or dense_bundle.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty dense bundle: {dense_bundle}")
    return {
        "metadata": str(metadata_path.resolve()),
        "steps": str(steps_path.resolve()),
        "video": str(video_path.resolve()),
        "dense_bundle": str(dense_bundle.resolve()),
        "success": bool(metadata["success"]),
        "transitions": int(metadata["transitions"]),
        "task_language": metadata["task_language"],
        "schema": metadata["schema"],
        "wait_steps": int(metadata["wait_steps"]),
        "open_loop_steps": int(metadata["open_loop_steps"]),
    }


def main() -> None:
    args = arguments()
    for path in (args.global_manifest, args.baseline_root):
        if not path.exists():
            raise FileNotFoundError(path)
    global_manifest = json.loads(args.global_manifest.read_text())
    if global_manifest.get("kind") != "libero_plus_official_global_manifest_v1":
        raise ValueError("unsupported global manifest kind")
    records = [record for record in global_manifest["records"] if record["suite"] == "libero_10"]
    if len(records) != 2519:
        raise RuntimeError(f"expected 2519 LIBERO-10 records, found {len(records)}")

    assignments: dict[str, list[dict]] = {slug: [] for slug, _, _ in FAMILIES}
    for record in records:
        matched = [slug for slug, phrase, _ in FAMILIES if phrase in record["official_name"]]
        if len(matched) != 1:
            raise RuntimeError(
                f"{record['sample_id']} must match exactly one semantic family, matched {matched}"
            )
        assignments[matched[0]].append(record)
    all_ids = [record["sample_id"] for family in assignments.values() for record in family]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != {record["sample_id"] for record in records}:
        raise RuntimeError("semantic families are not a disjoint exhaustive partition")

    args.output_root.mkdir(parents=True, exist_ok=True)
    family_index = []
    suite_successes = 0
    suite_failures = 0
    for slug, phrase, expected_count in FAMILIES:
        family_records = sorted(assignments[slug], key=lambda record: record["task_id"])
        if len(family_records) != expected_count:
            raise RuntimeError(
                f"family {slug} expected {expected_count} records, found {len(family_records)}"
            )
        family_root = args.output_root / "families" / slug
        artifacts = []
        for record in family_records:
            artifacts.append({
                "sample_id": record["sample_id"],
                "task_id": record["task_id"],
                "episode_id": record["episode_id"],
                "perturbation_category": record["perturbation_category"],
                "difficulty_level": record["difficulty_level"],
                **baseline_artifact(
                    args.baseline_root, int(record["task_id"]), int(record["episode_id"]),
                ),
            })
        successful = [artifact for artifact in artifacts if artifact["success"]]
        failed = [artifact for artifact in artifacts if not artifact["success"]]
        suite_successes += len(successful)
        suite_failures += len(failed)
        manifest = {
            "kind": "libero_plus_task_family_manifest_v1",
            "schema_version": 1,
            "family_name": slug,
            "semantic_name_fragment": phrase,
            "global_manifest": str(args.global_manifest.resolve()),
            "global_manifest_sha256": sha256(args.global_manifest),
            "selector": {"suite": "libero_10", "official_name_contains": phrase},
            "num_instances": len(family_records),
            "perturbation_counts": counts(family_records, "perturbation_category"),
            "difficulty_counts": counts(family_records, "difficulty_level"),
            "records": family_records,
        }
        summary = {
            "kind": "libero_10_family_baseline_summary_v1",
            "schema_version": 1,
            "family_name": slug,
            "family_manifest": str((family_root / "family-manifest.json").resolve()),
            "baseline_root": str(args.baseline_root.resolve()),
            "num_instances": len(artifacts),
            "successes": len(successful),
            "failures": len(failed),
            "success_rate": len(successful) / len(artifacts),
            "success_task_ids": [artifact["task_id"] for artifact in successful],
            "failure_task_ids": [artifact["task_id"] for artifact in failed],
            "perturbation": {
                category: {
                    "instances": sum(a["perturbation_category"] == category for a in artifacts),
                    "successes": sum(
                        a["perturbation_category"] == category and a["success"] for a in artifacts
                    ),
                    "failures": sum(
                        a["perturbation_category"] == category and not a["success"] for a in artifacts
                    ),
                }
                for category in sorted({a["perturbation_category"] for a in artifacts})
            },
            "artifacts": artifacts,
        }
        family_root.mkdir(parents=True, exist_ok=True)
        (family_root / "family-manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        (family_root / "baseline-summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        family_index.append({
            "family_name": slug,
            "semantic_name_fragment": phrase,
            "num_instances": len(artifacts),
            "successes": len(successful),
            "failures": len(failed),
            "success_rate": len(successful) / len(artifacts),
            "family_manifest": str((family_root / "family-manifest.json").resolve()),
            "baseline_summary": str((family_root / "baseline-summary.json").resolve()),
        })

    if suite_successes != 1960 or suite_failures != 559:
        raise RuntimeError(
            f"baseline totals differ from frozen audit: {suite_successes} success, {suite_failures} failure"
        )
    index = {
        "kind": "libero_10_semantic_family_index_v1",
        "schema_version": 1,
        "global_manifest": str(args.global_manifest.resolve()),
        "global_manifest_sha256": sha256(args.global_manifest),
        "baseline_root": str(args.baseline_root.resolve()),
        "partition_contract": "exactly one of ten semantic families; task id is not a model input",
        "num_families": len(family_index),
        "num_instances": sum(record["num_instances"] for record in family_index),
        "successes": suite_successes,
        "failures": suite_failures,
        "families": family_index,
    }
    (args.output_root / "family-index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
