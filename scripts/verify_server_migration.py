#!/usr/bin/env python3
"""Verify the non-reconstructible assets required after a server migration."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


EXPECTED_GIT_COMMITS = {
    "openvla-oft": "e4287e94541f459edc4feabc4e181f537cd569a8",
    "LIBERO-plus": "4976dc30028e805ff8094b55501d532c48fec182",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data/lty/code/naive-e-slotssm"))
    parser.add_argument("--oft-root", type=Path, default=Path("/data/lty/code/openvla-oft-official"))
    parser.add_argument("--libero-root", type=Path, default=Path("/data/lty/code/LIBERO-plus"))
    parser.add_argument("--models-root", type=Path, default=Path("/data2/lty/models"))
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/data2/lty/naive-e-slotssm-runs"),
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Only check existence and git commits; useful for a quick first pass.",
    )
    args = parser.parse_args()

    model = args.models_root / "openvla-7b-oft-finetuned-libero-plus-mixdata"
    global_root = args.runs_root / "libero-plus-global-v1"
    libero_results = args.runs_root / "libero-plus"
    v4_root = args.runs_root / "v4-relational-recovery-v1"

    required_paths = [
        args.repo_root / "scripts/run_v44_dual_batch.py",
        args.repo_root / "docs/reference/v4/ARTIFACT_INDEX.md",
        args.oft_root / "pyproject.toml",
        args.libero_root / "baseline_libero_config/config.yaml",
        args.libero_root / "libero/libero/assets",
        model / "config.json",
        model / "dataset_statistics.json",
        model / "action_head--150000_checkpoint.pt",
        model / "proprio_projector--150000_checkpoint.pt",
        model / "model.safetensors.index.json",
        model / "model-00001-of-00004.safetensors",
        model / "model-00002-of-00004.safetensors",
        model / "model-00003-of-00004.safetensors",
        model / "model-00004-of-00004.safetensors",
        global_root / "official-manifest.json",
        global_root / "baseline-inventory.json",
        global_root / "full-audit-v1",
        global_root / "libero-10-recovery-v1",
        global_root / "recovery-families",
        args.runs_root / "oft-plus-robot-global-v1/forks",
        libero_results / "assets/extracted",
        libero_results / "oft-plus-robot-levels1-5-v1",
        libero_results / "oft-plus-robot-levels1-4-v1",
        libero_results / "oft-plus-robot-level5-v1",
        libero_results / "mug-pudding-checkpoint-v1",
        v4_root / "plus-robot-book/book34-overfit-selector-v1/report.json",
        v4_root
        / "plus-robot-book/v43-book34-expanded-r2/"
        "direct-v3-gw4-lr1e4-seed20260818/best.safetensors",
        v4_root
        / "mug-pudding/counterfactual-factory-unified-v1/training/"
        "v43-relation3d-insample-dagger-action-v3/best.safetensors",
        v4_root
        / "black-bowl-bottom-drawer/detailed-v1/model/best.safetensors",
        v4_root
        / "black-bowl-bottom-drawer/evaluation/grasp-only-all44-final-v1",
    ]

    expected_hashes = {
        global_root / "official-manifest.json": "07cf2af20bf1a7132376ef181f5a7be91fd71437008a4afa1baaa03d4fd605fd",
        global_root / "baseline-inventory.json": "857dd111a58e8ed084444ccd846c5a1e766a84f2461ef0f57b7e694bbb022e82",
        global_root
        / "libero-10-recovery-v1/model-v3/native-slot-h1-action/best.safetensors": "6cac2553419f87bfeeb88becb298411b36f6841a74f2dc03702c21e822e6bede",
        global_root
        / "libero-10-recovery-v1/model-v3/native-slot-h1-gate/best.safetensors": "9751d0a0d4fb5e88600476a854719c0781d131f5f143ab1db9263be93a2b29c7",
        global_root
        / "libero-10-recovery-v1/model-v3/family-gate-deployment-v1.json": "0ec095ce16193c9e321f5943994cf3e4dc0f3aa8f1e5802f3f8b17ccee2b2af2",
        v4_root
        / "plus-robot-book/v43-book34-expanded-r2/"
        "direct-v3-gw4-lr1e4-seed20260818/best.safetensors": "a5acdcdabdcf7b429174cbed4e880ace5e487d7a0523f36bd5a9966450e4e693",
        v4_root
        / "plus-robot-book/book34-overfit-selector-v1/report.json": "82c5a27d4ebfa5af078bc7c3b83a7ab50ee6a56cf1de297cf32f3d5ba72410c0",
        v4_root
        / "mug-pudding/counterfactual-factory-unified-v1/training/"
        "v43-relation3d-insample-dagger-action-v3/best.safetensors": "de8026889700cbefc396550a40cc0540eafc4032b41ea66cf03bf301e5f05834",
        v4_root
        / "black-bowl-bottom-drawer/detailed-v1/model/best.safetensors": "1e370c7dc073ee5de6db28c87692616de2879cc3aacb6a0da55fa3247ee77172",
    }

    failures: list[str] = []
    for path in required_paths:
        if path.exists():
            print(f"OK path {path}")
        else:
            failures.append(f"missing path: {path}")

    for label, root in (("openvla-oft", args.oft_root), ("LIBERO-plus", args.libero_root)):
        observed = git_head(root)
        expected = EXPECTED_GIT_COMMITS[label]
        if observed == expected:
            print(f"OK git  {label} {observed}")
        else:
            failures.append(f"wrong git commit for {label}: expected {expected}, found {observed}")

    if not args.skip_hashes:
        for path, expected in expected_hashes.items():
            if not path.is_file():
                failures.append(f"cannot hash missing file: {path}")
                continue
            observed = sha256(path)
            if observed == expected:
                print(f"OK hash {path}")
            else:
                failures.append(f"hash mismatch: {path}; expected {expected}, found {observed}")

    if failures:
        print("\nMIGRATION CHECK FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nMIGRATION CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
