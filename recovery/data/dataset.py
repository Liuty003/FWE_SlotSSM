"""JSONL IO for the unified recovery transition format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from recovery.data.schema import RecoveryTransition


def read_jsonl(path: Path, *, check_files: bool = False) -> list[RecoveryTransition]:
    rows: list[RecoveryTransition] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                rows.append(RecoveryTransition.from_dict(value, check_files=check_files))
            except Exception as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def write_jsonl(path: Path, rows: Iterable[RecoveryTransition]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            row.validate(check_files=False)
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
