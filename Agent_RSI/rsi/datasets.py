"""Reproducible task selection for ModularRSI and local Harbor datasets."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SelectedTask:
    name: str
    path: Path
    split: str
    domain: str


def modularrsi_splits(
    dataset_root: str | Path,
    domain: str,
    seed: int = 2026,
) -> dict[str, list[SelectedTask]]:
    root = Path(dataset_root) / domain
    manifest_path = root / "manifest.json"
    tasks_dir = root / "tasks"
    if not manifest_path.is_file() or not tasks_dir.is_dir():
        raise FileNotFoundError(
            f"Expected ModularRSI domain with manifest.json and tasks/: {root}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = [row for row in manifest["tasks"] if row.get("split") == "train"]
    if len(train) != 120:
        raise ValueError(f"Expected 120 ModularRSI train tasks, found {len(train)}")
    rng = random.Random(seed)
    rng.shuffle(train)
    roles = {
        "evolve": train[:80],
        "dev": train[80:100],
        "heldout": train[100:120],
        "pilot": train[:30],
    }
    selected = {}
    for split, rows in roles.items():
        values = []
        for row in rows:
            path = tasks_dir / row["name"]
            if not path.is_dir():
                raise FileNotFoundError(f"Task missing from checkout: {path}")
            values.append(SelectedTask(row["name"], path, split, domain))
        selected[split] = values
    return selected


def local_harbor_tasks(
    tasks_dir: str | Path,
    domain: str,
    split: str = "external",
    names: list[str] | None = None,
) -> list[SelectedTask]:
    root = Path(tasks_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Harbor tasks directory does not exist: {root}")
    allowed = set(names or ())
    tasks = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "task.toml").is_file():
            continue
        if allowed and path.name not in allowed:
            continue
        tasks.append(SelectedTask(path.name, path, split, domain))
    if not tasks:
        raise ValueError(f"No Harbor tasks found under {root}")
    return tasks


def selection_digest(tasks: list[SelectedTask]) -> str:
    value = "\n".join(f"{t.domain}\t{t.split}\t{t.name}" for t in tasks)
    return hashlib.sha256(value.encode()).hexdigest()
