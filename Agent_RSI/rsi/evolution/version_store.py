"""Persistent harness lineage with explicit validation and rollback records."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.harness.schema import HarnessSpec


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    parent_score: float
    candidate_score: float
    regression_rate: float
    reason: str
    same_batch_score: float | None = None
    heldout_score: float | None = None


class VersionStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions = self.root / "versions"
        self.events_path = self.root / "events.jsonl"
        self.active_path = self.root / "active.json"
        self.versions.mkdir(parents=True, exist_ok=True)

    def initialize(self, baseline: HarnessSpec | None = None) -> HarnessSpec:
        baseline = baseline or HarnessSpec.baseline()
        if not self.active_path.exists():
            self._write_version(baseline)
            self._set_active(baseline.version_id)
            self._event("initialize", {"version_id": baseline.version_id})
        return self.active()

    def active(self) -> HarnessSpec:
        if not self.active_path.exists():
            raise FileNotFoundError("Version store is not initialized")
        value = json.loads(self.active_path.read_text(encoding="utf-8"))
        return self.load(value["version_id"])

    def load(self, version_id: str) -> HarnessSpec:
        return HarnessSpec.load(self.versions / f"{version_id}.json")

    def decide(
        self,
        parent: HarnessSpec,
        candidate: HarnessSpec,
        decision: GateDecision,
        metadata: dict[str, Any] | None = None,
    ) -> HarnessSpec:
        self._write_version(candidate)
        payload = {
            "parent_version": parent.version_id,
            "candidate_version": candidate.version_id,
            "decision": asdict(decision),
            "metadata": metadata or {},
        }
        if decision.accepted:
            self._set_active(candidate.version_id)
            self._event("accept", payload)
            return candidate
        self._event("reject", payload)
        return parent

    def rollback(self, version_id: str, reason: str) -> HarnessSpec:
        target = self.load(version_id)
        previous = self.active().version_id
        self._set_active(version_id)
        self._event(
            "rollback",
            {"from_version": previous, "to_version": version_id, "reason": reason},
        )
        return target

    def _write_version(self, harness: HarnessSpec) -> None:
        target = self.versions / f"{harness.version_id}.json"
        if target.exists():
            existing = HarnessSpec.load(target)
            if existing.to_dict() != harness.to_dict():
                raise ValueError(f"Version id collision: {harness.version_id}")
            return
        self._atomic_json(target, harness.to_dict())

    def _set_active(self, version_id: str) -> None:
        self._atomic_json(self.active_path, {"version_id": version_id})
        self._atomic_json(
            self.root / "active-harness.json", self.load(version_id).to_dict()
        )

    def _event(self, kind: str, payload: dict[str, Any]) -> None:
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": kind,
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)
