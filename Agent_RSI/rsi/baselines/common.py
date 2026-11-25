"""Lossless conversion between official optimizer candidates and HarnessSpec."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict

from rsi.contracts import HarnessModule
from rsi.harness.schema import HarnessModuleConfig, HarnessSpec


def harness_to_candidate(harness: HarnessSpec) -> dict[str, str]:
    """Encode all five modules as canonical JSON strings for GEPA/AutoSaddler."""
    return {
        module.value: json.dumps(
            asdict(harness.modules[module]), ensure_ascii=False, sort_keys=True
        )
        for module in HarnessModule
    }


def candidate_to_harness(
    candidate: Mapping[str, str],
    *,
    parent_version: str = "h0",
    prefix: str = "baseline",
) -> HarnessSpec:
    expected = {module.value for module in HarnessModule}
    if set(candidate) != expected:
        missing, extra = expected - set(candidate), set(candidate) - expected
        raise ValueError(
            f"Candidate module mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    modules = {}
    baseline = HarnessSpec.baseline()
    for module in HarnessModule:
        text = candidate[module.value].strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            lines = tuple(line.strip(" -\t") for line in text.splitlines() if line.strip(" -\t"))
            if not lines:
                raise ValueError(f"{module.value} is empty")
            modules[module] = HarnessModuleConfig(
                instructions=lines,
                parameters=dict(baseline.modules[module].parameters),
            )
            continue
        if not isinstance(value, dict):
            raise TypeError(f"{module.value} must encode a JSON object or instruction text")
        modules[module] = HarnessModuleConfig.from_dict(value)
    canonical = json.dumps(dict(candidate), ensure_ascii=False, sort_keys=True)
    version = f"{prefix}-{hashlib.sha256(canonical.encode()).hexdigest()[:12]}"
    return HarnessSpec(version, parent_version, modules)
