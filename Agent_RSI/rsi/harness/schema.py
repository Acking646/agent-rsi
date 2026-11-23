"""Versioned five-module harness used by every real Agent RSI run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rsi.contracts import HarnessModule


@dataclass(frozen=True)
class HarnessModuleConfig:
    instructions: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HarnessModuleConfig:
        return cls(
            instructions=tuple(str(x) for x in value.get("instructions", ())),
            parameters=dict(value.get("parameters", {})),
        )


@dataclass(frozen=True)
class HarnessSpec:
    version_id: str
    parent_version: str | None
    modules: dict[HarnessModule, HarnessModuleConfig]
    accepted_proposal: str | None = None

    @classmethod
    def baseline(cls) -> HarnessSpec:
        modules = {
            HarnessModule.AGENT_LOOP: HarnessModuleConfig(
                (
                    "Work in a linear observe-think-act loop.",
                    "Inspect the repository before editing and verify the final state.",
                ),
                {"max_steps": 40, "retry_limit": 2},
            ),
            HarnessModule.TOOL_USE: HarnessModuleConfig(
                (
                    "Use one shell command per action and keep commands auditable.",
                    "Never read task tests, reference solutions, or verifier logs.",
                ),
                {"command_timeout_sec": 120, "tool": "shell"},
            ),
            HarnessModule.OBSERVATION_MANAGEMENT: HarnessModuleConfig(
                ("Preserve command errors and the final lines of long output.",),
                {"max_chars": 12000, "head_fraction": 0.35},
            ),
            HarnessModule.CONTEXT_MANAGEMENT: HarnessModuleConfig(
                (
                    "Retain the task, current plan, edits, failures, and verification evidence.",
                ),
                {"max_step_messages": 24, "summary_chars": 6000},
            ),
            HarnessModule.COMPLETION_DETECTION: HarnessModuleConfig(
                (
                    "Declare completion only after a relevant verification command succeeds.",
                    "Return task_complete=true only when no required work remains.",
                ),
                {"required_consecutive": 1, "require_successful_command": True},
            ),
        }
        return cls("h0", None, modules)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HarnessSpec:
        raw_modules = value.get("modules", value)
        modules: dict[HarnessModule, HarnessModuleConfig] = {}
        for module in HarnessModule:
            if module.value not in raw_modules:
                raise ValueError(f"Missing harness module: {module.value}")
            modules[module] = HarnessModuleConfig.from_dict(raw_modules[module.value])
        return cls(
            version_id=str(value.get("version_id", "unversioned")),
            parent_version=value.get("parent_version"),
            modules=modules,
            accepted_proposal=value.get("accepted_proposal"),
        )

    @classmethod
    def load(cls, path: str | Path) -> HarnessSpec:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "parent_version": self.parent_version,
            "accepted_proposal": self.accepted_proposal,
            "modules": {
                module.value: asdict(config) for module, config in self.modules.items()
            },
            "component_hashes": self.component_hashes(),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def component_hashes(self) -> dict[str, str]:
        hashes = {}
        for module, config in self.modules.items():
            payload = json.dumps(asdict(config), sort_keys=True, ensure_ascii=False)
            hashes[module.value] = hashlib.sha256(payload.encode()).hexdigest()
        return hashes

    def prompt(self) -> str:
        sections = []
        for module in HarnessModule:
            config = self.modules[module]
            body = "\n".join(f"- {line}" for line in config.instructions)
            parameters = json.dumps(
                config.parameters, ensure_ascii=False, sort_keys=True
            )
            sections.append(
                f"## {module.value}\n{body}\nRuntime parameters: {parameters}"
            )
        return "\n\n".join(sections)

    def parameter(self, module: HarnessModule, name: str, default: Any) -> Any:
        return self.modules[module].parameters.get(name, default)
