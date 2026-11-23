"""Deterministic compiler for the preregistered harness update space."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from rsi.contracts import HarnessModule, UpdateOperator
from rsi.harness.schema import HarnessModuleConfig, HarnessSpec


@dataclass(frozen=True)
class StructuredUpdate:
    proposal_id: str
    target_module: HarnessModule
    operator: UpdateOperator
    scope: str
    rationale: str
    arguments: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StructuredUpdate:
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=False)
        proposal_id = str(
            value.get("proposal_id")
            or hashlib.sha256(canonical.encode()).hexdigest()[:16]
        )
        return cls(
            proposal_id=proposal_id,
            target_module=HarnessModule(value["target_module"]),
            operator=UpdateOperator(value["operator"]),
            scope=str(value.get("scope", "global")),
            rationale=str(value.get("rationale", "")),
            arguments=dict(value.get("arguments", {})),
        )


@dataclass(frozen=True)
class CompileResult:
    parent_version: str
    child: HarnessSpec
    proposal: StructuredUpdate
    diff: tuple[dict[str, Any], ...]


class HarnessCompiler:
    """Compile one validated operation without executing generated code."""

    def compile(self, parent: HarnessSpec, update: StructuredUpdate) -> CompileResult:
        if update.operator is UpdateOperator.NO_OP:
            return CompileResult(parent.version_id, parent, update, ())
        modules = dict(parent.modules)
        old = modules[update.target_module]
        instructions = list(old.instructions)
        parameters = dict(old.parameters)
        diff: list[dict[str, Any]] = []

        instruction = str(update.arguments.get("instruction", "")).strip()
        condition = str(update.arguments.get("condition", update.scope)).strip()
        value = update.arguments.get("value")

        if update.operator in {UpdateOperator.ADD_GUARD, UpdateOperator.ADD_VALIDATION}:
            if not instruction:
                raise ValueError(
                    f"{update.operator.value} requires arguments.instruction"
                )
            prefix = (
                "Guard" if update.operator is UpdateOperator.ADD_GUARD else "Validation"
            )
            line = f"{prefix} ({condition}): {instruction}"
            if line not in instructions:
                instructions.append(line)
                diff.append({"op": "append_instruction", "value": line})
        elif update.operator is UpdateOperator.RETRY_ON_FAILURE:
            retries = int(
                value if value is not None else update.arguments.get("retries", 2)
            )
            if not 0 <= retries <= 5:
                raise ValueError("retry count must be between 0 and 5")
            parameters["retry_limit"] = retries
            diff.append(
                {"op": "set_parameter", "name": "retry_limit", "value": retries}
            )
        elif update.operator is UpdateOperator.REORDER:
            order = update.arguments.get("order")
            if not isinstance(order, list) or not order:
                raise ValueError("REORDER requires a non-empty arguments.order list")
            parameters["stage_order"] = [str(x) for x in order]
            diff.append(
                {
                    "op": "set_parameter",
                    "name": "stage_order",
                    "value": parameters["stage_order"],
                }
            )
        elif update.operator is UpdateOperator.RETAIN:
            pattern = str(update.arguments.get("pattern", condition)).strip()
            patterns = list(parameters.get("retain_patterns", []))
            if pattern and pattern not in patterns:
                patterns.append(pattern)
            parameters["retain_patterns"] = patterns
            diff.append(
                {"op": "set_parameter", "name": "retain_patterns", "value": patterns}
            )
        elif update.operator is UpdateOperator.COMPRESS:
            max_chars = int(
                value if value is not None else update.arguments.get("max_chars", 8000)
            )
            if max_chars < 1000:
                raise ValueError("max_chars must be at least 1000")
            parameters["max_chars"] = max_chars
            diff.append(
                {"op": "set_parameter", "name": "max_chars", "value": max_chars}
            )
        elif update.operator is UpdateOperator.RETRIEVE:
            key = str(update.arguments.get("key", condition)).strip()
            keys = list(parameters.get("retrieval_keys", []))
            if key and key not in keys:
                keys.append(key)
            parameters["retrieval_keys"] = keys
            diff.append(
                {"op": "set_parameter", "name": "retrieval_keys", "value": keys}
            )
        elif update.operator is UpdateOperator.STOP_RULE:
            required = int(
                value
                if value is not None
                else update.arguments.get("required_consecutive", 1)
            )
            if not 1 <= required <= 3:
                raise ValueError("required_consecutive must be between 1 and 3")
            parameters["required_consecutive"] = required
            diff.append(
                {
                    "op": "set_parameter",
                    "name": "required_consecutive",
                    "value": required,
                }
            )
        else:
            raise ValueError(f"Unsupported update operator: {update.operator.value}")

        modules[update.target_module] = HarnessModuleConfig(
            tuple(instructions), parameters
        )
        payload = {
            "parent": parent.version_id,
            "proposal": asdict(update),
            "modules": {key.value: asdict(config) for key, config in modules.items()},
        }
        version_id = (
            "h-"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode()
            ).hexdigest()[:12]
        )
        child = HarnessSpec(version_id, parent.version_id, modules, update.proposal_id)
        return CompileResult(parent.version_id, child, update, tuple(diff))
