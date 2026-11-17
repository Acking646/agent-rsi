"""Shared records for real Agent RSI experiments.

The official environment runners remain responsible for task execution and
scoring.  These records only normalize their outputs for mechanism analysis,
harness evolution, and reproducible version tracking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class HarnessModule(str, Enum):
    AGENT_LOOP = "agent_loop"
    TOOL_USE = "tool_use"
    OBSERVATION_MANAGEMENT = "observation_management"
    CONTEXT_MANAGEMENT = "context_management"
    COMPLETION_DETECTION = "completion_detection"


class UpdateOperator(str, Enum):
    ADD_GUARD = "ADD_GUARD"
    ADD_VALIDATION = "ADD_VALIDATION"
    RETRY_ON_FAILURE = "RETRY_ON_FAILURE"
    REORDER = "REORDER"
    RETAIN = "RETAIN"
    COMPRESS = "COMPRESS"
    RETRIEVE = "RETRIEVE"
    STOP_RULE = "STOP_RULE"
    NO_OP = "NO_OP"


@dataclass(frozen=True)
class TaskRef:
    dataset: str
    dataset_revision: str
    task_id: str
    split: str
    domain: str
    category: str | None = None


@dataclass(frozen=True)
class AgentStep:
    index: int
    model_call_id: str
    action: str
    observation: str
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    reward: float | None = None
    done: bool = False


@dataclass(frozen=True)
class FailureLabel:
    failed: bool
    responsible_step: int | None
    root_cause: str | None
    target_module: HarnessModule | None
    recommended_operator: UpdateOperator
    label_source: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class UpdateProposal:
    proposal_id: str
    parent_version: str
    target_module: HarnessModule
    operator: UpdateOperator
    scope: str
    rationale: str
    structured_arguments: dict[str, Any] = field(default_factory=dict)
    candidate_diff: str | None = None


@dataclass(frozen=True)
class HarnessVersion:
    version_id: str
    parent_version: str | None
    component_hashes: dict[str, str]
    accepted_proposal: str | None
    accepted: bool
    validation_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeRecord:
    run_id: str
    rollout_id: str
    task: TaskRef
    model_id: str
    model_revision: str
    tokenizer_revision: str
    harness_version: str
    seed: int
    temperature: float
    max_steps: int
    success: bool
    reward: float
    steps: tuple[AgentStep, ...]
    reflection_text: str
    token_boundaries: dict[str, tuple[int, int]]
    activation_index: str | None
    verifier: dict[str, Any]
    failure_label: FailureLabel | None = None
    proposal_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
