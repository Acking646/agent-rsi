"""Convert official Harbor TrialResult objects into Agent RSI records."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from rsi.contracts import (
    AgentStep,
    EpisodeRecord,
    FailureLabel,
    HarnessModule,
    TaskRef,
    UpdateOperator,
)


def _dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {"value": str(value)}


def from_trial_result(
    result,
    *,
    dataset: str,
    dataset_revision: str,
    split: str,
    domain: str,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    harness_version: str,
    seed: int,
    temperature: float,
    max_steps: int,
    rollout_id: str,
) -> EpisodeRecord:
    context = result.agent_result
    metadata = dict((context.metadata if context else None) or {})
    raw_steps = metadata.get("steps", [])
    steps = tuple(
        AgentStep(
            index=int(row.get("index", index)),
            model_call_id=f"{rollout_id}-call-{index}",
            action=str(row.get("command", "")),
            observation=str(row.get("observation", "")),
            tool_name="shell" if row.get("command") else None,
            tool_arguments={"command": row.get("command", "")}
            if row.get("command")
            else {},
            reward=None,
            done=bool(row.get("accepted_complete", False)),
        )
        for index, row in enumerate(raw_steps)
    )
    verifier = _dump(result.verifier_result)
    rewards = verifier.get("rewards") or {}
    reward = float(rewards.get("reward", next(iter(rewards.values()), 0.0)))
    success = reward > 0.0 and result.exception_info is None
    trace = metadata.get("agent_rsi_trace")
    reflection = "\n".join(str(row.get("reflection", "")) for row in raw_steps)
    return EpisodeRecord(
        run_id=result.trial_name,
        rollout_id=rollout_id,
        task=TaskRef(dataset, dataset_revision, result.task_name, split, domain),
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        harness_version=harness_version,
        seed=seed,
        temperature=temperature,
        max_steps=max_steps,
        success=success,
        reward=reward,
        steps=steps,
        reflection_text=reflection,
        token_boundaries={},
        activation_index=str(trace) if trace else None,
        verifier=verifier | {"exception": _dump(result.exception_info)},
    )


def with_diagnosis(
    record: EpisodeRecord,
    *,
    failed: bool,
    responsible_step: int | None,
    root_cause: str | None,
    target_module: HarnessModule | None,
    operator: UpdateOperator,
    proposal_id: str | None,
    evidence: tuple[str, ...],
) -> EpisodeRecord:
    label = FailureLabel(
        failed=failed,
        responsible_step=responsible_step,
        root_cause=root_cause,
        target_module=target_module,
        recommended_operator=operator,
        label_source="official_verifier_plus_whitebox_diagnosis",
        evidence=evidence,
    )
    return replace(record, failure_label=label, proposal_id=proposal_id)
