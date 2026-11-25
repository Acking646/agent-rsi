"""Official AutoSaddler V2 engine connected to Agent RSI harness evaluation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rsi.baselines.common import candidate_to_harness, harness_to_candidate
from rsi.datasets import SelectedTask
from rsi.harness.schema import HarnessSpec


class QwenAgentProvider:
    """AutoSaddler AgentProvider using the study's pinned Qwen3 checkpoint."""

    def __init__(self, runner):
        self.runner = runner
        self.index = 0

    async def run(self, request):
        from autosaddler.v2.core.domain import Cost
        from autosaddler.v2.prompting.models import SessionResult, Usage

        from rsi.adapters.harbor_agent import _extract_json, get_model

        assets = "\n\n".join(
            f"## {name}\n{text}" for name, text in request.spec.workspace_files.items()
        )
        messages = [
            {
                "role": "system",
                "content": request.spec.system_context
                + "\nReturn exactly one JSON object matching this schema:\n"
                + json.dumps(request.spec.output_schema, ensure_ascii=False),
            },
            {
                "role": "user",
                "content": request.spec.task_prompt + "\n\nSession assets:\n" + assets,
            },
        ]
        try:
            model = get_model(
                self.runner.model["id"],
                self.runner.model.get("revision"),
                self.runner.model.get("device", "cuda"),
                self.runner.model.get("dtype", "bfloat16"),
                self.runner.layers,
            )
            call = model.generate(
                messages,
                self.runner.output
                / "autosaddler"
                / "provider_activations"
                / f"{self.index:05d}.npz",
                max_new_tokens=1536,
                temperature=0.0,
                top_p=1.0,
                thinking=bool(self.runner.agent.get("thinking", True)),
                seed=int(self.runner.config["seed"]) + self.index,
            )
            self.index += 1
            output = _extract_json(call["text"])
            usage = Usage(
                input_tokens=call["input_tokens"],
                output_tokens=call["output_tokens"],
                model=self.runner.model["id"],
            )
            return SessionResult(
                status="completed",
                structured_output=output,
                raw_response=call["text"],
                tool_calls=(),
                usage=(usage,),
                cost=Cost(
                    sessions=1,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - provider errors become session failures
            return SessionResult(
                status="failed",
                structured_output=None,
                raw_response="",
                tool_calls=(),
                usage=(),
                cost=Cost(sessions=1),
                error=f"{type(exc).__name__}: {exc}",
            )


class HarnessEvaluator:
    """AutoSaddler evaluator backed by official Harbor task verifiers."""

    def __init__(self, runner, harness_space, store):
        self.runner = runner
        self.harness_space = harness_space
        self.store = store
        self.index = 0
        self.fingerprint = "agent-rsi-harbor-evaluator/v1"

    async def evaluate(self, candidate, cases, context):
        from autosaddler.v2.core.domain import (
            ArtifactRef,
            Cost,
            Evaluation,
            Observation,
            sha256_digest,
        )

        if not cases or any(case.split != context.split for case in cases):
            raise ValueError("Cases must be nonempty and match EvaluationContext.split")
        materialized = self.harness_space.materialize(candidate, "evaluate")
        try:
            values = json.loads(
                (materialized.root / "candidate.json").read_text(encoding="utf-8")
            )
        finally:
            materialized.release()
        harness = candidate_to_harness(values, prefix="autosaddler")
        tasks = [
            SelectedTask(
                name=case.case_id,
                path=Path(str(case.payload["path"])),
                split=case.split,
                domain=str(case.payload["domain"]),
            )
            for case in cases
        ]
        observations = []
        for repetition in range(context.repetitions):
            pending = []
            cached_by_case = {}
            attempts = {}
            for case in cases:
                cached = context.attempt_sink.completed(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    repetition=repetition,
                )
                if cached is not None:
                    cached_by_case[case.case_id] = cached
                else:
                    attempt_id, attempt_number = context.attempt_sink.start(
                        candidate_id=candidate.candidate_id,
                        case_id=case.case_id,
                        repetition=repetition,
                    )
                    attempts[case.case_id] = (attempt_id, attempt_number)
                    pending.append(case.case_id)
            records = []
            if pending:
                selected = [task for task in tasks if task.name in pending]
                records = await self.runner.run_tasks(
                    selected,
                    harness,
                    f"autosaddler/evaluations/{self.index:05d}-r{repetition}",
                    1,
                )
                self.index += 1
            by_task = {record.task.task_id: record for record in records}
            for case in cases:
                if case.case_id in cached_by_case:
                    observations.append(cached_by_case[case.case_id])
                    continue
                attempt_id, attempt_number = attempts[case.case_id]
                record = by_task.get(case.case_id)
                if record is None:
                    context.attempt_sink.fail(
                        attempt_id, "missing_harbor_record", Cost(rollouts=1)
                    )
                    observation = Observation.create(
                        candidate_id=candidate.candidate_id,
                        case_id=case.case_id,
                        split=case.split,
                        repetition=repetition,
                        disposition="execution_error",
                        score=None,
                        evaluator_fingerprint=self.fingerprint,
                        attempts=attempt_number,
                        cost=Cost(rollouts=1),
                        metadata={"error": "missing_harbor_record"},
                    )
                else:
                    record_ref = self.store.write_json(
                        f"evaluations/records/{record.run_id}.json",
                        record.to_dict(),
                        kind="harbor-episode",
                    )
                    observation = Observation.create(
                        candidate_id=candidate.candidate_id,
                        case_id=case.case_id,
                        split=case.split,
                        repetition=repetition,
                        disposition="success" if record.success else "task_failure",
                        score=float(record.success),
                        objectives={"task_success": float(record.success)},
                        output=record_ref,
                        trace=record_ref,
                        evaluator_fingerprint=self.fingerprint,
                        attempts=attempt_number,
                        cost=Cost(rollouts=1),
                        metadata={
                            "reward": record.reward,
                            "run_id": record.run_id,
                            "harness_version": harness.version_id,
                        },
                    )
                    context.attempt_sink.complete(
                        attempt_id, observation, Cost(rollouts=1)
                    )
                observations.append(observation)
        evaluation_id = sha256_digest(context.operation_id)
        prefix = "evaluations" if context.split == "train" else "quarantine/dev"
        return Evaluation(
            evaluation_id=evaluation_id,
            candidate_id=candidate.candidate_id,
            split=context.split,
            purpose=context.purpose,
            iteration=context.iteration,
            requested_case_ids=tuple(case.case_id for case in cases),
            observations=tuple(observations),
            artifact_dir=ArtifactRef(
                uri=f"{prefix}/{evaluation_id.removeprefix('sha256:')}",
                kind="evaluation-directory",
            ),
        )


class HarnessEvidenceBuilder:
    def __init__(self, store):
        self.store = store

    def build(self, evaluation):
        from autosaddler.v2.core.domain import sha256_digest

        if evaluation.split != "train":
            raise ValueError("Optimization evidence must come from training")
        evidence_id = sha256_digest(evaluation.evaluation_id).removeprefix("sha256:")
        return self.store.write_json(
            f"evidence/{evidence_id}/evidence.json",
            {
                "schema_version": "agent-rsi-autosaddler-evidence/v1",
                "evaluation_id": evaluation.evaluation_id,
                "observations": [
                    {
                        "case_id": item.case_id,
                        "disposition": item.disposition,
                        "score": item.score,
                        "reward": item.metadata.get("reward"),
                        "run_id": item.metadata.get("run_id"),
                        "trace": item.trace.uri if item.trace else None,
                    }
                    for item in evaluation.observations
                ],
            },
            kind="training-evidence",
        )


def _object_schema(required, properties):
    return {
        "type": "object",
        "required": list(required),
        "properties": properties,
        "additionalProperties": False,
    }


class HarnessPromptPack:
    """Structured prompts for AutoSaddler's diagnose/patch/evolve/reflect phases."""

    def session(self, kind: str, context: Mapping[str, Any]):
        from autosaddler.v2.core.domain import canonical_json
        from autosaddler.v2.prompting.models import SessionSpec

        if kind == "diagnose_patch":
            properties = {
                "schema_version": {"const": "autosaddler-diagnosis-patch/v1"},
                "updates": {
                    "type": "object",
                    "propertyNames": {
                        "enum": [
                            "agent_loop",
                            "tool_use",
                            "observation_management",
                            "context_management",
                            "completion_detection",
                        ]
                    },
                    "additionalProperties": {"type": "string"},
                    "minProperties": 1,
                },
                "diagnosis": {"type": "string"},
            }
            required = ("schema_version", "updates", "diagnosis")
            task = (
                "Diagnose the failed training episodes. Propose the smallest useful update "
                "to one or more five-module harness components. Every update value must be "
                "a complete JSON-encoded object with keys instructions (string array) and "
                "parameters (object). Preserve safe constraints and never add test leakage."
            )
        elif kind == "evolve":
            properties = {
                "schema_version": {"const": "autosaddler-evolution/v1"},
                "parent_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "component_sources": {"type": "object", "additionalProperties": {"type": "string"}},
                "rationale": {"type": "string"},
            }
            required = (
                "schema_version",
                "parent_ids",
                "component_sources",
                "rationale",
            )
            task = (
                "Select accepted parent candidates and, where justified, select a source "
                "candidate for each component. Use only candidate IDs present in context."
            )
        elif kind == "reflect":
            properties = {
                "schema_version": {"const": "autosaddler-reflection/v1"},
                "lessons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["scope", "statement", "evidence_case_ids"],
                        "properties": {
                            "scope": {"type": "string"},
                            "statement": {"type": "string"},
                            "evidence_case_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            }
            required = ("schema_version", "lessons")
            task = (
                "Summarize only lessons supported by the aggregate development feedback. "
                "Do not reveal or speculate about held-out test tasks."
            )
        else:
            raise ValueError(f"Unknown AutoSaddler session kind: {kind}")
        return SessionSpec(
            kind=kind,
            system_context=(
                "You optimize a constrained software-agent harness. Use only provided "
                "training/development artifacts and obey the structured output contract."
            ),
            task_prompt=task,
            skills={},
            output_schema=_object_schema(required, properties),
            workspace_files={"session_context.json": canonical_json(context) + "\n"},
            capabilities=frozenset({"read_workspace"}),
        )


def _cases(tasks, split):
    from autosaddler.v2.core.domain import Case

    return tuple(
        Case(
            case_id=task.name,
            split=split,
            payload={"path": str(task.path.resolve()), "domain": task.domain},
        )
        for task in tasks
    )


def build_scenario(runner, train_tasks, dev_tasks, run_dir, store, *, rollouts=1):
    from autosaddler.v2.core.ports import ScenarioComponents
    from autosaddler.v2.harness.component_map import ComponentMapHarnessSpace

    harness_space = ComponentMapHarnessSpace(
        baseline=harness_to_candidate(HarnessSpec.baseline()),
        store_root=run_dir / "candidates",
        materialization_root=run_dir / "materialized",
        validator=lambda value: candidate_to_harness(
            value, prefix="autosaddler-validation"
        ),
    )
    return ScenarioComponents(
        name="agent-rsi-harness",
        version="1",
        harness_space=harness_space,
        evaluator=HarnessEvaluator(runner, harness_space, store),
        evidence_builder=HarnessEvidenceBuilder(store),
        prompt_pack=HarnessPromptPack(),
        train_cases=_cases(train_tasks, "train"),
        development_cases=_cases(dev_tasks, "development"),
        required_capabilities=frozenset({"read_workspace"}),
        evaluation_repetitions=rollouts,
        resolved_entities={
            "resolved/sources/harness.json": {
                "space": "component_map",
                "modules": list(harness_to_candidate(HarnessSpec.baseline())),
            },
            "resolved/sources/dataset.json": {
                "train_case_ids": [task.name for task in train_tasks],
                "development_case_ids": [task.name for task in dev_tasks],
                "test": {"state": "opaque", "opened": False},
            },
        },
    )


def run_autosaddler(
    runner,
    train_tasks,
    dev_tasks,
    heldout_tasks,
    *,
    max_rollouts: int,
    max_iterations: int,
    batch_size: int,
    rollouts: int = 1,
):
    """Run official AutoSaddlerEngine and evaluate its selected harness."""
    from autosaddler.v2.core.engine import AutoSaddlerEngine
    from autosaddler.v2.core.policies import (
        BudgetPolicy,
        EpochShuffledTaskSelectionPolicy,
        FullOnAcceptDevelopment,
        MatchedValidStrictImprovement,
        MeanDevelopmentRanking,
        PolicyBundle,
    )
    from autosaddler.v2.storage.local import LocalRunStore

    run_dir = runner.output / "autosaddler" / "official_run"
    store = LocalRunStore(run_dir=run_dir, run_id="agent-rsi-autosaddler")
    scenario = build_scenario(
        runner, train_tasks, dev_tasks, run_dir, store, rollouts=rollouts
    )
    policies = PolicyBundle(
        task_selection=EpochShuffledTaskSelectionPolicy(
            batch_size=batch_size, seed=int(runner.config["seed"])
        ),
        acceptance=MatchedValidStrictImprovement(),
        development=FullOnAcceptDevelopment(),
        ranking=MeanDevelopmentRanking(),
        budget=BudgetPolicy(
            max_rollouts=max_rollouts, max_iterations=max_iterations
        ),
    )
    store.initialize(
        resolved_config={
            "schema_version": "agent-rsi-autosaddler/v1",
            "max_rollouts": max_rollouts,
            "max_iterations": max_iterations,
            "batch_size": batch_size,
        },
        resolved_entities=scenario.resolved_entities,
    )
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=QwenAgentProvider(runner),
        policies=policies,
        diagnosis_patch_timeout_seconds=1800,
        selection_timeout_seconds=1800,
        reflection_timeout_seconds=1800,
    )
    result = engine.run()
    selected = next(
        candidate
        for candidate in engine._state().candidates.values()
        if candidate.candidate_id == result.selected_candidate_id
    )
    materialized = scenario.harness_space.materialize(selected, "inspect")
    try:
        values = json.loads(
            (materialized.root / "candidate.json").read_text(encoding="utf-8")
        )
    finally:
        materialized.release()
    harness = candidate_to_harness(values, prefix="autosaddler")
    harness.save(runner.output / "autosaddler" / "selected_harness.json")
    heldout = asyncio.run(
        runner.run_tasks(
            list(heldout_tasks),
            harness,
            "autosaddler/frozen-internal-heldout",
            rollouts,
        )
    )
    summary = {
        "method": "AutoSaddler V2 official engine",
        "selected_harness": harness.version_id,
        "selected_candidate_id": result.selected_candidate_id,
        "development_score": result.development_score,
        "iterations": result.iterations,
        "heldout_rollouts": len(heldout),
        "heldout_successes": sum(record.success for record in heldout),
    }
    target = runner.output / "autosaddler" / "result.json"
    target.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
