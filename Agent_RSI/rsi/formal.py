"""Formal ModularRSI/Harbor experiment runner. No synthetic task path exists here."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from rsi.adapters.harbor_agent import generate_structured_update
from rsi.adapters.harbor_records import from_trial_result, with_diagnosis
from rsi.datasets import SelectedTask, modularrsi_splits, selection_digest
from rsi.evolution.version_store import GateDecision, VersionStore
from rsi.harness.compiler import HarnessCompiler
from rsi.harness.schema import HarnessSpec


def _json_default(value: Any):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, ensure_ascii=False, default=_json_default) + "\n"
        )


class FormalRunner:
    def __init__(self, config: dict[str, Any], output: Path):
        self.config = config
        self.output = output
        self.output.mkdir(parents=True, exist_ok=False)
        self.model = config["model"]
        self.agent = config["agent"]
        self.dataset = config["dataset"]
        self.layers = tuple(
            config.get("activation_capture", {}).get("layers", (3, 9, 15, 21, 27))
        )

    async def run_tasks(
        self,
        tasks: list[SelectedTask],
        harness: HarnessSpec,
        run_name: str,
        rollouts: int,
        reflection_memory: dict[str, str] | None = None,
    ) -> list[Any]:
        from harbor.models.trial.config import (
            AgentConfig,
            EnvironmentConfig,
            TaskConfig,
            TrialConfig,
        )
        from harbor.trial.trial import Trial

        run_dir = self.output / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        harness_path = run_dir / "harness.json"
        harness.save(harness_path)
        records = []
        for task in tasks:
            for rollout in range(rollouts):
                seed = int(self.config["seed"]) + rollout
                digest = hashlib.sha256(
                    f"{run_name}:{task.name}:{rollout}:{harness.version_id}".encode()
                ).hexdigest()[:8]
                trial_name = f"{task.name[:28]}__r{rollout}__{digest}"
                kwargs = {
                    "harness_path": str(harness_path.resolve()),
                    "revision": self.model.get("revision"),
                    "device": self.model.get("device", "cuda"),
                    "dtype": self.model.get("dtype", "bfloat16"),
                    "layers": list(self.layers),
                    "max_steps": int(self.agent.get("max_steps", 40)),
                    "max_new_tokens": int(self.agent.get("max_new_tokens", 768)),
                    "temperature": float(self.agent.get("temperature", 0.0)),
                    "top_p": float(self.agent.get("top_p", 0.95)),
                    "thinking": bool(self.agent.get("thinking", True)),
                    "seed": seed,
                    "workdir": self.agent.get("workdir"),
                    "reflection_memory": (reflection_memory or {}).get(task.name),
                }
                trial_config = TrialConfig(
                    task=TaskConfig(path=task.path.resolve()),
                    trial_name=trial_name,
                    trials_dir=run_dir / "trials",
                    agent=AgentConfig(
                        import_path="rsi.adapters.harbor_agent:WhiteboxHarborAgent",
                        model_name=self.model["id"],
                        kwargs=kwargs,
                    ),
                    environment=EnvironmentConfig(
                        type=self.agent.get("environment", "docker"), delete=True
                    ),
                )
                result = await (await Trial.create(trial_config)).run()
                _write_json(run_dir / "raw_results" / f"{trial_name}.json", result)
                record = from_trial_result(
                    result,
                    dataset=self.dataset["id"],
                    dataset_revision=str(self.dataset.get("revision", "unknown")),
                    split=task.split,
                    domain=task.domain,
                    model_id=self.model["id"],
                    model_revision=str(self.model.get("revision", "unknown")),
                    tokenizer_revision=str(self.model.get("revision", "unknown")),
                    harness_version=harness.version_id,
                    seed=seed,
                    temperature=float(self.agent.get("temperature", 0.0)),
                    max_steps=int(self.agent.get("max_steps", 40)),
                    rollout_id=f"{task.name}-r{rollout}",
                )
                records.append(record)
                _append_jsonl(run_dir / "episodes.jsonl", record.to_dict())
        _write_json(run_dir / "summary.json", self.summary(records, tasks, harness))
        return records

    @staticmethod
    def summary(records, tasks, harness):
        successes = sum(int(record.success) for record in records)
        return {
            "harness_version": harness.version_id,
            "tasks": len(tasks),
            "rollouts": len(records),
            "successes": successes,
            "pass_rate": successes / len(records) if records else 0.0,
            "selection_digest": selection_digest(tasks),
            "mean_reward": sum(record.reward for record in records) / len(records)
            if records
            else 0.0,
        }

    def diagnose(self, records, harness, name):
        diagnosed = []
        target = self.output / name
        target.mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(records):
            if record.success:
                continue
            try:
                update, info = generate_structured_update(
                    model_name=self.model["id"],
                    revision=self.model.get("revision"),
                    device=self.model.get("device", "cuda"),
                    dtype=self.model.get("dtype", "bfloat16"),
                    layers=self.layers,
                    harness=harness,
                    episode=record.to_dict(),
                    output_path=target / "activations" / f"proposal-{index:04d}.npz",
                    seed=int(self.config["seed"]) + index,
                    thinking=bool(self.agent.get("thinking", True)),
                )
                enriched = with_diagnosis(
                    record,
                    failed=True,
                    responsible_step=int(info["responsible_step"])
                    if info.get("responsible_step") is not None
                    else None,
                    root_cause=info.get("root_cause"),
                    target_module=update.target_module,
                    operator=update.operator,
                    proposal_id=update.proposal_id,
                    evidence=("official verifier reward <= 0", info["reflection"]),
                )
                diagnosed.append((enriched, update))
                _append_jsonl(
                    target / "diagnoses.jsonl",
                    {
                        "record": enriched.to_dict(),
                        "proposal": asdict(update),
                        "diagnosis": info,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - record one invalid proposal and continue
                _append_jsonl(
                    target / "diagnosis_errors.jsonl",
                    {"run_id": record.run_id, "error": f"{type(exc).__name__}: {exc}"},
                )
        return diagnosed


def _paired_gate(
    parent_records,
    candidate_records,
    same_parent,
    same_candidate,
    min_gain,
    max_regression,
):
    def keyed(records):
        return {
            (r.task.task_id, r.rollout_id.rsplit("-r", 1)[-1]): r.success
            for r in records
        }

    parent, candidate = keyed(parent_records), keyed(candidate_records)
    common = sorted(set(parent) & set(candidate))
    if not common:
        raise ValueError("No paired dev rollouts between parent and candidate")
    parent_score = sum(parent[key] for key in common) / len(common)
    candidate_score = sum(candidate[key] for key in common) / len(common)
    regression_rate = sum(parent[key] and not candidate[key] for key in common) / len(
        common
    )
    same_p, same_c = keyed(same_parent), keyed(same_candidate)
    same_common = set(same_p) & set(same_c)
    same_batch_score = (
        sum(same_c[key] for key in same_common) / len(same_common)
        if same_common
        else None
    )
    same_non_regress = not same_common or sum(
        same_c[key] for key in same_common
    ) >= sum(same_p[key] for key in same_common)
    accepted = (
        candidate_score >= parent_score + min_gain
        and regression_rate <= max_regression
        and same_non_regress
    )
    reason = (
        "dev gain and regression gate passed"
        if accepted
        else "candidate failed paired validation gate"
    )
    return GateDecision(
        accepted,
        parent_score,
        candidate_score,
        regression_rate,
        reason,
        same_batch_score,
    )


async def pilot(args, config):
    tasks = modularrsi_splits(args.dataset_root, args.domain, int(config["seed"]))[
        "pilot"
    ][: args.max_tasks or None]
    runner = FormalRunner(config, args.out)
    harness = HarnessSpec.baseline()
    records = await runner.run_tasks(
        tasks, harness, "pilot", args.rollouts or config["agent"]["rollouts_per_task"]
    )
    runner.diagnose(records, harness, "pilot_diagnosis")


async def evolve(args, config):
    splits = modularrsi_splits(args.dataset_root, args.domain, int(config["seed"]))
    evolve_tasks, dev_tasks = (
        splits["evolve"][: args.max_evolve_tasks or None],
        splits["dev"][: args.max_dev_tasks or None],
    )
    runner = FormalRunner(config, args.out)
    store, compiler = VersionStore(args.out / "lineage"), HarnessCompiler()
    active = store.initialize(HarnessSpec.baseline())
    rollouts = args.rollouts or config["agent"]["rollouts_per_task"]
    for generation in range(args.generations):
        prefix = f"generation-{generation:02d}"
        parent_evolve = await runner.run_tasks(
            evolve_tasks, active, prefix + "/parent-evolve", rollouts
        )
        unique = {}
        for record, proposal in runner.diagnose(
            parent_evolve, active, prefix + "/diagnosis"
        ):
            unique.setdefault(proposal.proposal_id, (record, proposal))
        candidates = list(unique.values())[: args.candidates]
        if not candidates:
            break
        parent_dev = await runner.run_tasks(
            dev_tasks, active, prefix + "/parent-dev", rollouts
        )
        evaluated = []
        for index, (source_record, proposal) in enumerate(candidates):
            try:
                compiled = compiler.compile(active, proposal)
            except (TypeError, ValueError) as exc:
                _append_jsonl(
                    runner.output / prefix / "invalid_candidates.jsonl",
                    {
                        "proposal": asdict(proposal),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                continue
            if compiled.child.version_id == active.version_id:
                continue
            source_tasks = [
                task for task in evolve_tasks if task.name == source_record.task.task_id
            ]
            same_parent = [
                r for r in parent_evolve if r.task.task_id == source_record.task.task_id
            ]
            candidate_same = await runner.run_tasks(
                source_tasks,
                compiled.child,
                prefix + f"/candidate-{index:02d}-same-batch",
                rollouts,
            )
            candidate_dev = await runner.run_tasks(
                dev_tasks,
                compiled.child,
                prefix + f"/candidate-{index:02d}-dev",
                rollouts,
            )
            decision = _paired_gate(
                parent_dev,
                candidate_dev,
                same_parent,
                candidate_same,
                args.min_gain,
                args.max_regression,
            )
            evaluated.append((decision, compiled, source_record.run_id))
        if not evaluated:
            break
        _write_json(
            runner.output / prefix / "candidate_evaluations.json",
            [
                {
                    "decision": asdict(item[0]),
                    "proposal": asdict(item[1].proposal),
                    "candidate_version": item[1].child.version_id,
                    "diff": item[1].diff,
                    "source_run": item[2],
                }
                for item in evaluated
            ],
        )
        decision, best, source_run = max(
            evaluated,
            key=lambda item: (
                item[0].accepted,
                item[0].candidate_score - item[0].parent_score,
                -item[0].regression_rate,
            ),
        )
        active = store.decide(
            active,
            best.child,
            decision,
            {"generation": generation, "source_run": source_run, "diff": best.diff},
        )
        if not decision.accepted:
            break
    await runner.run_tasks(
        splits["heldout"][: args.max_heldout_tasks or None],
        active,
        "frozen-internal-heldout",
        rollouts,
    )


async def text_reflection_baseline(args, config):
    from rsi.baselines.text_reflection import run_text_reflection

    tasks = modularrsi_splits(args.dataset_root, args.domain, int(config["seed"]))[
        "pilot"
    ][: args.max_tasks or None]
    runner = FormalRunner(config, args.out)
    await run_text_reflection(
        runner,
        tasks,
        rollouts=args.rollouts or 1,
    )


def gepa_baseline(args, config):
    from rsi.baselines.gepa import run_gepa

    splits = modularrsi_splits(args.dataset_root, args.domain, int(config["seed"]))
    runner = FormalRunner(config, args.out)
    run_gepa(
        runner,
        splits["evolve"][: args.max_evolve_tasks or None],
        splits["dev"][: args.max_dev_tasks or None],
        splits["heldout"][: args.max_heldout_tasks or None],
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        rollouts=args.rollouts or 1,
    )


def autosaddler_baseline(args, config):
    from rsi.baselines.autosaddler import run_autosaddler

    splits = modularrsi_splits(args.dataset_root, args.domain, int(config["seed"]))
    runner = FormalRunner(config, args.out)
    run_autosaddler(
        runner,
        splits["evolve"][: args.max_evolve_tasks or None],
        splits["dev"][: args.max_dev_tasks or None],
        splits["heldout"][: args.max_heldout_tasks or None],
        max_rollouts=args.max_rollouts,
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        rollouts=args.rollouts or 1,
    )

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/real_pilot.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("pilot", "evolve", "text-reflection", "gepa", "autosaddler"):
        p = sub.add_parser(name)
        p.add_argument("--dataset-root", type=Path, required=True)
        p.add_argument("--domain", choices=("tb", "swe"), default="tb")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--rollouts", type=int)
        if name in {"pilot", "text-reflection"}:
            p.add_argument("--max-tasks", type=int)
        else:
            p.add_argument("--max-evolve-tasks", type=int)
            p.add_argument("--max-dev-tasks", type=int)
            p.add_argument("--max-heldout-tasks", type=int)
        if name == "evolve":
            p.add_argument("--generations", type=int, default=3)
            p.add_argument("--candidates", type=int, default=3)
            p.add_argument("--min-gain", type=float, default=0.01)
            p.add_argument("--max-regression", type=float, default=0.05)
        elif name == "gepa":
            p.add_argument("--max-metric-calls", type=int, default=400)
            p.add_argument("--reflection-minibatch-size", type=int, default=3)
        elif name == "autosaddler":
            p.add_argument("--max-rollouts", type=int, default=400)
            p.add_argument("--max-iterations", type=int, default=3)
            p.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.out.exists():
        raise SystemExit("--out must be a new directory")
    if args.command == "pilot":
        asyncio.run(pilot(args, config))
    elif args.command == "evolve":
        asyncio.run(evolve(args, config))
    elif args.command == "text-reflection":
        asyncio.run(text_reflection_baseline(args, config))
    elif args.command == "gepa":
        gepa_baseline(args, config)
    else:
        autosaddler_baseline(args, config)


if __name__ == "__main__":
    main()
