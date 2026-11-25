"""Official GEPA optimizer connected to the five-module Harbor harness."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any

from rsi.baselines.common import candidate_to_harness, harness_to_candidate
from rsi.harness.schema import HarnessSpec


class HarborGEPAAdapter:
    """GEPA adapter whose metric calls are official Harbor verifier rollouts."""

    def __init__(self, runner, *, rollouts: int = 1):
        self.runner = runner
        self.rollouts = rollouts
        self.evaluation_index = 0

    def evaluate(self, batch, candidate, capture_traces: bool = False):
        from gepa.core.adapter import EvaluationBatch

        harness = candidate_to_harness(candidate, prefix="gepa")
        run_name = f"gepa/evaluations/{self.evaluation_index:05d}"
        self.evaluation_index += 1
        records = asyncio.run(
            self.runner.run_tasks(list(batch), harness, run_name, self.rollouts)
        )
        by_task = {}
        for record in records:
            by_task.setdefault(record.task.task_id, []).append(record)
        outputs, scores, trajectories = [], [], []
        for task in batch:
            task_records = by_task.get(task.name, [])
            score = (
                sum(float(record.success) for record in task_records) / len(task_records)
                if task_records
                else 0.0
            )
            output = {
                "task": task.name,
                "score": score,
                "rollouts": [record.to_dict() for record in task_records],
            }
            outputs.append(output)
            scores.append(score)
            trajectories.append(output)
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            objective_scores=[{"task_success": score} for score in scores],
            num_metric_calls=len(records),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        trajectories = eval_batch.trajectories or eval_batch.outputs
        dataset = {}
        for component in components_to_update:
            examples = []
            for trace in trajectories:
                rollouts = trace.get("rollouts", [])
                compact = []
                for rollout in rollouts:
                    compact.append(
                        {
                            "success": rollout.get("success"),
                            "reward": rollout.get("reward"),
                            "steps": rollout.get("steps", []),
                            "reflection_text": rollout.get("reflection_text", ""),
                            "verifier": rollout.get("verifier", {}),
                        }
                    )
                examples.append(
                    {
                        "Inputs": {
                            "task_id": trace.get("task", ""),
                            "current_component": candidate[component],
                        },
                        "Generated Outputs": compact,
                        "Feedback": (
                            "Official verifier passed."
                            if trace.get("score", 0.0) >= 1.0
                            else "Official verifier failed; revise only this harness component "
                            "using the observed commands, errors, and completion evidence."
                        ),
                        "score": trace.get("score", 0.0),
                    }
                )
            dataset[component] = examples
        return dataset


class QwenReflectionLM:
    """Callable GEPA reflection LM backed by the same pinned open checkpoint."""

    def __init__(self, runner):
        self.runner = runner
        self.index = 0

    def __call__(self, prompt: str | list[dict[str, str]]) -> str:
        from rsi.adapters.harbor_agent import get_model

        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": str(prompt)}]
        )
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
            / "gepa"
            / "reflection_activations"
            / f"{self.index:05d}.npz",
            max_new_tokens=1536,
            temperature=0.0,
            top_p=1.0,
            thinking=bool(self.runner.agent.get("thinking", True)),
            seed=int(self.runner.config["seed"]) + self.index,
        )
        self.index += 1
        return call["text"]


def run_gepa(
    runner,
    train_tasks,
    dev_tasks,
    heldout_tasks,
    *,
    max_metric_calls: int,
    reflection_minibatch_size: int = 3,
    rollouts: int = 1,
):
    """Run the official GEPA search and freeze its selected candidate for heldout."""
    import gepa

    seed_candidate = harness_to_candidate(HarnessSpec.baseline())
    adapter = HarborGEPAAdapter(runner, rollouts=rollouts)
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=list(train_tasks),
        valset=list(dev_tasks),
        adapter=adapter,
        reflection_lm=QwenReflectionLM(runner),
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        module_selector="round_robin",
        reflection_prompt_template=(
            "Improve this software-agent harness module using the execution evidence. "
            "The current module is <curr_param>. Evidence: <side_info>. Return only the "
            "complete replacement: either a JSON object with instructions and parameters "
            "or plain instruction lines. Preserve benchmark privacy constraints."
        ),
        run_dir=str(runner.output / "gepa" / "official_run"),
        seed=int(runner.config["seed"]),
        write_agent_state=True,
        display_progress_bar=True,
    )
    best = result.best_candidate
    if not isinstance(best, dict):
        raise TypeError("GEPA returned a non-mapping candidate")
    harness = candidate_to_harness(best, prefix="gepa")
    harness.save(runner.output / "gepa" / "selected_harness.json")
    heldout = asyncio.run(
        runner.run_tasks(
            list(heldout_tasks),
            harness,
            "gepa/frozen-internal-heldout",
            rollouts,
        )
    )
    summary = {
        "method": "GEPA official optimizer",
        "selected_harness": harness.version_id,
        "max_metric_calls": max_metric_calls,
        "adapter_metric_calls": adapter.evaluation_index,
        "heldout_rollouts": len(heldout),
        "heldout_successes": sum(record.success for record in heldout),
        "heldout_pass_rate": (
            sum(record.success for record in heldout) / len(heldout) if heldout else 0.0
        ),
        "official_result": result.to_dict(),
    }
    target = runner.output / "gepa" / "result.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return summary
