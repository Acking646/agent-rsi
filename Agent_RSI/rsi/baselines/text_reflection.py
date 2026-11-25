"""Reflexion-style failure reflection and same-task retry baseline."""

from __future__ import annotations

from rsi.adapters.harbor_agent import generate_text_reflection
from rsi.harness.schema import HarnessSpec


async def run_text_reflection(runner, tasks, *, rollouts: int):
    """Run H0 once, reflect only on failures, then retry each failed task once."""
    from rsi.formal import _append_jsonl, _write_json

    harness = HarnessSpec.baseline()
    initial = await runner.run_tasks(
        tasks, harness, "text-reflection/initial", rollouts
    )
    failed_by_task = {}
    for index, record in enumerate(initial):
        if record.success or record.task.task_id in failed_by_task:
            continue
        reflection, info = generate_text_reflection(
            model_name=runner.model["id"],
            revision=runner.model.get("revision"),
            device=runner.model.get("device", "cuda"),
            dtype=runner.model.get("dtype", "bfloat16"),
            layers=runner.layers,
            episode=record.to_dict(),
            output_path=runner.output
            / "text-reflection"
            / "reflections"
            / f"{index:04d}.npz",
            seed=int(runner.config["seed"]) + index,
            thinking=bool(runner.agent.get("thinking", True)),
        )
        failed_by_task[record.task.task_id] = reflection
        _append_jsonl(
            runner.output / "text-reflection" / "reflection_records.jsonl",
            {
                "task": record.task.task_id,
                "reflection": reflection,
                "model_call": info["call"],
            },
        )
    failed_tasks = [task for task in tasks if task.name in failed_by_task]
    retry = (
        await runner.run_tasks(
            failed_tasks,
            harness,
            "text-reflection/retry",
            1,
            reflection_memory=failed_by_task,
        )
        if failed_tasks
        else []
    )
    retry_success = sum(record.success for record in retry)
    summary = {
        "method": "Text Reflection (Reflexion semantics)",
        "persistent_harness_update": False,
        "initial_rollouts": len(initial),
        "initial_successes": sum(record.success for record in initial),
        "reflected_tasks": len(failed_by_task),
        "retry_rollouts": len(retry),
        "retry_successes": retry_success,
        "same_task_repair_rate": retry_success / len(retry) if retry else 0.0,
        "verifier_calls": len(initial) + len(retry),
    }
    _write_json(runner.output / "text-reflection" / "result.json", summary)
    return summary
