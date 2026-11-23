"""Collect task-disjoint real activations and fit probes on ModularRSI tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rsi.datasets import modularrsi_splits
from rsi.formal import FormalRunner, _write_json
from rsi.harness.schema import HarnessSpec
from rsi.mechanism.probe import load_rows, run


async def collect(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    splits = modularrsi_splits(args.dataset_root, args.domain, int(config["seed"]))
    runner = FormalRunner(config, args.out)
    harness = HarnessSpec.baseline()
    rollouts = args.rollouts or config["agent"]["rollouts_per_task"]
    definitions = (
        ("evolve", "train", args.max_train_tasks),
        ("dev", "dev", args.max_dev_tasks),
        ("heldout", "test", args.max_test_tasks),
    )
    diagnosis_paths = {}
    for source, target, limit in definitions:
        records = await runner.run_tasks(
            splits[source][: limit or None], harness, f"{target}/episodes", rollouts
        )
        runner.diagnose(records, harness, f"{target}/diagnosis")
        diagnosis_paths[target] = args.out / target / "diagnosis" / "diagnoses.jsonl"
    if not all(path.is_file() for path in diagnosis_paths.values()):
        raise RuntimeError("At least one split produced no failed episodes to diagnose")
    report = run(
        *(load_rows(diagnosis_paths[name]) for name in ("train", "dev", "test"))
    )
    _write_json(args.out / "probe_report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/real_pilot.json"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--domain", choices=("tb", "swe"), default="tb")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rollouts", type=int)
    parser.add_argument("--max-train-tasks", type=int)
    parser.add_argument("--max-dev-tasks", type=int)
    parser.add_argument("--max-test-tasks", type=int)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("--out must be a new directory")
    asyncio.run(collect(args))


if __name__ == "__main__":
    main()
