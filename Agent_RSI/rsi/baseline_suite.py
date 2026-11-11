"""Run comparable RSI controls and literature-inspired baselines on one task stream."""
import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .backends import CODES, HFBackend, ToyBackend, context_for
from .environment import ACTIONS, FAULTS, Harness, assess_update, evaluate, execute, proposal

METHODS = (
    "frozen",
    "no_reflection_writeback",
    "reflect_writeback",
    "rise_retry",
    "reasoningbank_memory",
    "gepa_prompt",
    "godel_sequential",
    "dgm_archive",
)


@dataclass(frozen=True)
class Version:
    id: int
    parent: int | None
    harness: Harness
    score: float
    action: str


@dataclass(frozen=True)
class Memory:
    turn: int
    key: str
    lesson: str
    action: str


class Experiment:
    def __init__(self, backend, method: str, config: dict, calibration: range, holdout: range):
        self.backend = backend
        self.method = method
        self.config = config
        self.calibration = calibration
        self.holdout = holdout
        self.rng = random.Random(config["seed"])
        self.harness = Harness()
        self.cost = Counter()
        self.archive = []
        self.deployed_version = None
        if method == "dgm_archive":
            initial_score = mean(evaluate(self.harness, calibration).values())
            self.cost["calibration_environment_calls"] += len(calibration) * len(FAULTS)
            self.archive.append(Version(0, None, self.harness, initial_score, "A"))
            self.deployed_version = 0
        self.children = Counter()
        self.memory = []
        self.vectorizer = HashingVectorizer(n_features=4096, ngram_range=(1, 2), alternate_sign=False)

    def validate(self, parent, code):
        # assess_update always evaluates the parent and evaluates a new candidate only if distinct.
        self.cost["calibration_environment_calls"] += len(self.calibration) * len(FAULTS) * (
            1 + int(code != "A" and ACTIONS[code][1] not in parent.rules)
        )
        return assess_update(parent, code, self.calibration)

    def reflection_decision(self, episode, parent, memory=None):
        context = context_for(episode, parent)
        if memory is not None:
            # Inserting before the reflection marker keeps the same template boundary.
            lesson = json.dumps(asdict(memory), ensure_ascii=False)
            prefix, marker = context.rsplit("[REFLECTION]", 1)
            context = prefix + "Previously validated experience: " + lesson + "\n[REFLECTION]" + marker
        reflection = []
        if self.method != "no_reflection_writeback":
            self.cost["reflection_calls"] += 1
            reflection = self.backend.reflect(context, episode)
        self.cost["decision_forwards"] += 1
        run = self.backend.run(context, episode, reflection)
        ranked = [CODES[i] for i in np.argsort(-run.logits, kind="stable")]
        return ranked, run.text, run.logits.tolist(), context

    @staticmethod
    def memory_key(episode):
        trace = " ".join(step["action"] for step in episode.trace)
        return f'{episode.task["request"]} {trace} {episode.feedback["message"]}'

    def retrieve(self, episode):
        if not self.memory:
            return None, None
        keys = [self.memory_key(episode), *(m.key for m in self.memory)]
        vectors = self.vectorizer.transform(keys)
        sims = (vectors[0] @ vectors[1:].T).toarray()[0]
        index = int(np.argmax(sims))
        similarity = float(sims[index])
        return (self.memory[index] if similarity >= self.config["memory_min_similarity"] else None,
                similarity)

    def select_parent(self):
        # Matches DGM_outer.py choose_selfimproves() with score_child_prop.
        weights = [1 / ((1 + np.exp(-10 * (v.score - .5))) * (1 + self.children[v.id]))
                   for v in self.archive]
        return self.rng.choices(self.archive, weights=weights, k=1)[0]

    def search(self, parent: Harness, ranked: list[str], parent_version: Version | None):
        candidate_logs = []
        accepted_versions = []
        used_rule_sets = {tuple(sorted(v.harness.rules)) for v in self.archive}
        codes = [code for code in ranked if code != "A" and ACTIONS[code][1] not in parent.rules]
        for code in codes[:self.config["candidates"]]:
            candidate, audit = self.validate(parent, code)
            value = mean(audit["calibration_candidate"].values())
            audit["calibration_score"] = value
            candidate_logs.append(audit)
            if parent_version is not None:
                # These declarative rules are all valid states. Archive membership is separate
                # from the calibration rule used to deploy a version.
                trial = Harness(parent.version + 1, (*parent.rules, ACTIONS[code][1]))
                key = tuple(sorted(trial.rules))
                if key not in used_rule_sets:
                    version = Version(len(self.archive), parent_version.id, trial, value, code)
                    self.archive.append(version)
                    self.children[parent_version.id] += 1
                    used_rule_sets.add(key)
            if audit["accepted"]:
                accepted_versions.append((value, candidate, code))
        if parent_version is not None:
            selected = max(self.archive, key=lambda v: (v.score, -v.id))
            action = selected.action if selected.id != self.deployed_version else "A"
            self.deployed_version = selected.id
            return selected.harness, action, candidate_logs
        if accepted_versions:
            _, candidate, code = max(accepted_versions, key=lambda item: item[0])
            return candidate, code, candidate_logs
        return parent, "A", candidate_logs

    def run_turn(self, turn: int, case: int, fault: str):
        before = self.harness
        deployed_before = self.deployed_version
        parent_version = self.select_parent() if self.method == "dgm_archive" else None
        parent = parent_version.harness if parent_version is not None else before
        attempt = execute(case, fault, before)
        # All methods see identical input task IDs, in the same order. The archive may
        # additionally run the task using a parent different from the deployed version.
        if parent != before:
            diagnostic_episode = execute(case, fault, parent)
            self.cost["parent_replays"] += 1
        else:
            diagnostic_episode = attempt

        recalled, similarity = (self.retrieve(diagnostic_episode)
                                if self.method == "reasoningbank_memory" else (None, None))
        ranked, reflection, logits, context = (
            self.reflection_decision(diagnostic_episode, parent, recalled)
            if self.method != "frozen" else (["A"], "", None, None)
        )
        action = ranked[0]
        candidate_logs = []
        retry = None
        if self.method == "gepa_prompt" and action != "A":
            self.harness, action, candidate_logs = self.search(parent, ranked, None)
        elif parent_version is not None and action != "A":
            self.harness, action, candidate_logs = self.search(parent, ranked, parent_version)
        elif self.method in {"rise_retry", "reasoningbank_memory"} and action != "A":
            operation = ACTIONS[action][1]
            if operation not in before.rules:
                self.cost["retry_environment_calls"] += 1
                trial = Harness(before.version, (*before.rules, operation))
                retry = execute(case, fault, trial)
                if self.method == "reasoningbank_memory" and retry.feedback["success"]:
                    self.memory.append(Memory(turn, self.memory_key(attempt), reflection, action))
        elif self.method in {"reflect_writeback", "no_reflection_writeback"} and action != "A":
            self.harness, audit = self.validate(before, action)
            candidate_logs.append(audit)
        elif self.method == "godel_sequential" and action != "A":
            operation = ACTIONS[action][1]
            if operation not in before.rules:
                trial = Harness(before.version + 1, (*before.rules, operation))
                self.cost["retry_environment_calls"] += 1
                retry = execute(case, fault, trial)
                if not attempt.feedback["success"] and retry.feedback["success"]:
                    self.harness = trial

        final_attempt = retry if retry is not None and retry.feedback["success"] else attempt
        return {
            "turn": turn, "case_id": case, "scenario": fault,
            "before": asdict(before), "after": asdict(self.harness),
            "parent_id": parent_version.id if parent_version is not None else None,
            "deployed_before_id": deployed_before, "deployed_after_id": self.deployed_version,
            "initial_attempt": asdict(attempt), "diagnostic_attempt": asdict(diagnostic_episode),
            "initial_success": attempt.feedback["success"], "within_task_success": final_attempt.feedback["success"],
            "reflection": reflection, "context": context, "proposal_logits": logits,
            "model_top_proposal": proposal(ranked[0]), "selected_action": proposal(action),
            "calibration_checks": candidate_logs, "retry": asdict(retry) if retry else None,
            "retrieved_memory": asdict(recalled) if recalled else None, "retrieval_similarity": similarity,
            "memory_size": len(self.memory), "archive_size": len(self.archive),
            "holdout_after_writeback": evaluate(self.harness, self.holdout),
        }

    def run(self, stream):
        rows = [self.run_turn(turn, case, fault) for turn, (case, fault) in enumerate(stream)]
        result = {
            "stream_first_attempt_success": mean(r["initial_success"] for r in rows),
            "stream_within_task_success": mean(r["within_task_success"] for r in rows),
            "final_holdout_by_scenario": evaluate(self.harness, self.holdout),
            "accepted_rule_updates": sum(r["before"] != r["after"] for r in rows),
            "final_rules": list(self.harness.rules),
            "archive_size": len(self.archive),
            "memory_size": len(self.memory),
            "cost": {key: int(self.cost[key]) for key in (
                "reflection_calls", "decision_forwards", "calibration_environment_calls",
                "retry_environment_calls", "parent_replays")},
            "primary_environment_calls": len(rows),
            "heldout_evaluation_calls": (len(rows) + 1) * len(self.holdout) * len(FAULTS),
        }
        return rows, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=("toy", "hf"), default="toy")
    p.add_argument("--model")
    p.add_argument("--revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--layers", type=int, nargs="+", default=[0])
    p.add_argument("--reflection-tokens", type=int, default=64)
    args = p.parse_args()
    if args.backend == "hf" and not args.model:
        p.error("--backend hf requires --model")
    if args.out.exists():
        p.error("--out must be a new directory")
    if args.reflection_tokens < 1 or len(args.methods) != len(set(args.methods)):
        p.error("Use a positive reflection token limit and list each method once")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pattern = config["stream_pattern"]
    if not pattern or any(fault not in FAULTS for fault in pattern):
        p.error("The stream pattern must contain only valid task scenarios")
    if any(type(config[key]) is not int or config[key] < 1 for key in (
            "calibration_cases", "holdout_cases", "candidates")):
        p.error("Calibration cases, holdout cases and candidates must be positive integers")
    if not 0 <= config["memory_min_similarity"] <= 1:
        p.error("Memory threshold must be between zero and one")
    backend = (ToyBackend(args.reflection_tokens) if args.backend == "toy" else
               HFBackend(args.model, args.layers, args.reflection_tokens, args.device, args.revision))
    args.out.mkdir(parents=True)
    stream = list(enumerate(pattern))
    calibration = range(len(stream), len(stream) + config["calibration_cases"])
    holdout = range(calibration.stop, calibration.stop + config["holdout_cases"])
    manifest = {"backend": backend.metadata, "config": config, "methods": args.methods,
                "reflection_tokens": args.reflection_tokens, "stream": stream,
                "calibration_ids": list(calibration), "holdout_ids": list(holdout)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = {}
    for method in args.methods:
        experiment = Experiment(backend, method, config, calibration, holdout)
        rows, result[method] = experiment.run(stream)
        with (args.out / f"{method}.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if method == "dgm_archive":
            archive = [asdict(v) for v in experiment.archive]
            (args.out / "dgm_versions.json").write_text(json.dumps(archive, indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {len(result)} method summaries to {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
