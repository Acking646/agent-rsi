"""Run the complete experiment, or switch to a local Hugging Face checkpoint."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .analysis import probe, run_interventions, summarize_interventions
from .backends import CODES, HFBackend, ToyBackend, context_for
from .baselines import evaluate_baselines
from .environment import ACTIONS, FAULTS, Harness, assess_update, evaluate, execute, proposal


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_line(file, row):
    file.write(json.dumps(row, ensure_ascii=False) + "\n")


def choice(run):
    return CODES[int(np.argmax(run.logits))]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("toy", "hf"), default="toy")
    p.add_argument("--model", help="Local model directory or Hugging Face checkpoint ID")
    p.add_argument("--revision", help="Optional model/tokenizer revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 1])
    p.add_argument("--site", nargs=2, metavar=("LAYER", "COMPONENT"),
                   help="Use a fixed intervention site, bypassing dev site selection")
    p.add_argument("--cases-per-split", type=int, default=8)
    p.add_argument("--reflection-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--sae-width", type=int, default=0, help="Optional SAE width (requires torch)")
    p.add_argument("--sae-steps", type=int, default=400)
    p.add_argument("--out", type=Path, required=True, help="New output directory")
    args = p.parse_args()
    if args.cases_per_split < 3 or args.reflection_tokens < 1 or args.seed < 0:
        p.error("Expected at least 3 cases per split, a positive token budget and nonnegative seed")
    if args.sae_width < 0 or args.sae_steps < 1:
        p.error("Expected nonnegative SAE width and positive SAE steps")
    if args.backend == "hf" and not args.model:
        p.error("--backend hf requires --model")
    if args.out.exists():
        p.error("--out must be a new directory")
    return args


def main():
    args = parse_args()
    backend = (ToyBackend(args.reflection_tokens) if args.backend == "toy" else
               HFBackend(args.model, args.layers, args.reflection_tokens, args.device, args.revision))
    out = args.out
    out.mkdir(parents=True)
    (out / "activations").mkdir()
    seed_harness = Harness()
    order = np.random.default_rng(args.seed).permutation(args.cases_per_split * 3).tolist()
    split_ids = {split: order[i * args.cases_per_split:(i + 1) * args.cases_per_split]
                 for i, split in enumerate(("train", "dev", "test"))}
    boundary = len(order)
    calibration = range(boundary, boundary + 12)
    transfer = range(boundary + 12, boundary + 24)
    stream_cases = range(boundary + 24, boundary + 30)
    write_json(out / "manifest.json", {
        "backend": backend.metadata, "cli": vars(args) | {"out": str(out)},
        "split_ids": split_ids, "calibration_ids": list(calibration),
        "transfer_ids": list(transfer), "stream_ids": list(stream_cases),
        "action_map": {code: proposal(code) for code in ACTIONS},
    })

    records = {}
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as file:
        for split, cases in split_ids.items():
            records[split] = []
            for case in cases:
                for fault in FAULTS:
                    run_id = f"{split}-{case}-{fault}"
                    episode = execute(case, fault, seed_harness)
                    context = context_for(episode, seed_harness)
                    reflection = backend.reflect(context, episode)
                    run = backend.run(context, episode, reflection)
                    keep = sorted({pos for positions in run.phases.values() for pos in positions})
                    npz = f"activations/{run_id}.npz"
                    np.savez_compressed(out / npz, token_positions=np.asarray(keep),
                                        **{f"l{layer}_{module}": act[keep]
                                           for (layer, module), act in run.activations.items()})
                    run.activations = {
                        site: {pos: values[pos].copy() for pos in keep}
                        for site, values in run.activations.items()
                    }
                    jsonl_line(file, {
                        "run_id": run_id, "split": split, "harness": asdict(seed_harness),
                        **asdict(episode), "scenario": fault, "reflection": run.text,
                        "context": run.context, "token_ids": run.token_ids, "phases": run.phases,
                        "readout": run.tokens, "candidate_logits": run.logits.tolist(),
                        "proposal": proposal(choice(run)), "activation_file": npz,
                    })
                    records[split].append((run_id, episode, run))
    sites = list(records["train"][0][2].activations)
    forced = (int(args.site[0]), args.site[1]) if args.site else None
    if forced is not None and forced not in sites:
        raise SystemExit(f"Intervention site must be one of {sites}")
    probe_report, site = probe(records, sites, forced)
    write_json(out / "probes.json", probe_report)

    baselines = evaluate_baselines(backend, records)
    write_json(out / "baselines.json", baselines)

    sae = None
    if args.sae_width:
        from .sae import SparseAutoencoder

        def anchor_matrix(split):
            return np.stack([r[2].activations[site][r[2].phases["reflection_end"][0]]
                             for r in records[split]])

        sae = SparseAutoencoder(anchor_matrix("train"), args.sae_width, args.seed, args.sae_steps)
        sae.save(out / "sae.npz")
        write_json(out / "sae.json", sae.stats | {"site": list(site),
                                                  "heldout_reconstruction_mse": sae.mse(anchor_matrix("test"))})

    intervention_rows = list(run_interventions(backend, records, site, args.seed, sae))
    with (out / "interventions.jsonl").open("w", encoding="utf-8") as file:
        for row in intervention_rows:
            jsonl_line(file, row)

    # Replay each intervention proposal independently through the same update gate.
    # The intervention comparison starts from the identical frozen harness snapshot.
    with (out / "intervention_updates.jsonl").open("w", encoding="utf-8") as file:
        cache = {}
        for row in intervention_rows:
            arms = {}
            for field in ("clean_choice", "intervention_choice"):
                code = row[field]
                if code not in cache:
                    updated, audit = assess_update(seed_harness, code, calibration)
                    cache[code] = audit | {"transfer_after": evaluate(updated, transfer)}
                arms[field] = cache[code]
            jsonl_line(file, {"target_run_id": row["target_run_id"], "method": row["method"], **arms})

    # A separate stream tests the persistence of accepted updates across actual calls.
    harness = seed_harness
    with (out / "stream.jsonl").open("w", encoding="utf-8") as file:
        for turn, (case, fault) in enumerate(zip(stream_cases, ("missing", "stale", "verify", "missing", "stale", "verify"))):
            episode = execute(case, fault, harness)
            context = context_for(episode, harness)
            run = backend.run(context, episode, backend.reflect(context, episode))
            harness, update = assess_update(harness, choice(run), calibration)
            jsonl_line(file, {
                "run_id": f"stream-{turn}", "turn_id": turn, **asdict(episode),
                "actual_input": context, "reflection": run.text,
                "proposal_logits": run.logits.tolist(), "update": update,
                "transfer_after": evaluate(harness, transfer),
            })

    summary = {
        "backend": backend.metadata["backend"], "episodes": sum(map(len, records.values())),
        "heldout_proposal_accuracy": float(np.mean([
            choice(run) == ep.labels["recommended_action"] for _, ep, run in records["test"]
        ])),
        "selected_site": list(site),
        "baseline_accuracies": baselines["heldout_accuracy"],
        "interventions": summarize_interventions(intervention_rows, args.seed),
        "seed_transfer": evaluate(seed_harness, transfer),
        "final_transfer": evaluate(harness, transfer),
        "final_harness": asdict(harness),
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
