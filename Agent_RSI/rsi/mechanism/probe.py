"""Task-disjoint text controls and prospective activation probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELS = ("target_module", "operator", "root_cause")
STAGES = ("prompt_end", "proposal_prefix")
NPZ_METADATA = {
    "token_ids",
    "activation_token_positions",
    "input_tokens",
}


def load_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        proposal, diagnosis, record = (
            value["proposal"],
            value["diagnosis"],
            value["record"],
        )
        call = diagnosis["call"]
        proposal_prefix_index = int(
            call.get("field_prefix_positions", {}).get("target_module", 0)
        )
        rows.append(
            {
                "task_id": record["task"]["task_id"],
                "text": diagnosis.get("reflection", ""),
                "target_module": proposal["target_module"],
                "operator": proposal["operator"],
                "root_cause": diagnosis.get("root_cause") or "unknown",
                "activation_path": Path(call["activation_file"]),
                "proposal_prefix_index": proposal_prefix_index,
            }
        )
    return rows


def metrics(model, x, y):
    pred = model.predict(x)
    result = {
        "n": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    if hasattr(model, "predict_proba") and len(set(y)) > 1:
        try:
            result["macro_auroc_ovr"] = float(
                roc_auc_score(
                    y,
                    model.predict_proba(x),
                    multi_class="ovr",
                    average="macro",
                    labels=model.classes_,
                )
            )
        except ValueError:
            result["macro_auroc_ovr"] = None
    return result


def _stage_index(data, row, stage: str) -> int:
    input_tokens = int(data["input_tokens"][0])
    if input_tokens < 1:
        raise ValueError(f"Invalid input token count: {row['activation_path']}")
    if stage == "prompt_end":
        return 0
    if stage == "proposal_prefix":
        return row["proposal_prefix_index"]
    raise ValueError(f"Unknown activation stage: {stage}")


def feature(row, site, stage):
    with np.load(row["activation_path"], allow_pickle=False) as data:
        matrix = data[site]
        if matrix.ndim != 2 or not len(matrix):
            raise ValueError(f"Empty activation {site}: {row['activation_path']}")
        index = _stage_index(data, row, stage)
        if not 0 <= index < len(matrix):
            raise ValueError(
                f"Stage {stage} index {index} outside activation length {len(matrix)}"
            )
        return matrix[index].astype(np.float32)


def run(train, dev, test):
    if not train or not dev or not test:
        raise ValueError("train, dev, and test diagnoses must all be non-empty")
    overlap = (
        ({r["task_id"] for r in train} & {r["task_id"] for r in dev})
        | ({r["task_id"] for r in train} & {r["task_id"] for r in test})
        | ({r["task_id"] for r in dev} & {r["task_id"] for r in test})
    )
    if overlap:
        raise ValueError(f"Task leakage across splits: {sorted(overlap)[:5]}")
    with np.load(train[0]["activation_path"], allow_pickle=False) as data:
        sites = sorted(key for key in data.files if key not in NPZ_METADATA)
    report = {
        "text_controls": {},
        "linear_probes": {},
        "selected_sites": {},
        "stages": {
            "prompt_end": "last prompt token before any diagnosis text",
            "proposal_prefix": "last token before the target_module field",
        },
    }
    for label in LABELS:
        if len({row[label] for row in train}) < 2:
            continue
        text = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2)),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        text.fit(
            [row["text"] or "[empty]" for row in train], [row[label] for row in train]
        )
        report["text_controls"][label] = {
            "dev": metrics(
                text,
                [row["text"] or "[empty]" for row in dev],
                [row[label] for row in dev],
            ),
            "test": metrics(
                text,
                [row["text"] or "[empty]" for row in test],
                [row[label] for row in test],
            ),
        }
        candidates = []
        for stage in STAGES:
            for site in sites:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                )
                model.fit(
                    np.stack([feature(row, site, stage) for row in train]),
                    [row[label] for row in train],
                )
                values = {
                    "dev": metrics(
                        model,
                        np.stack([feature(row, site, stage) for row in dev]),
                        [row[label] for row in dev],
                    ),
                    "test": metrics(
                        model,
                        np.stack([feature(row, site, stage) for row in test]),
                        [row[label] for row in test],
                    ),
                }
                key = f"{label}/{stage}/{site}"
                report["linear_probes"][key] = values
                candidates.append((values["dev"]["macro_f1"], stage, site))
        _, stage, site = max(candidates)
        report["selected_sites"][label] = {"stage": stage, "site": site}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(load_rows(args.train), load_rows(args.dev), load_rows(args.test))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()