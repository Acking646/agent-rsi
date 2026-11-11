"""Linear probes, deterministic site selection, and paired activation interventions."""
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .backends import CODES, Intervention
from .environment import Episode

STAGES = ("feedback_end", "reflection_25", "reflection_50",
          "reflection_75", "reflection_100", "reflection_end", "proposal")
LABELS = ("outcome", "error_location", "root_cause", "recommended_action")


def phase_position(phases: dict, stage: str) -> int | None:
    if stage.startswith("reflection_") and stage != "reflection_end":
        positions = phases["reflection"]
        fraction = int(stage.rsplit("_", 1)[1]) / 100
        return positions[int((len(positions) - 1) * fraction)] if positions else None
    return phases[stage][0]


def score(model, features, y) -> dict:
    pred = model.predict(features)
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def probe(records: dict, sites: list, forced_site=None) -> tuple[dict, tuple[int, str]]:
    rows = {}
    selected_scores = {}
    text_controls = {}
    for label in LABELS:
        train = records["train"]
        for variant, get_text in {
            "task_only": lambda r: str(r[1].task),
            "feedback_only": lambda r: r[1].feedback["message"],
            "reflection_only": lambda r: r[2].text or "[empty]",
        }.items():
            clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"),
                                LogisticRegression(max_iter=500))
            clf.fit([get_text(r) for r in train], [r[1].labels[label] for r in train])
            text_controls[f"{label}/{variant}"] = {
                split: score(clf, [get_text(r) for r in items],
                             [r[1].labels[label] for r in items])
                for split, items in records.items() if split != "train"
            }

        for site in sites:
            for stage in STAGES:
                def samples(items):
                    return [(r[2].activations[site][pos], r[1].labels[label])
                            for r in items
                            if (pos := phase_position(r[2].phases, stage)) is not None]

                data = {key: samples(items) for key, items in records.items()}
                if any(not d for d in data.values()) or len({y for _, y in data["train"]}) < 2:
                    continue
                train_x, train_y = zip(*data["train"])
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
                model.fit(np.stack(train_x), train_y)
                metrics = {}
                for split in ("dev", "test"):
                    x, y = zip(*data[split])
                    metrics[split] = score(model, np.stack(x), y)
                rows[f"{label}/L{site[0]}/{site[1]}/{stage}"] = metrics
                if label == "recommended_action" and stage == "reflection_end":
                    selected_scores[site] = metrics["dev"]["macro_f1"]
    if forced_site is None:
        # Residual sites are the preregistered intervention candidates.
        selected = max((s for s in selected_scores if s[1] == "residual"),
                       key=lambda s: (selected_scores[s], -s[0]))
    else:
        selected = forced_site
    return {"linear_probes": rows, "text_baselines": text_controls,
            "selection_split": "dev", "selected_site": list(selected)}, selected


def run_interventions(backend, records: dict, site: tuple[int, str], seed: int, sae=None):
    by_task = defaultdict(list)
    for r in records["test"]:
        by_task[r[1].task_id].append(r)
    centroids = defaultdict(list)
    for _, ep, run in records["train"]:
        centroids[ep.labels["recommended_action"]].append(
            run.activations[site][run.phases["reflection_end"][0]])
    centroids = {c: np.mean(x, axis=0) for c, x in centroids.items()}
    center = np.mean(list(centroids.values()), axis=0)
    rng = np.random.default_rng(seed)
    for group in by_task.values():
        for i, (run_id, target_ep, target) in enumerate(group):
            donor_i = (i + 1 + int(target_ep.task_id) % (len(group) - 1)) % len(group)
            donor_id, donor_ep, donor = group[donor_i]
            target_label = target_ep.labels["recommended_action"]
            donor_label = donor_ep.labels["recommended_action"]
            a = target.phases["reflection_end"][0]
            b = donor.phases["reflection_end"][0]
            if target.token_ids[a] != donor.token_ids[b]:
                raise ValueError("Paired anchor tokens differ")
            src, ref = target.activations[site][a], donor.activations[site][b]
            direction = centroids[donor_label] - centroids[target_label]
            unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
            scale = float(np.linalg.norm(ref - src))
            random = rng.standard_normal(len(src)).astype(np.float32)
            random = random * (scale / max(float(np.linalg.norm(random)), 1e-12))
            interventions = {
                "self_patch": Intervention(site, src),
                "donor_patch": Intervention(site, ref),
                "random_matched": Intervention(site, random, add=True),
                "mean_direction_steer": Intervention(site, .5 * scale * unit, add=True),
                "mean_direction_ablate": Intervention(site, src - np.dot(src - center, unit) * unit),
            }
            sae_feature = None
            if sae is not None:
                sae_feature, patches = sae.patches(site, src, ref, direction)
                interventions.update(patches)
            reflection = [target.token_ids[p] for p in target.phases["reflection"]]
            s, t = CODES.index(donor_label), CODES.index(target_label)
            margin = float(target.logits[s] - target.logits[t])
            for method, op in interventions.items():
                result = backend.run(target.context, target_ep, reflection, op)
                patched_margin = float(result.logits[s] - result.logits[t])
                before = CODES[int(np.argmax(target.logits))]
                after = CODES[int(np.argmax(result.logits))]
                yield {
                    "task_id": target_ep.task_id, "target_run_id": run_id, "donor_run_id": donor_id,
                    "site": list(site), "anchor": "reflection_end",
                    "target_token_index": a, "donor_token_index": b,
                    "method": method, "sae_feature": sae_feature if method.startswith("sae_") else None,
                    "target_label": target_label, "donor_label": donor_label,
                    "clean_choice": before, "intervention_choice": after,
                    "clean_logits": target.logits.tolist(), "intervention_logits": result.logits.tolist(),
                    "donor_minus_target_clean": margin, "donor_minus_target_intervention": patched_margin,
                    "paired_margin_change": patched_margin - margin,
                    "became_donor_choice": bool(before != donor_label and after == donor_label),
                    "donor_choice_rate_change": int(after == donor_label) - int(before == donor_label),
                }


def summarize_interventions(rows: list[dict], seed: int) -> dict:
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["method"]][row["task_id"]].append(row)
    rng = np.random.default_rng(seed)
    summaries = {}
    for method, clusters in groups.items():
        task_means = np.array([[np.mean([r[k] for r in cluster]) for k in
                                ("paired_margin_change", "donor_choice_rate_change")]
                               for cluster in clusters.values()])
        n = len(task_means)
        bootstrap = task_means[rng.integers(n, size=(1000, n))].mean(axis=1)
        summaries[method] = {
            "task_groups": n, "pairs": sum(map(len, clusters.values())),
            "mean_paired_margin_change": float(task_means[:, 0].mean()),
            "margin_change_cluster_bootstrap_95ci": np.quantile(bootstrap[:, 0], [.025, .975]).tolist(),
            "donor_choice_rate_change": float(task_means[:, 1].mean()),
        }
    return summaries
