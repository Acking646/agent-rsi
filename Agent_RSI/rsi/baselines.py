"""Shared held-out readouts for reflection and simple memory controls."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .backends import CODES


def evaluate_baselines(backend, records):
    train = records["train"]
    test = records["test"]

    def corpus(row):
        _, episode, _ = row
        return str(episode.task) + " " + str(episode.trace) + " " + episode.feedback["message"]

    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    memory = tfidf.fit_transform([corpus(row) for row in train])
    similarities = (tfidf.transform([corpus(row) for row in test]) @ memory.T).toarray()
    results = []
    for row, similarities_row in zip(test, similarities):
        run_id, episode, run = row
        picked = int(similarities_row.argmax())
        recalled_id, _, recalled = train[picked]
        no_reflection = backend.run(run.context, episode, [])
        results.append({
            "run_id": run_id, "task_id": episode.task_id,
            "expected": episode.labels["recommended_action"],
            "keep": "A",
            "no_reflection": CODES[int(no_reflection.logits.argmax())],
            "trajectory_tfidf_retrieval": CODES[int(recalled.logits.argmax())],
            "retrieved_run_id": recalled_id,
            "retrieved_similarity": float(similarities_row[picked]),
            "full_reflection": CODES[int(run.logits.argmax())],
        })
    metrics = {
        k: float(np.mean([row[k] == row["expected"] for row in results]))
        for k in ("keep", "no_reflection", "trajectory_tfidf_retrieval", "full_reflection")
    }
    return {"heldout_accuracy": metrics, "predictions": results}
