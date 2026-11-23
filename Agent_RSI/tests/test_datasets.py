import json

from rsi.datasets import modularrsi_splits, selection_digest


def test_modularrsi_task_disjoint_split(tmp_path):
    domain = tmp_path / "tb"
    tasks = domain / "tasks"
    tasks.mkdir(parents=True)
    rows = []
    for index in range(120):
        name = f"task-{index:03d}"
        (tasks / name).mkdir()
        rows.append({"name": name, "split": "train"})
    (domain / "manifest.json").write_text(json.dumps({"tasks": rows}), encoding="utf-8")
    first = modularrsi_splits(tmp_path, "tb", 2026)
    second = modularrsi_splits(tmp_path, "tb", 2026)
    assert [len(first[x]) for x in ("evolve", "dev", "heldout", "pilot")] == [
        80,
        20,
        20,
        30,
    ]
    assert selection_digest(first["evolve"]) == selection_digest(second["evolve"])
    assert not ({x.name for x in first["evolve"]} & {x.name for x in first["dev"]})
    assert not ({x.name for x in first["dev"]} & {x.name for x in first["heldout"]})
