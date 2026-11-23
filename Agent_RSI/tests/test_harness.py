import json

import pytest

from rsi.contracts import HarnessModule, UpdateOperator
from rsi.evolution.version_store import GateDecision, VersionStore
from rsi.harness.compiler import HarnessCompiler, StructuredUpdate
from rsi.harness.schema import HarnessSpec


def test_compiler_and_version_lineage(tmp_path):
    parent = HarnessSpec.baseline()
    update = StructuredUpdate(
        "p1",
        HarnessModule.TOOL_USE,
        UpdateOperator.ADD_VALIDATION,
        "after edits",
        "Observed unverified edits",
        {"instruction": "Run the repository's public validation command."},
    )
    compiled = HarnessCompiler().compile(parent, update)
    assert compiled.child.parent_version == "h0"
    assert compiled.child.version_id != parent.version_id
    assert any(
        "public validation" in x
        for x in compiled.child.modules[HarnessModule.TOOL_USE].instructions
    )

    store = VersionStore(tmp_path)
    store.initialize(parent)
    active = store.decide(
        parent, compiled.child, GateDecision(True, 0.2, 0.3, 0.0, "passed")
    )
    assert active.version_id == compiled.child.version_id
    assert (
        HarnessSpec.load(tmp_path / "active-harness.json").version_id
        == active.version_id
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["initialize", "accept"]


def test_compiler_rejects_unbounded_retry():
    update = StructuredUpdate(
        "p2",
        HarnessModule.AGENT_LOOP,
        UpdateOperator.RETRY_ON_FAILURE,
        "tool failure",
        "",
        {"retries": 99},
    )
    with pytest.raises(ValueError, match="retry count"):
        HarnessCompiler().compile(HarnessSpec.baseline(), update)
