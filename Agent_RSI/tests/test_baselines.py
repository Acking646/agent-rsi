from rsi.baselines.common import candidate_to_harness, harness_to_candidate
from rsi.harness.schema import HarnessSpec


def test_baseline_candidate_round_trip():
    source = HarnessSpec.baseline()
    encoded = harness_to_candidate(source)
    restored = candidate_to_harness(encoded, prefix="test")
    assert set(encoded) == {
        "agent_loop",
        "tool_use",
        "observation_management",
        "context_management",
        "completion_detection",
    }
    assert restored.modules == source.modules
    assert restored.version_id.startswith("test-")


def test_candidate_keys_are_frozen():
    candidate = harness_to_candidate(HarnessSpec.baseline())
    candidate.pop("tool_use")
    try:
        candidate_to_harness(candidate)
    except ValueError as error:
        assert "tool_use" in str(error)
    else:
        raise AssertionError("missing module was accepted")


def test_plain_text_candidate_preserves_runtime_parameters():
    candidate = harness_to_candidate(HarnessSpec.baseline())
    candidate["agent_loop"] = "Inspect before editing.\nVerify before completion."
    restored = candidate_to_harness(candidate, prefix="text")
    assert restored.modules[next(iter(restored.modules))].instructions == (
        "Inspect before editing.",
        "Verify before completion.",
    )
    assert restored.modules[next(iter(restored.modules))].parameters["max_steps"] == 40