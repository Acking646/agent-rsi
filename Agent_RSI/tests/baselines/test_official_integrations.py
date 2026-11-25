from types import SimpleNamespace

import pytest

from rsi.baselines.common import harness_to_candidate
from rsi.harness.schema import HarnessSpec


class _Record:
    def __init__(self, task_id, success):
        self.task = SimpleNamespace(task_id=task_id)
        self.success = success
        self.reward = float(success)

    def to_dict(self):
        return {
            "success": self.success,
            "reward": self.reward,
            "steps": [{"action": "echo test", "observation": "ok"}],
            "reflection_text": "checked",
            "verifier": {"reward": self.reward},
        }


class _Runner:
    def __init__(self):
        self.calls = []

    async def run_tasks(self, tasks, harness, run_name, rollouts):
        self.calls.append((tasks, harness, run_name, rollouts))
        return [_Record(task.name, task.name == "pass") for task in tasks]


def test_gepa_harbor_adapter_contract():
    pytest.importorskip("gepa")
    from rsi.baselines.gepa import HarborGEPAAdapter

    runner = _Runner()
    tasks = [
        SimpleNamespace(name="pass"),
        SimpleNamespace(name="fail"),
    ]
    adapter = HarborGEPAAdapter(runner)
    result = adapter.evaluate(
        tasks, harness_to_candidate(HarnessSpec.baseline()), capture_traces=True
    )
    assert result.scores == [1.0, 0.0]
    assert result.num_metric_calls == 2
    reflective = adapter.make_reflective_dataset(
        harness_to_candidate(HarnessSpec.baseline()),
        result,
        ["agent_loop"],
    )
    assert list(reflective) == ["agent_loop"]
    assert len(reflective["agent_loop"]) == 2
    assert runner.calls[0][1].version_id.startswith("gepa-")


def test_autosaddler_prompt_pack_contracts():
    pytest.importorskip("autosaddler")
    from rsi.baselines.autosaddler import HarnessPromptPack

    pack = HarnessPromptPack()
    diagnose = pack.session("diagnose_patch", {"train_case_ids": ["a"]})
    evolve = pack.session("evolve", {"candidate_ids": ["sha256:x"]})
    reflect = pack.session("reflect", {"train_case_ids": ["a"]})
    assert diagnose.kind == "diagnose_patch"
    assert set(diagnose.output_schema["required"]) == {
        "schema_version",
        "updates",
        "diagnosis",
    }
    assert evolve.kind == "evolve"
    assert reflect.kind == "reflect"
