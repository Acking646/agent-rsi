# Baseline integrations

The repository vendors the exact official implementations as Git submodules:

| Baseline | Official source | Pinned commit | Integration |
|---|---|---:|---|
| Text Reflection | [Reflexion](https://github.com/noahshinn/reflexion) | `218cf0ef1df84b05ce379dd4a8e47f17766733a0` | Native Harbor retry loop using the paper/repository's failure-reflection memory semantics |
| GEPA | [gepa-ai/gepa](https://github.com/gepa-ai/gepa) | `ce5b0d6e3d6af6c6dbec6de2c54536bbb961d309` | Official `gepa.optimize` and `GEPAAdapter` |
| AutoSaddler | [microsoft/AutoSaddler](https://github.com/microsoft/AutoSaddler) | `bb0217ae9743e5e1bc3428458fd2d75be3f67894` | Official V2 `AutoSaddlerEngine` and `ComponentMapHarnessSpace` |

Initialize the pinned sources after cloning:

```bash
git submodule update --init --recursive
```

Install the optional baseline packages in the experiment environment:

```bash
uv pip install -r requirements-baselines.txt
```

All three methods use the same Qwen3 checkpoint, task split, Harbor environment, official
task verifier, rollout record format, and five Harness modules. They differ only in the
authorized optimization behavior.

## Text Reflection

Text Reflection follows Reflexion's task-local retry design. It runs H0, asks the same
model to summarize a failed trajectory, and injects that reflection into one retry of
the same task. It does not create a new Harness version and its memory is not reused on
other tasks.

```bash
agent-rsi --config configs/real_pilot.json text-reflection \
  --dataset-root data/modularrsi --domain tb \
  --out runs/text-reflection-tb --rollouts 1 --max-tasks 30
```

## GEPA

The adapter maps the five modules to GEPA's native `dict[str, str]` candidate. Each
metric call runs Harbor and scores task success from its official verifier. Rich traces
contain agent commands, observations, model reflections, rewards, and verifier output.
GEPA's official Pareto selection, round-robin component mutation, validation tracking,
budget, persistence, and resume code remain unchanged.

```bash
agent-rsi --config configs/real_pilot.json gepa \
  --dataset-root data/modularrsi --domain tb --out runs/gepa-tb \
  --max-metric-calls 400 --reflection-minibatch-size 3 --rollouts 1
```

## AutoSaddler

The V2 scenario uses AutoSaddler's frozen-key component-map space. Its official engine
selects the train minibatch, diagnoses failures, applies structured component updates,
performs matched before/after acceptance, quarantines development evaluation, ranks
accepted candidates, records the event log, and enforces rollout/iteration budgets.
The project supplies the Harbor evaluator, evidence artifact builder, structured prompt
pack, and a Qwen3 `AgentProvider`.

```bash
agent-rsi --config configs/real_pilot.json autosaddler \
  --dataset-root data/modularrsi --domain tb --out runs/autosaddler-tb \
  --max-rollouts 400 --max-iterations 3 --batch-size 8 --rollouts 1
```

A full benchmark run requires Linux, Docker, the ModularRSI task checkout, and an NVIDIA
GPU large enough for the configured Qwen3 checkpoint. Local CPU checks validate candidate
round trips, prompt/result contracts, and the upstream optimizer engines without claiming
benchmark task completion.
