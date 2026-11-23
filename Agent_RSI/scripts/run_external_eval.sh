#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODEL_REVISION="${MODEL_REVISION:-}"
HARNESS_PATH="${HARNESS_PATH:-$PROJECT_ROOT/runs/lineage/active-harness.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/external-$(date -u +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-terminal-bench@2.1}"
DATASET_PATH="${DATASET_PATH:-}"
LAYERS="${LAYERS:-[3,10,17,24,31]}"
MAX_STEPS="${MAX_STEPS:-40}"
N_ATTEMPTS="${N_ATTEMPTS:-1}"

[[ -x "$PYTHON_BIN" ]] || { echo "Run scripts/setup_school_machine.sh first" >&2; exit 2; }
[[ -f "$HARNESS_PATH" ]] || { echo "Harness not found: $HARNESS_PATH" >&2; exit 2; }

ARGS=(run --yes --agent-import-path rsi.adapters.harbor_agent:WhiteboxHarborAgent -m "$MODEL" -e docker -n 1 -k "$N_ATTEMPTS" -o "$OUTPUT_DIR"
  --ak "harness_path=$HARNESS_PATH" --ak "device=cuda" --ak "dtype=bfloat16" --ak "layers=$LAYERS"
  --ak "max_steps=$MAX_STEPS" --ak "thinking=true")
[[ -z "$MODEL_REVISION" ]] || ARGS+=(--ak "revision=$MODEL_REVISION")
if [[ -n "$DATASET_PATH" ]]; then ARGS+=(-p "$DATASET_PATH"); else ARGS+=(-d "$DATASET"); fi
exec "$PYTHON_BIN" -m harbor.cli.main "${ARGS[@]}"
