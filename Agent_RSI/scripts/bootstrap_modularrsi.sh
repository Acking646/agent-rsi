#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel)"
CODE_PATH="$PROJECT_ROOT/Agent_RSI/third_party/ModularRSI"
REVISION_FILE="$PROJECT_ROOT/Agent_RSI/data/manifests/DATASET_REVISION.txt"
DATA_REVISION="$(tr -d '[:space:]' < "$REVISION_FILE")"
DATA_PATH="${MODULARRSI_DATA_DIR:-$PROJECT_ROOT/Agent_RSI/.local/ModularRSI_2000_Instances}"
DATA_URL="https://huggingface.co/datasets/IQuestLab/ModularRSI_2000_Instances"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    echo "The dataset contains ':' in task paths and cannot be checked out on Windows/NTFS." >&2
    echo "Run this script on the school Linux machine or inside a Linux filesystem." >&2
    exit 2
    ;;
esac

git -C "$REPO_ROOT" submodule update --init --depth 1 -- Agent_RSI/third_party/ModularRSI

if [[ -d "$DATA_PATH/.git" ]]; then
  git -C "$DATA_PATH" fetch origin "$DATA_REVISION" --depth 1
else
  if [[ -e "$DATA_PATH" ]]; then
    echo "Refusing to overwrite non-repository path: $DATA_PATH" >&2
    exit 3
  fi
  mkdir -p "$(dirname "$DATA_PATH")"
  git clone --no-checkout "$DATA_URL" "$DATA_PATH"
fi

git -C "$DATA_PATH" checkout --detach "$DATA_REVISION"
if git -C "$DATA_PATH" lfs env >/dev/null 2>&1; then
  git -C "$DATA_PATH" lfs pull
fi

cmp "$PROJECT_ROOT/Agent_RSI/data/manifests/tb_manifest.json" "$DATA_PATH/tb/manifest.json"
cmp "$PROJECT_ROOT/Agent_RSI/data/manifests/swe_manifest.json" "$DATA_PATH/swe/manifest.json"

printf 'ModularRSI code: %s\n' "$(git -C "$CODE_PATH" rev-parse HEAD)"
printf 'ModularRSI data: %s\n' "$(git -C "$DATA_PATH" rev-parse HEAD)"
printf 'Dataset path: %s\n' "$DATA_PATH"