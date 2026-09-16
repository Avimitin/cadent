#!/usr/bin/env bash
set -euo pipefail

backend="${1:-cpu}"
if [[ $# -gt 1 || ( "$backend" != cpu && "$backend" != cuda ) ]]; then
  echo "Usage: bash scripts/setup.sh [cpu|cuda]" >&2
  exit 2
fi
if [[ -z "${UV_PYTHON:-}" ]]; then
  echo "Enter the Nix environment first: nix develop" >&2
  exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv sync --locked --extra "$backend"

check_args=()
if [[ "$backend" == cuda ]]; then
  check_args+=(--require-cuda)
fi
uv run --no-sync python scripts/check_environment.py "${check_args[@]}"
