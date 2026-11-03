#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" -u "$script_dir/main.py" \
  --data "$script_dir/data/drozy.npz" "$@"
