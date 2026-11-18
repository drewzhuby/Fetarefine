#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TMPDIR="$PWD/.runtime/tmp" TMP="$PWD/.runtime/tmp" TEMP="$PWD/.runtime/tmp"
export XDG_CACHE_HOME="$PWD/.runtime/cache" MPLCONFIGDIR="$PWD/.runtime/matplotlib" TORCH_HOME="$PWD/.runtime/torch"
export PYTHONPYCACHEPREFIX="$PWD/.runtime/pycache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TORCH_HOME" "$PYTHONPYCACHEPREFIX"
exec python -B "$@"
