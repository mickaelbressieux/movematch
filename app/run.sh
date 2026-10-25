#!/usr/bin/env bash
# run.sh — start MoveMatch with gunicorn (Linux/macOS).
# On Windows, or without gunicorn, use:  python app/app.py
set -euo pipefail
cd "$(dirname "$0")"

export KEEP_UPLOADS="${KEEP_UPLOADS:-0}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5000}"
WEB_WORKERS="${WEB_WORKERS:-2}"

ROOT="$(cd .. && pwd)"
VENV="${MOVEMATCH_VENV:-$ROOT/.venv}"
PY="$VENV/bin/python"
STAMP="$VENV/.movematch-installed"   # newer than pyproject.toml == deps up to date

die() { echo "run.sh: $*" >&2; exit 1; }

py_ok() {  # $1 is a Python >= 3.11
  "$1" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null
}

find_python() {  # first Python >= 3.11 on PATH
  local c
  for c in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$c" >/dev/null && py_ok "$(command -v "$c")"; then
      command -v "$c"; return 0
    fi
  done
  return 1
}

# An explicit MOVEMATCH_VENV is used as-is; only the repo's own .venv is managed.
if [[ -n "${MOVEMATCH_VENV:-}" ]]; then
  "$PY" -c 'import flask, gunicorn' 2>/dev/null \
    || die "$PY lacks the MoveMatch dependencies (run: $PY -m pip install $ROOT)."
else
  if [[ ! -x "$PY" ]]; then
    echo "run.sh: creating $VENV (first run)..."
    if command -v uv >/dev/null; then
      uv venv -q -p 3.12 "$VENV"        # uv fetches Python 3.12 if none is installed
    elif SYS_PY="$(find_python)"; then
      if ! "$SYS_PY" -m venv "$VENV"; then
        rm -rf "$VENV"   # don't leave a half-built venv for the next run
        die "'$SYS_PY -m venv' failed (Debian/Ubuntu: apt install python3-venv, or install uv)."
      fi
    else
      die "no Python >= 3.11 found. Install Python 3.11+ or uv (https://docs.astral.sh/uv/)."
    fi
  fi
  py_ok "$PY" || die "$VENV uses Python < 3.11; delete it and re-run."

  if [[ ! -f "$STAMP" || "$ROOT/pyproject.toml" -nt "$STAMP" ]]; then
    echo "run.sh: installing dependencies into $VENV..."
    if command -v uv >/dev/null; then
      VIRTUAL_ENV="$VENV" uv pip install -q "$ROOT"
    else
      "$PY" -m pip install -q "$ROOT"
    fi
    rm -rf "$ROOT/build" "$ROOT"/*.egg-info
    touch "$STAMP"
  fi
fi

exec "$PY" -m gunicorn --timeout 1200 --graceful-timeout 60 \
  --workers "$WEB_WORKERS" --bind "$HOST:$PORT" app:app
