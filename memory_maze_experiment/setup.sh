#!/usr/bin/env bash
# One-command setup for the Memory Maze territorial experiment.
#
# Creates a project-local .venv/ inside this folder and installs dependencies.
# Run once:
#     bash setup.sh
# Then to use the env in future sessions:
#     source .venv/bin/activate
#
# To force a specific Python:
#     PYTHON=python3.11 bash setup.sh

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- Pick a Python, preferring 3.10/3.11/3.12 over 3.9 ---
pick_python() {
    if [ -n "${PYTHON:-}" ]; then
        echo "$PYTHON"
        return
    fi
    for cand in python3.12 python3.11 python3.10 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            echo "$cand"
            return
        fi
    done
    echo ""
}

PY="$(pick_python)"
if [ -z "$PY" ] || ! command -v "$PY" >/dev/null 2>&1; then
    echo "error: no usable Python found. Install Python 3.10+ (e.g. 'brew install python@3.11') and retry."
    exit 1
fi

PYVER=$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using $PY ($PYVER) at $(command -v $PY)"

# Warn on Python 3.9 — mujoco 3.3+ has no py39 macOS wheels; 3.2.x still works.
case "$PYVER" in
    3.9)
        echo "⚠️  Python 3.9 is end-of-life soon and many packages are dropping support."
        echo "    requirements.txt pins mujoco<3.3 so this still works, but 3.11+ is smoother."
        echo "    To upgrade:  brew install python@3.11  and rerun with PYTHON=python3.11 bash setup.sh"
        ;;
    3.8|3.7|3.6)
        echo "error: Python $PYVER is too old. Please install Python 3.10+."
        exit 1
        ;;
esac

# --- Create venv ---
if [ ! -d .venv ]; then
    echo "Creating .venv/ ..."
    "$PY" -m venv .venv
else
    echo ".venv/ already exists, reusing (delete it and rerun to start fresh)"
fi

# shellcheck source=/dev/null
source .venv/bin/activate

# --- Install dependencies ---
echo "Upgrading pip ..."
python -m pip install --upgrade pip wheel >/dev/null

echo "Installing requirements ..."
python -m pip install -r requirements.txt

# --- Smoke test import ---
echo
echo "=== Smoke test: import memory_maze ==="
if python -c "import os; os.environ.setdefault('MUJOCO_GL','glfw'); import memory_maze; print('memory_maze imported OK')"; then
    echo
    echo "✅ Setup complete. Next:"
    echo "    source .venv/bin/activate"
    echo "    python env_probe.py"
    echo "    python experiment.py --size 9x9 --episodes 20"
else
    echo
    echo "⚠️  memory_maze imported with errors (usually a MuJoCo rendering driver issue)."
    echo "    On macOS desktop try:  export MUJOCO_GL=glfw"
    echo "    On headless Linux:     export MUJOCO_GL=egl"
    echo "    CPU-only Linux:        export MUJOCO_GL=osmesa (needs: apt install libosmesa6-dev)"
fi
