#!/usr/bin/env bash
# 30-second demo: three tickets, every agent step printed.
#   T-008  routine refund question   -> answered from the knowledge base
#   T-006  password reset in Spanish -> answered in Spanish
#   T-005  account takeover          -> escalated to a human
# Usage: ./demo.sh [A|B]   (which model from .env; default A)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=python
if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; fi
if [ -x .venv/Scripts/python.exe ]; then PYTHON=.venv/Scripts/python.exe; fi

mkdir -p results
"$PYTHON" -m app.cli --tickets T-008 T-006 T-005 --model "${1:-A}" | tee results/demo-output.txt
