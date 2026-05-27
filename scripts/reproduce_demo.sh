#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
VENV_DIR="${VENV_DIR:-.venv}"

"${PYTHON_BIN}" --version

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if [ -x "${VENV_DIR}/Scripts/python.exe" ]; then
  VENV_PYTHON="${VENV_DIR}/Scripts/python.exe"
else
  VENV_PYTHON="${VENV_DIR}/bin/python"
fi

"${VENV_PYTHON}" -m pip install -r requirements.txt
"${VENV_PYTHON}" scripts/run_transformer_roofline.py
"${VENV_PYTHON}" scripts/plot_roofline.py
"${VENV_PYTHON}" -m pytest
