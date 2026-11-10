#!/bin/bash
# Double-click to start the CO2 backfill predictor. The first start installs it (one time only).
cd "$(dirname "$0")/app" || exit 1
if [ ! -f .venv/installed.ok ]; then
  PY=""
  for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then PY=$c; break; fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.11 or newer was not found. Please install Python 3.12 from https://www.python.org/downloads/"
    echo "and then double-click this file again."
    read -r -p "Press Enter to close."; exit 1
  fi
  echo "First start: installing the app. This happens only once and takes 5-10 minutes."
  if "$PY" -m venv .venv && { .venv/bin/python -m pip install -r requirements.txt --timeout 30 ||
       .venv/bin/python -m pip install -r requirements.txt --timeout 30 -i https://pypi.tuna.tsinghua.edu.cn/simple; }; then
    touch .venv/installed.ok
  else
    rm -rf .venv
    echo "Installation failed. Please check the internet connection and try again."
    read -r -p "Press Enter to close."; exit 1
  fi
fi
echo "The app will open in your web browser in a few seconds (http://localhost:8501)."
echo "The first page takes about 1 minute while the models train. Close this window to stop the app."
(sleep 8; open http://localhost:8501 2>/dev/null || xdg-open http://localhost:8501) &
.venv/bin/python -m streamlit run app.py --server.port 8501
