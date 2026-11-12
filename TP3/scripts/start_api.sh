#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
case "${1:-films}" in
  films) exec .venv/bin/python -m uvicorn api.films_api:app --host 127.0.0.1 --port 8001 ;;
  accidents) exec .venv/bin/python -m uvicorn api.accidents_api:app --host 127.0.0.1 --port 8002 ;;
  *) echo 'Usage: bash scripts/start_api.sh films|accidents' >&2; exit 2 ;;
esac
