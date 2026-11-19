#!/bin/sh
# One image, three ways to run it (task T-602).
#
#   api        uvicorn only, on 8000              - what docker-compose runs
#   dashboard  streamlit only, on 8501            - what docker-compose runs
#   both       uvicorn in the background + streamlit in the foreground, one container
#
# `both` exists for Hugging Face Spaces, which runs a single container and routes one port.
# There the dashboard is the public surface and the API stays internal on 127.0.0.1, which is
# also why the dashboard's default API URL is a loopback address in the image.
set -eu

# The env var wins over the CMD argument on purpose: Hugging Face runs the image as-is and
# can only set variables, so `LEADSCORE_MODE=both` in the Space settings is enough and no
# edited copy of the Dockerfile has to be kept in sync.
MODE="${LEADSCORE_MODE:-${1:-api}}"
API_PORT="${LEADSCORE_API_PORT:-8000}"
DASHBOARD_PORT="${PORT:-${LEADSCORE_DASHBOARD_PORT:-8501}}"

start_api() {
    exec python -m uvicorn leadscore.api.main:app \
        --host "${LEADSCORE_API_HOST:-0.0.0.0}" --port "${API_PORT}"
}

start_dashboard() {
    exec python -m streamlit run dashboard/app.py \
        --server.port "${DASHBOARD_PORT}" \
        --server.address 0.0.0.0 \
        --server.headless true \
        --browser.gatherUsageStats false
}

case "${MODE}" in
    api)
        echo "starting the API on ${API_PORT}"
        start_api
        ;;
    dashboard)
        echo "starting the dashboard on ${DASHBOARD_PORT}"
        start_dashboard
        ;;
    both)
        echo "starting the API on 127.0.0.1:${API_PORT} and the dashboard on ${DASHBOARD_PORT}"
        python -m uvicorn leadscore.api.main:app --host 127.0.0.1 --port "${API_PORT}" &
        API_PID=$!

        # If the API dies, the dashboard is useless, so take the container down with it
        # rather than serving three pages that all say "service not responding".
        trap 'kill "${API_PID}" 2>/dev/null || true' INT TERM EXIT

        # Wait for the model to load before the dashboard's first health check, but never
        # forever: a container that hangs on start is harder to diagnose than one that exits.
        waited=0
        until python -c "
import sys, urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:${API_PORT}/health', timeout=2)
except Exception:
    sys.exit(1)
" 2>/dev/null; do
            waited=$((waited + 2))
            if [ "${waited}" -ge 60 ]; then
                echo "the API did not become healthy within 60s; exiting" >&2
                exit 1
            fi
            sleep 2
        done
        echo "API healthy after ${waited}s"
        start_dashboard
        ;;
    *)
        echo "unknown mode '${MODE}'. Use: api | dashboard | both" >&2
        exit 2
        ;;
esac
