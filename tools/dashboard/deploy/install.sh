#!/usr/bin/env bash
# Install / update the AIUQ dashboard as a systemd --user service on the Autosubmit host.
# Idempotent: safe to re-run after updating the code.
#
#   bash deploy/install.sh            # env + vendor JS + service (+ enable-linger)
#   SKIP_LINGER=1 bash deploy/install.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-aiuq-dash}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
UNIT_DIR="$HOME/.config/systemd/user"
VENDOR="$APP_DIR/aiuq_dash/static/vendor"

echo ">> app dir: $APP_DIR"
# shellcheck disable=SC1090
source "$CONDA_SH"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">> conda env $ENV_NAME exists, updating"
  conda env update -n "$ENV_NAME" -f "$APP_DIR/environment.yml" --prune
else
  echo ">> creating conda env $ENV_NAME (base is not touched)"
  conda env create -n "$ENV_NAME" -f "$APP_DIR/environment.yml"
fi
ENV_PYTHON="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
echo ">> python: $ENV_PYTHON"

echo ">> vendoring front-end libraries (served locally, CDN only as fallback)"
mkdir -p "$VENDOR"
fetch() { [ -s "$VENDOR/$2" ] || curl -fsSL --retry 2 -o "$VENDOR/$2" "$1"; }
fetch "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js" plotly.min.js
fetch "https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/dist/topojson-client.min.js" topojson-client.min.js
fetch "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-110m.json" land-110m.json

echo ">> installing systemd user unit"
mkdir -p "$UNIT_DIR" "$HOME/.config"
sed -e "s#@APP_DIR@#$APP_DIR#g" -e "s#@ENV_PYTHON@#$ENV_PYTHON#g" \
  "$APP_DIR/deploy/aiuq-dashboard.service" > "$UNIT_DIR/aiuq-dashboard.service"
ENV_FILE="$HOME/.config/aiuq-dashboard.env"
[ -f "$ENV_FILE" ] || cp "$APP_DIR/deploy/aiuq-dashboard.env" "$ENV_FILE"
# Site settings have no defaults: name the ones still to fill in (the dashboard starts anyway and says so too).
MISSING=""
for k in AIUQ_REMOTE_HOST AIUQ_REMOTE_BASE AIUQ_LAUNCH_PLATFORM AIUQ_SUPPORT_FOLDER AIUQ_REMOTE_DELETE_ROOTS; do
  grep -Eq "^[[:space:]]*$k=[^[:space:]#]" "$ENV_FILE" || MISSING="$MISSING $k"
done

if [ -z "${SKIP_LINGER:-}" ] && [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
  echo ">> enabling linger so the user service survives logout"
  loginctl enable-linger "$USER"
fi

systemctl --user daemon-reload
systemctl --user enable aiuq-dashboard.service
systemctl --user restart aiuq-dashboard.service
sleep 3
systemctl --user --no-pager status aiuq-dashboard.service | head -12
if [ -n "$MISSING" ]; then
  echo ">> TO DO: set$MISSING in $ENV_FILE, then: systemctl --user restart aiuq-dashboard"
fi
echo ">> done. Health: curl -s --unix-socket ~/.aiuq-dashboard/dash.sock http://localhost/api/health"
echo ">> From your laptop: ssh -N -L 8765:$HOME/.aiuq-dashboard/dash.sock $(hostname -s)  then open http://localhost:8765"
