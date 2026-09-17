#!/usr/bin/env bash
# Install the user units. No sudo: systemctl --user plus loginctl enable-linger
# gives reboot survival, Restart=always, journald and cgroup limits as the
# login user. `james` has no passwordless sudo, so this is the only path.
set -uo pipefail
ROOT="$HOME/fruit"
mkdir -p "$HOME/.config/systemd/user" "$ROOT/env"

[ -f "$ROOT/env/model.env" ] || cat > "$ROOT/env/model.env" <<ENV
MODEL=$ROOT/models/gguf/fruit/fruit-30b-a3b-Q5_K_M.gguf
ALIAS=fruit
PORT=8000
HOST=127.0.0.1
SLOTS=8
CTX=65536
THREADS=16
BIN=$ROOT/llama.cpp/build/bin/llama-server
ENV
[ -f "$ROOT/env/gateway.env" ] || cat > "$ROOT/env/gateway.env" <<ENV
FRUIT_UPSTREAM=http://127.0.0.1:8000
FRUIT_MODEL=fruit
FRUIT_SLOTS=8
FRUIT_RATE_PER_MIN=6
ENV

cp "$ROOT/app/serve/systemd/fruit-model.service" \
   "$ROOT/app/serve/systemd/fruit-gateway.service" \
   "$HOME/.config/systemd/user/"

# Linger is what makes the units survive logout and reboot. It is normally
# permitted for one's own user without a password; if polkit refuses, fall back
# to an @reboot crontab (also sudo-free).
if ! loginctl enable-linger "$USER" 2>/dev/null; then
  echo "WARNING: loginctl enable-linger failed. Units will not survive reboot."
  echo "Fallback: crontab -e  ->  @reboot systemctl --user start fruit-model fruit-gateway"
fi
systemctl --user daemon-reload
echo "installed. enable with:  systemctl --user enable --now fruit-model fruit-gateway"
loginctl show-user "$USER" -p Linger 2>/dev/null || true
