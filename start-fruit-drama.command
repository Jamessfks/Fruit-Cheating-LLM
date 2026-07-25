#!/bin/bash
# Double-click this file to run the Fruit Drama Story Builder.
# It (1) starts the AI model server on your DGX Spark if it isn't already
# running, (2) waits until it's ready, and (3) opens the interface in your
# browser. Close the window when the interface has opened.
cd "$(dirname "$0")" || exit 1

echo "==================================================="
echo "   Fruit Drama - Story Builder"
echo "==================================================="
echo

echo "[1/3] Checking the model server on your DGX Spark (ssh james)..."
if ssh -o ConnectTimeout=12 james 'tmux has-session -t vs 2>/dev/null'; then
  echo "      Server already running."
else
  echo "      Not running - starting it now..."
  ssh -o ConnectTimeout=12 james 'tmux new-session -d -s vs "bash /home/james/fruit-ft/run_vllm.sh"'
fi
echo

echo "[2/3] Waiting for the server to be ready (a fresh start takes ~2 min)..."
if ssh -o ConnectTimeout=12 james 'for i in $(seq 1 80); do curl -s localhost:8000/v1/models >/dev/null 2>&1 && exit 0; sleep 3; done; exit 1'; then
  echo "      Server is READY."
else
  echo "      WARNING: it did not come up in time."
  echo "      See the log with:  ssh james 'tail -f /home/james/fruit-ft/vllm.log'"
fi
echo

echo "[3/3] Opening the interface in your browser..."
open "interface/index.html"
echo
echo "Done. Pick your cast and click the 'Generate episode' button."
echo "When you're finished, free the Spark's GPU with:"
echo "    ssh james 'tmux kill-session -t vs'"
echo
read -n 1 -s -r -p "Press any key to close this window..."
echo
