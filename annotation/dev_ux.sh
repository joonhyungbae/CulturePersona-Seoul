#!/usr/bin/env bash
# Annotation-screen UX dev mode — opens a screen over tailscale that updates as soon as the code changes (HMR).
#
#   ./dev_ux.sh          start it in tmux session anno-dev (if already running, only print the address)
#   ./dev_ux.sh stop     stop the session
#   ./dev_ux.sh fg       in this terminal, without tmux
#
# Storage is .dev_store/ (sandbox, gitignore). **Unrelated to the real spreadsheet and to outputs/anno/ in the main repository.**
# DEV bar at the bottom of the screen: switch role · jump to an item · restart this item from stage 1 · auto-score (open adjudication) · wipe all.
# Served over https — opened over http from outside, the browser blocks unsealing (Web Crypto) and stage 2 does not appear.
# The certificate is self-signed, so the browser warns once (Advanced → continue).
# Run `sudo tailscale set --operator=$USER` once to get a real tailscale certificate and open without the warning.
set -euo pipefail
cd "$(dirname "$0")"
SESSION=anno-dev
PORT=${ANNO_PORT:-5183}

if [ "${1:-}" = stop ]; then tmux kill-session -t "$SESSION" 2>/dev/null && echo "stopped"; exit 0; fi

[ -d node_modules ] || npm install
[ -f data_private/items_full.json ] || { echo "data_private/items_full.json is missing (built by scripts/anno_build_sample.py in the main repository)"; exit 1; }
[ -f public/data/items.json ] || npm run build:data

STORE=$PWD/.dev_store
mkdir -p "$STORE"
[ -f "$STORE/raters.json" ] ||
  printf '{"R1": "DEVR1", "R2": "DEVR2", "ADJ": "DEVADJ", "OWNER": "DEVOWNER"}\n' > "$STORE/raters.json"

TS_IP=$(tailscale ip -4 2>/dev/null | head -1 || true)
TS_DNS=$(tailscale status --json 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))" 2>/dev/null || true)
HOST=${ANNO_HOST:-${TS_IP:-127.0.0.1}}

CERT=$PWD/.dev_cert
mkdir -p "$CERT"
if [ -n "$TS_DNS" ] && [ ! -f "$CERT/ts.crt" ] && tailscale cert --cert-file "$CERT/ts.crt" --key-file "$CERT/ts.key" "$TS_DNS" >/dev/null 2>&1; then
  :
fi
if [ -f "$CERT/ts.crt" ]; then
  CRT=$CERT/ts.crt; KEY=$CERT/ts.key
else
  CRT=$CERT/self.crt; KEY=$CERT/self.key
  [ -f "$CRT" ] || openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=${TS_DNS:-localhost}" \
    -addext "subjectAltName=DNS:${TS_DNS:-localhost},DNS:localhost,IP:${HOST},IP:127.0.0.1" \
    -keyout "$KEY" -out "$CRT" >/dev/null 2>&1
fi

URL="https://${TS_DNS:-$HOST}:$PORT/"
[ "$HOST" = 127.0.0.1 ] && URL="https://127.0.0.1:$PORT/"
CMD="ANNO_DEV=1 ANNO_STORE='$STORE' ANNO_HOST='$HOST' ANNO_PORT='$PORT' ANNO_HTTPS_CERT='$CRT' ANNO_HTTPS_KEY='$KEY' npx vite"

echo "Dev mode: $URL"
echo "  codes  R1=DEVR1  R2=DEVR2  ADJ=DEVADJ  OWNER=DEVOWNER   (store: .dev_store/)"
if [ "${1:-}" = fg ]; then exec bash -c "$CMD"; fi
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "  already running (tmux attach -t $SESSION)"; exit 0; fi
tmux new-session -d -s "$SESSION" "bash -c \"$CMD\""
echo "  running in tmux session $SESSION — logs: tmux attach -t $SESSION · stop: ./dev_ux.sh stop"
