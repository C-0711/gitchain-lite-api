#!/usr/bin/env bash
# 60-Sekunden-Demo — braucht NUR git + Node >= 18. Keine Modelle, kein Python.
# Startet einen Wegwerf-Server, baut einen modell-freien Container, pusht ihn,
# stellt eine Frage und zeigt die zitierten Treffer.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${DEMO_PORT:-7429}
BASE=$(mktemp -d) WORK=$(mktemp -d)

REPO_BASE_PATH="$BASE" PORT="$PORT" EMBED_URL="" CHAT_URL="" node "$HERE/gitchain-lite.mjs" >/dev/null 2>&1 &
SRV=$!
disown "$SRV" 2>/dev/null || true
trap 'kill $SRV 2>/dev/null; rm -rf "$BASE" "$WORK"' EXIT
for i in $(seq 1 20); do curl -sf "http://127.0.0.1:$PORT/api/v1/health" >/dev/null 2>&1 && break; sleep 0.3; done

node "$HERE/demo/build-lite.mjs" "$HERE/demo/docs" "$WORK"
cd "$WORK"
git init -q -b main . && git add -A
git -c user.email=demo@local -c user.name=demo commit -qm "Demo-Wissen"
git push -q "http://127.0.0.1:$PORT/git/demo/wissen/handbuch.git" main
echo
echo "Container demo/wissen/handbuch gepusht (push-to-create). Frage:"
echo '  "Welcher Port ist der Standard von gitchain-lite?"'
echo
curl -s "http://127.0.0.1:$PORT/api/v1/query" -H 'Content-Type: application/json' \
  -d '{"container":"demo/wissen/handbuch","q":"Welcher Port ist der Standard von gitchain-lite?","k":2}' \
  | node -e 'let d="";process.stdin.on("data",(c)=>d+=c).on("end",()=>{const j=JSON.parse(d);console.log("  Modus:",j.mode);for(const m of j.matches)console.log(`  [${m.source}] ${m.text.slice(0,110)}`)})'
echo
echo "✓ Demo ok — dasselbe mit eigenen Docs: docs/ + Push, Ingest baut den Rest (siehe README)."
