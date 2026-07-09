#!/usr/bin/env bash
# Sovereign GitChain backend: git-native store + local ingest + local query,
# wired to your local models (Ollama by default). No database, no cloud, no egress.
set -e
cd "$(dirname "$0")"
export REPO_BASE_PATH="${REPO_BASE_PATH:-$HOME/gitchain-repos}"
export PORT="${PORT:-7420}"
export TOOLKIT_DIR="$(pwd)/toolkit"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export EMBED_URL="${EMBED_URL:-http://127.0.0.1:11434/v1/embeddings}"   # Ollama serves this
export CHAT_URL="${CHAT_URL:-http://127.0.0.1:11434/v1/chat/completions}"
export INGEST_CMD="node $(pwd)/ingest-worker.mjs --ingest"
echo "GitChain sovereign backend"
echo "  store   $REPO_BASE_PATH"
echo "  server  http://127.0.0.1:$PORT   (point the app: GITCHAIN_API=http://127.0.0.1:$PORT)"
echo "  embed   $EMBED_URL"
echo "  chat    $CHAT_URL"
exec node gitchain-lite.mjs
