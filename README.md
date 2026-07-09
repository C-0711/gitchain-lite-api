# gitchain-lite-api

A **sovereign git-native server** — a container store with local ingest, grounded semantic query, and
two ways to put it in front of an LLM. **Zero dependencies** (only `git` + Node ≥18, plus Python for the
document reader). No database, no cloud, no egress.

This is the standalone server. It runs on its own; any desktop or web UI can consume it over HTTP.

## Components
```
gitchain-lite.mjs        Git server (clone/fetch/push over Smart-HTTP, push-to-create) + ingest-on-push
                         + local /api/v1/query (embed → cosine over a container's .brain vectors → optional
                         local answer). Also compressed-store decode-on-open.
ingest-worker.mjs        Builds/atomizes a container on push: reads PDFs, embeds them.
gitchain-mcp.mjs         MCP (stdio) server → expose the RAG to any MCP client as a tool.
gitchain-rag-proxy.mjs   OpenAI-compatible /v1/chat/completions grounding gateway → RAG in front of ANY
                         UI/LLM (hard retrieve-first, LLM-agnostic via LLM_URL).
toolkit/                 build/atomize/quantize + the document reader + the pure-numpy vector codec.
gitchain-up.sh           One-command launcher that wires the local models.
INTEGRATION.md           Full setup for the MCP + proxy integration paths.
```

## Quickstart
```bash
# 1. git server + query engine (serves http://127.0.0.1:7420)
REPO_BASE_PATH=./repos ./gitchain-up.sh
# or minimal:  PORT=7420 REPO_BASE_PATH=./repos node gitchain-lite.mjs

# 2. local models (any OpenAI-compatible server), the ingest + query use these endpoints:
export EMBED_URL=http://127.0.0.1:11434/v1/embeddings      # a 768-d text embedding model
export CHAT_URL=http://127.0.0.1:11434/v1/chat/completions # optional, for /api/v1/query answer:true

# 3. Python for the document reader (ingest):
pip3 install numpy pymupdf pypdf
```

Create a container by pushing a git repo (files under `docs/`, optional `.gitchain/ingest.json` to build):
```bash
git remote add gc http://127.0.0.1:7420/git/<tenant>/<project>/<id>.git
git push gc main         # first push creates the container; with an ingest config it builds
```
Query it:
```bash
curl -s http://127.0.0.1:7420/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"container":"<tenant>/<project>/<id>","q":"question?","k":5,"answer":false}'
```

## Endpoints
- `GET /api/v1/health` · `GET /git/repos` · git push/clone at `/git/<t>/<p>/<id>.git`
- `POST /api/v1/query {container, q, k, answer}` — local embed → rank the container's vectors → optional local answer

## Env
`PORT` (7420) · `REPO_BASE_PATH` · `EMBED_URL` · `CHAT_URL` · `INGEST_CMD` / `INGEST_CMD_JSON` ·
`TOOLKIT_DIR` · `PYTHON_BIN`. All extras are opt-in — unset, it's a plain zero-config git server.

See `INTEGRATION.md` for wiring the RAG into an MCP client or any chat UI (grounding gateway).
