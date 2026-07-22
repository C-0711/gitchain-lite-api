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

## Quickstart — in 60 Sekunden zum ersten zitierten Treffer

Voraussetzungen: nur `git` + Node ≥ 18. **Keine Modelle, kein Python, keine Datenbank.**

```bash
git clone https://github.com/C-0711/gitchain-lite-api.git && cd gitchain-lite-api
npm run doctor    # Selbsttest: was ist da, was fehlt, was ist optional
npm run demo      # Wegwerf-Server + Beispiel-Container + Frage mit zitierten Treffern
npm start         # eigener Server auf http://127.0.0.1:7420
```

Die Suche funktioniert **ohne jede Konfiguration** lexikalisch (BM25). Jede weitere
Faehigkeit ist ein opt-in — das System degradiert sanft statt zu brechen:

| Stufe | Was du bekommst | Was du brauchst |
|---|---|---|
| 0 (Default) | Git-Host + BM25-Suche, jeder Treffer **seitenzitiert** (Glyph-Provenienz: Datei + Seite + y bei PDFs, Zeilenbereich bei Text) | nichts (PDFs: `pip3 install numpy pymupdf`) |
| 1 | dense+BM25 fusionierte Suche (RRF) | `EMBED_URL` (OpenAI-kompatibler Embeddings-Server) |
| 2 | zitierte **Antworten** statt nur Treffer | `CHAT_URL` (OpenAI-kompatibler Chat-Server) |
| 3 | PDF-Ingest per `git push` | `pip3 install numpy pymupdf` + `INGEST_CMD` |

Konfiguration: `cp .env.example .env`, anpassen, exportieren — oder Variablen direkt setzen.
`npm run doctor` sagt dir jederzeit, welche Stufe aktiv ist.

### Claims statt Prosa (Formulardokumente) + messen

Fuer Formulare (z.B. Lohnsteuerbescheinigungen) sind **typisierte Claims** die besseren Atome
als Prosa-Fenster — gemessen ~3x recall@1 gegenueber dem besten Prosa-Stack:

```bash
npm run claims -- meine-pdfs/ /tmp/claims-container      # deterministische Extraktion (LStB-Profil)
npm run eval   -- /tmp/claims-container examples/goldset-lstb.jsonl   # recall@1/@3 + diagnostizierte Fails
```

Jeder Claim traegt eCode, Wert, Quelle, Seite, y und Zeugenzahl. `gitchain-eval.py` trennt
Fails in "kein Claim" (Extraktionsluecke) vs "Rank>3" (Retrieval) — so wird jeder Fehlschlag
ein adressierbares Ticket statt eines Gefuehls. Referenzlauf (54 GT-Fragen, BM25-only,
Single-Witness): Orakel 72%, recall@3 56%.

Rerank (Level 5): Request-Body `"rerank": true` laesst das CHAT_URL-Modell die Top-20 neu
ordnen (8s-Timeout, Fallback = Fusionsreihenfolge). Prefix (Level 4): `"prefix":"auto"` in
ingest.json stellt jeden Atom-Text den erkannten Dokument-Kopf voran (Firmenname).

Quantisierte Container tragen ihre GEMESSENE Karte in `.brain/tq_report.json`
(TurboQuant/PolarQuant, arXiv 2504.19874): Kompression, Cosinus-Treue, Score-Recall@10.

### Eigene Inhalte — zwei Wege

**Ohne Modelle** (Texte/Markdown): Container-Worktree bauen und pushen — sofort suchbar:
```bash
node demo/build-lite.mjs meine-docs/ /tmp/mein-container
cd /tmp/mein-container && git init -b main . && git add -A && git commit -m docs
git push http://127.0.0.1:7420/git/acme/wissen/handbuch.git main   # push-to-create
curl -s http://127.0.0.1:7420/api/v1/query -H 'Content-Type: application/json' \
  -d '{"container":"acme/wissen/handbuch","q":"...?","k":5}'
```

**Mit Ingest** (PDFs, Stufe 3): Repo mit `docs/*.pdf` + `.gitchain/ingest.json`
(`{"mode":"build","source":"docs","model":"embeddinggemma","dims":768,"level":"b2"}`)
pushen — der Ingest liest, atomisiert, embeddet und committet die Artefakte zurueck.

## Endpoints
- `GET /api/v1/health` · `GET /git/repos` · git push/clone at `/git/<t>/<p>/<id>.git`
- `POST /api/v1/query {container, q, k, answer}` — BM25 immer; dense+RRF-Fusion wenn `EMBED_URL` gesetzt; `answer:true` + `CHAT_URL` -> zitierte Antwort. Antwort enthaelt `mode` (aktiver Suchpfad) und bei Fehlern einen `hint` mit dem naechsten Schritt.

## Env
`PORT` (7420) · `REPO_BASE_PATH` · `EMBED_URL` · `CHAT_URL` · `INGEST_CMD` / `INGEST_CMD_JSON` ·
`TOOLKIT_DIR` · `PYTHON_BIN`. All extras are opt-in — unset, it's a plain zero-config git server.

See `INTEGRATION.md` for wiring the RAG into an MCP client or any chat UI (grounding gateway).
