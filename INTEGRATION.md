# Putting the RAG in front of an LLM

Two front-ends onto the same server (`gitchain-lite /api/v1/query`). Pick by client:

| Client | Use | Retrieve-first? | LLM |
|---|---|---|---|
| **An MCP client** | `gitchain-mcp.mjs` (MCP tool) | via instruction (model-decided) | the client's model |
| **Any OpenAI-compatible UI** | `gitchain-rag-proxy.mjs` (grounding gateway) | **hard, every turn** | **any** |

Mechanism = RAG. Product value = *grounded, cited, sovereign* answers.

---

## A) An MCP client — MCP server

Add to your MCP client's server config (a stdio MCP server):
```json
{
  "mcpServers": {
    "gitchain": {
      "command": "node",
      "args": ["/absolute/path/to/gitchain-mcp.mjs"],
      "env": { "GITCHAIN_API": "http://127.0.0.1:7420" }
    }
  }
}
```
Then add a profile/project instruction: *"Before answering questions about my documents/products,
always call `gitchain_search` first and ground the answer in the returned passages with [n] citations."*
Restart the client.

Tools: `gitchain_search {query, container?, k?}`, `gitchain_list_containers`.

---

## B) Any other UI / any LLM — the grounding gateway

An OpenAI-compatible `/v1/chat/completions` proxy: every turn is retrieved + grounded, then forwarded.
```bash
LLM_URL=http://127.0.0.1:11434/v1/chat/completions LLM_MODEL=local-model \
GITCHAIN_API=http://127.0.0.1:7420 PORT=7008 \
node gitchain-rag-proxy.mjs
```
Point any chat UI at `http://127.0.0.1:7008/v1`. Response header `X-GitChain-Grounded: <n>` reports how
many passages were injected.

### Swap the LLM
Set `LLM_URL` to any OpenAI-compatible `/v1/chat/completions` endpoint (a local model, or a gateway that
fronts a cloud model). `LLM_KEY` sets a bearer token if the endpoint needs one.

### Query rewrite (optional)
Set `RAG_REWRITE=1` to have the LLM first turn a chatty user turn into a focused retrieval query
(entities/product names/numbers kept, chit-chat dropped) before searching. Costs one extra LLM call;
the query actually used is echoed in the `X-GitChain-Query` response header. Off by default.

### Env reference
`PORT` (7008) · `GITCHAIN_API` (7420) · `GITCHAIN_TOKEN` · `RAG_CONTAINER` (pin one; omit = search all) ·
`RAG_K` (8) · `LLM_URL` · `LLM_MODEL` · `LLM_KEY` · `RAG_REWRITE` (0/1).

---

Both require the server (with `/api/v1/query` + local embeddings) running on `:7420`.
