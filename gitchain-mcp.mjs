#!/usr/bin/env node
// gitchain-mcp — a zero-dependency MCP (stdio) server that exposes the sovereign GitChain-Lite
// RAG to the MCP client (or any MCP client). It is a thin bridge: MCP tool calls in → local
// gitchain-lite HTTP (/api/v1/query, /git/repos) → grounded context out. No cloud, no DB.
//
// Flow it enables in the MCP client:
//   user prompt → the model calls `gitchain_search` (nudged by a "retrieve first" instruction)
//   → this server embeds the query locally + cosine-ranks a container's .brain vectors
//   → returns top-k chunks WITH [n] citations → the model grounds its answer → shown in the client.
//
// The embeddings + ranking are fully local (gitchain-lite → embeddinggemma). The generator is
// whatever the MCP client uses (the model). Set `answer:true` mentally-equivalent flows if you want
// the local model to draft instead — but we return context and let the model write.
//
// Protocol: MCP over stdio = newline-delimited JSON-RPC 2.0. We implement initialize, tools/list,
// tools/call, ping. Env: GITCHAIN_API (default http://127.0.0.1:7420), GITCHAIN_TOKEN (optional).

const API = process.env.GITCHAIN_API || "http://127.0.0.1:7420";
const TOKEN = process.env.GITCHAIN_TOKEN || "";
const PROTOCOL = "2024-11-05";

const headers = () => TOKEN ? { "Content-Type": "application/json", Authorization: "Bearer " + TOKEN }
                            : { "Content-Type": "application/json" };

async function api(path, opts = {}) {
  const r = await fetch(API + path, { headers: headers(), ...opts });
  if (!r.ok) throw new Error(`gitchain ${path} → HTTP ${r.status}`);
  return r.json();
}

// GET /git/repos → normalized [{id, count?}]
async function listContainers() {
  const raw = await api("/git/repos");
  const arr = Array.isArray(raw) ? raw : (raw.repos || raw.data || raw.containers || []);
  return arr.map((c) => ({
    id: c.path_with_namespace || (Array.isArray(c.segs) ? c.segs.join("/") : c.id) || String(c),
    count: c.count,
  })).filter((c) => c.id);
}

// One container query. answer:false → we return context, the model composes the reply.
async function queryOne(container, q, k) {
  return api("/api/v1/query", { method: "POST", body: JSON.stringify({ container, q, k, answer: false }) });
}

// gitchain_search: if `container` omitted, fan out over all containers and merge by score.
async function search({ query, container, k = 8 }) {
  const kk = Math.min(Math.max(1, Number(k) || 8), 30);
  let matches = [], searched = [];
  if (container) {
    const out = await queryOne(container, query, kk);
    if (out.error) throw new Error(out.error);
    matches = (out.matches || []).map((m) => ({ ...m, container }));
    searched = [container];
  } else {
    const containers = (await listContainers()).slice(0, 20);
    const results = await Promise.allSettled(containers.map((c) => queryOne(c.id, query, kk)));
    results.forEach((r, i) => {
      if (r.status === "fulfilled" && Array.isArray(r.value.matches))
        matches.push(...r.value.matches.map((m) => ({ ...m, container: containers[i].id })));
    });
    searched = containers.map((c) => c.id);
    matches.sort((a, b) => (b.score || 0) - (a.score || 0));
    matches = matches.slice(0, kk);
  }
  if (!matches.length)
    return `No relevant context found in ${searched.length} container(s) for: "${query}".\n` +
           `Tell the user nothing matched — do NOT fabricate an answer from outside the knowledge base.`;
  // Formatted for grounding: numbered blocks the model cites as [n].
  const blocks = matches.map((m, i) =>
    `[${i + 1}] (container: ${m.container}${m.source ? `, source: ${m.source}` : ""}, score ${m.score})\n${(m.text || "").trim()}`
  ).join("\n\n");
  return `Retrieved ${matches.length} grounded passage(s) from the GitChain knowledge base. ` +
         `Answer the user's question using ONLY these passages and cite them as [n]:\n\n${blocks}`;
}

const TOOLS = [
  {
    name: "gitchain_search",
    description:
      "Search the user's sovereign GitChain knowledge base (local RAG over their own documents/products) " +
      "and return grounded passages with [n] citations. ALWAYS call this before answering questions about " +
      "the user's data, products, datasheets, or documents. Leave `container` empty to search everything.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "The search query (usually the user's question, or a focused rephrasing)." },
        container: { type: "string", description: "Optional container id (e.g. 'acme/knowledge/handbook'). Omit to search all." },
        k: { type: "integer", description: "How many passages to return (1–30, default 8).", default: 8 },
      },
      required: ["query"],
    },
  },
  {
    name: "gitchain_list_containers",
    description: "List the available containers (knowledge bases) in the local GitChain instance.",
    inputSchema: { type: "object", properties: {} },
  },
];

// ---- JSON-RPC dispatch ---------------------------------------------------------------
async function handle(msg) {
  const { id, method, params } = msg;
  if (method === "initialize")
    return { id, result: { protocolVersion: params?.protocolVersion || PROTOCOL,
      capabilities: { tools: {} }, serverInfo: { name: "gitchain-mcp", version: "0.1.0" } } };
  if (method === "ping") return { id, result: {} };
  if (method === "tools/list") return { id, result: { tools: TOOLS } };
  if (method === "tools/call") {
    const { name, arguments: args = {} } = params || {};
    try {
      const text = name === "gitchain_search" ? await search(args)
        : name === "gitchain_list_containers" ? JSON.stringify(await listContainers(), null, 2)
        : (() => { throw new Error(`unknown tool: ${name}`); })();
      return { id, result: { content: [{ type: "text", text }] } };
    } catch (e) {
      return { id, result: { content: [{ type: "text", text: `Error: ${String((e && e.message) || e)}` }], isError: true } };
    }
  }
  if (id === undefined) return null;                       // a notification (e.g. notifications/initialized)
  return { id, error: { code: -32601, message: `method not found: ${method}` } };
}

// ---- stdio transport: newline-delimited JSON-RPC -------------------------------------
let buf = "", pending = 0, ended = false;
const maybeExit = () => { if (ended && pending === 0) process.exit(0); };  // never exit mid-request
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  buf += chunk;
  let nl;
  while ((nl = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, nl).trim();
    buf = buf.slice(nl + 1);
    if (!line) continue;
    let msg; try { msg = JSON.parse(line); } catch { continue; }
    pending++;
    handle(msg)
      .catch((e) => ({ id: msg.id, error: { code: -32603, message: String(e) } }))
      .then((resp) => { if (resp) process.stdout.write(JSON.stringify({ jsonrpc: "2.0", ...resp }) + "\n"); })
      .finally(() => { pending--; maybeExit(); });
  }
});
process.stdin.on("end", () => { ended = true; maybeExit(); });
