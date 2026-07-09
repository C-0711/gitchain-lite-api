#!/usr/bin/env node
// gitchain-rag-proxy — an OpenAI-compatible GROUNDING gateway. It speaks /v1/chat/completions,
// so ANY OpenAI-compatible chat UI (open-webui, LibreChat, your own) can point at it — and it makes
// EVERY turn grounded in the sovereign GitChain knowledge base before forwarding to ANY LLM.
//
// This is the "own pipeline" that the MCP client can't be: hard retrieve-first (not model-decided),
// LLM-agnostic (swap one env), UI-agnostic. Mechanism = RAG; product value = grounded/cited answers.
//
//   chat UI ──▶ this proxy (/v1/chat/completions)
//                 1. take the last user turn as the query
//                 2. ALWAYS gitchain-lite /api/v1/query → top-k passages + [n] citations
//                 3. inject a grounding system message (context + "cite [n]")
//                 4. forward to LLM_URL (a local model │ a gateway │ the provider), stream passed through
//
// Zero-dependency Node. Env:
//   PORT           (default 7008)
//   GITCHAIN_API   local RAG backend            (default http://127.0.0.1:7420)
//   GITCHAIN_TOKEN optional bearer for the RAG backend
//   RAG_CONTAINER  pin one container; omit → search all and merge  (optional)
//   RAG_K          passages to retrieve         (default 8)
//   LLM_URL        downstream OpenAI-compatible  (default a local model :11435)
//   LLM_MODEL      model name to send downstream (default local-model)
//   LLM_KEY        optional bearer for the LLM   (the provider-compat / gateway)
//   LLM_AGENT      optional X-Agent header value (for a gateway that needs it)

import { createServer } from "node:http";

const PORT = Number(process.env.PORT) || 7008;
const GITCHAIN_API = process.env.GITCHAIN_API || "http://127.0.0.1:7420";
const GITCHAIN_TOKEN = process.env.GITCHAIN_TOKEN || "";
const RAG_CONTAINER = process.env.RAG_CONTAINER || "";
const RAG_K = Math.min(Math.max(1, Number(process.env.RAG_K) || 8), 30);
const LLM_URL = process.env.LLM_URL || "http://127.0.0.1:11435/v1/chat/completions";
const LLM_MODEL = process.env.LLM_MODEL || "local-model";
const LLM_KEY = process.env.LLM_KEY || "";
const LLM_AGENT = process.env.LLM_AGENT || "";
const RAG_REWRITE = process.env.RAG_REWRITE === "1" || process.env.RAG_REWRITE === "true";

const ragHeaders = () => GITCHAIN_TOKEN
  ? { "Content-Type": "application/json", Authorization: "Bearer " + GITCHAIN_TOKEN }
  : { "Content-Type": "application/json" };

const llmHeaders = () => {
  const h = { "Content-Type": "application/json" };
  if (LLM_KEY) h.Authorization = "Bearer " + LLM_KEY;
  if (LLM_AGENT) h["X-Agent"] = LLM_AGENT;
  return h;
};

// text of an OpenAI message (string or multimodal parts)
function textOf(msg) {
  if (!msg) return "";
  if (typeof msg.content === "string") return msg.content;
  if (Array.isArray(msg.content)) return msg.content.filter((p) => p.type === "text").map((p) => p.text).join(" ");
  return "";
}

async function ragQuery(container, q) {
  const r = await fetch(GITCHAIN_API + "/api/v1/query", {
    method: "POST", headers: ragHeaders(),
    body: JSON.stringify({ container, q, k: RAG_K, answer: false }),
  });
  if (!r.ok) throw new Error(`RAG HTTP ${r.status}`);
  return r.json();
}

async function listContainers() {
  const r = await fetch(GITCHAIN_API + "/git/repos", { headers: ragHeaders() });
  if (!r.ok) return [];
  const d = await r.json();
  const arr = Array.isArray(d) ? d : (d.repos || d.data || d.containers || []);
  return arr.map((c) => c.path_with_namespace || (Array.isArray(c.segs) ? c.segs.join("/") : c.id)).filter(Boolean);
}

// Retrieve → return {block, n} for the grounding system message, or null if nothing/failed.
async function retrieve(query) {
  let matches = [];
  try {
    if (RAG_CONTAINER) {
      const out = await ragQuery(RAG_CONTAINER, query);
      matches = (out.matches || []).map((m) => ({ ...m, container: RAG_CONTAINER }));
    } else {
      const containers = (await listContainers()).slice(0, 20);
      const rs = await Promise.allSettled(containers.map((c) => ragQuery(c, query)));
      rs.forEach((r, i) => {
        if (r.status === "fulfilled" && Array.isArray(r.value.matches))
          matches.push(...r.value.matches.map((m) => ({ ...m, container: containers[i] })));
      });
      matches.sort((a, b) => (b.score || 0) - (a.score || 0));
      matches = matches.slice(0, RAG_K);
    }
  } catch { return null; }
  if (!matches.length) return null;
  const block = matches.map((m, i) =>
    `[${i + 1}] (${m.container}${m.source ? `, ${m.source}` : ""}, score ${m.score})\n${(m.text || "").trim()}`
  ).join("\n\n");
  return { block, n: matches.length };
}

// Merge/insert exactly ONE grounding system message at the front, keeping any existing system text.
function ground(messages, block) {
  const sys = messages.filter((m) => m.role === "system").map(textOf).filter(Boolean).join("\n\n");
  const rest = messages.filter((m) => m.role !== "system");
  const grounding =
    "You answer using the user's sovereign GitChain knowledge base. Use ONLY the passages below and " +
    "cite them inline as [n]. If they don't cover the question, say so plainly — do not invent facts.\n\n" +
    "=== RETRIEVED PASSAGES ===\n" + block;
  return [{ role: "system", content: sys ? sys + "\n\n" + grounding : grounding }, ...rest];
}

// Optional query rewrite: ask the LLM to turn the raw user turn into a focused retrieval query
// (keep entities/product names/numbers, drop chit-chat). Falls back to the raw query on any failure.
async function rewriteQuery(query) {
  try {
    const r = await fetch(LLM_URL, { method: "POST", headers: llmHeaders(), body: JSON.stringify({
      model: LLM_MODEL, stream: false, temperature: 0, max_tokens: 64,
      messages: [
        { role: "system", content: "Rewrite the user's message into a concise search query for a document " +
          "retrieval engine: keep the key entities, product names, numbers and terms; drop chit-chat and " +
          "politeness; output ONLY the query — no quotes, no explanation." },
        { role: "user", content: query },
      ],
    }) });
    if (!r.ok) return query;
    const j = await r.json();
    const q = (j.choices?.[0]?.message?.content || "").trim();
    return q || query;
  } catch { return query; }
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let d = ""; req.on("data", (c) => { d += c; if (d.length > 8 << 20) { req.destroy(); reject(new Error("body too large")); } });
    req.on("end", () => { try { resolve(d ? JSON.parse(d) : {}); } catch (e) { reject(e); } });
    req.on("error", reject);
  });
}

async function chatCompletions(req, res) {
  const body = await readBody(req);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  const query = textOf(lastUser).trim();

  let grounded = messages, hits = 0, usedQuery = query;
  if (query) {
    if (RAG_REWRITE) usedQuery = await rewriteQuery(query);
    const r = await retrieve(usedQuery);
    if (r) { grounded = ground(messages, r.block); hits = r.n; }
  }

  const outBody = { ...body, model: LLM_MODEL || body.model, messages: grounded };
  const stream = !!body.stream;
  const upstream = await fetch(LLM_URL, { method: "POST", headers: llmHeaders(), body: JSON.stringify(outBody) });

  res.setHeader("X-GitChain-Grounded", String(hits));   // observability: how many passages were injected
  res.setHeader("X-GitChain-Query", encodeURIComponent(usedQuery).slice(0, 400));  // the (rewritten) query used
  if (stream && upstream.body) {
    res.writeHead(upstream.status, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });
    const reader = upstream.body.getReader();
    for (;;) { const { done, value } = await reader.read(); if (done) break; res.write(Buffer.from(value)); }
    return res.end();
  }
  const text = await upstream.text();
  res.writeHead(upstream.status, { "Content-Type": "application/json" });
  res.end(text);
}

createServer(async (req, res) => {
  try {
    const path = new URL(req.url, "http://x").pathname;
    if (path === "/health" || path === "/api/v1/health")
      return res.end(JSON.stringify({ status: "healthy", server: "gitchain-rag-proxy", rag: GITCHAIN_API, llm: LLM_URL, model: LLM_MODEL }));
    if (path === "/v1/models")
      return res.end(JSON.stringify({ object: "list", data: [{ id: LLM_MODEL, object: "model", owned_by: "gitchain" }] }));
    if (path === "/v1/chat/completions" && req.method === "POST") return chatCompletions(req, res);
    res.writeHead(404, { "Content-Type": "application/json" }); res.end(JSON.stringify({ error: "not found" }));
  } catch (e) {
    res.writeHead(500, { "Content-Type": "application/json" }); res.end(JSON.stringify({ error: String((e && e.message) || e) }));
  }
}).listen(PORT, "127.0.0.1", () =>
  console.log(`gitchain-rag-proxy on http://127.0.0.1:${PORT}/v1  ·  RAG ${GITCHAIN_API}  →  LLM ${LLM_URL} (${LLM_MODEL})`));
