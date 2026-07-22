#!/usr/bin/env node
/**
 * gitchain-lite — a local Git server in ONE file: no database, no Docker, no dependencies.
 *
 * A full Git host (clone/fetch/push over Smart-HTTP) over a local directory of bare repos,
 * with a GitLab-style HIERARCHY:
 *
 *     Tenant (hard isolation)                  acme · globex · initech
 *       └─ Project(s) (nestable, like subgroups) hardware / sensors / …
 *            └─ Repo container (bare .git)      product · model · source · …
 *                 └─ Files + Actions            tree/blob/commits/diff/compare, clone/push, MR
 *
 * Die Hierarchie IST der Pfad (genau wie bei GitLab: group/subgroup/project.git). Das Dateisystem
 * ist die Wahrheit; `hierarchy.json` ist nur ein optionales Metadaten-Overlay (Anzeigenamen,
 * Isolation, Container-Typ) — die Struktur kann nie driften.
 *
 *   Layout:  <REPO_BASE>/<tenant>/<project>/<…nested>/<container>.git
 *            <REPO_BASE>/hierarchy.json                (optional, wird bei Actions geschrieben)
 *
 * Fähigkeiten:
 *   - Smart-HTTP: git clone / fetch / push  (git-upload-pack / git-receive-pack, N-Ebenen-Pfad)
 *   - Push-to-create: der erste Push auf einen neuen Pfad legt das bare Repo an
 *   - Git-Objekt-API (read, GitLab-Format): /git/<pfad>/refs · tree · blob · raw · commits ·
 *       commits/:sha/diff · compare
 *   - Hierarchie-API: /api/hierarchy · /api/tenants[/:t] · /api/projects/:t/*  (read)
 *       + POST /api/tenants · /api/projects · /api/containers  (Actions: anlegen)
 *   - Web-UI (public/index.html): Tenant → Projekt-Baum → Container → Files, mit Actions je Ebene
 *
 * Requires: only `git` + Node ≥18. Start:  REPO_BASE_PATH=./repos node gitchain-lite.mjs
 * Clone:  git clone http://localhost:7420/git/acme/hardware/sensors/widget-3000.git
 * Push:   git remote add gc http://localhost:7420/git/<tenant>/<project>/<id>.git && git push gc main
 *         (the first push creates the repo — no admin init needed)
 *
 * Hardening: argv-only git (no shell injection), a whitelist per path segment,
 * realpath containment against ..-traversal, and a body limit on write endpoints.
 */
import { createServer } from "http";
import { spawn, execFileSync } from "child_process";
import {
  existsSync, realpathSync, readFileSync, writeFileSync, readdirSync, statSync, mkdirSync, unlinkSync,
} from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_BASE = process.env.REPO_BASE_PATH || join(HERE, "repos");
const PORT = Number(process.env.PORT || 7420);
// Sovereign extras (all opt-in via env — unset = plain git server, unchanged behaviour):
const EMBED_URL = process.env.EMBED_URL || "";    // local /v1/embeddings (ingest + query)
const CHAT_URL = process.env.CHAT_URL || "";      // local /v1/chat/completions (answers)
const INGEST_CMD = process.env.INGEST_CMD || "";  // e.g. "node /path/ingest-worker.mjs --ingest"
// Packaged apps have spaces in argv[0] (…/GitChain Lite.app/…) — pass a JSON argv to avoid whitespace-splitting.
let INGEST_ARGV = null;
try { if (process.env.INGEST_CMD_JSON) INGEST_ARGV = JSON.parse(process.env.INGEST_CMD_JSON); } catch {}
const INGEST_AUTHOR_EMAIL = process.env.INGEST_AUTHOR_EMAIL || "ingest@gitchain.local";
// TurboQuant decode-on-open: if a container stores compressed vectors_tq.npz, the query path
// decodes it to fp32 via the numpy tq_decode (same PYTHON_BIN/toolkit as ingest).
const TOOLKIT_DIR = process.env.TOOLKIT_DIR || join(HERE, "toolkit");
const PYTHON_BIN = process.env.PYTHON_BIN || "python3";
const HTML = join(HERE, "public", "index.html");
const HIER_FILE = join(REPO_BASE, "hierarchy.json");
const SEG = /^[A-Za-z0-9._-]+$/; // ein Pfad-Segment (Tenant/Projekt/Container-Slug)
const SAFE_REF = /^[A-Za-z0-9._\/-]+$/;
const RESERVED = new Set(["refs", "tree", "blob", "raw", "commits", "compare"]); // Objekt-API-Verben
const okRef = (r) => typeof r === "string" && r && r.length < 256 && SAFE_REF.test(r) && !r.startsWith("-");
const okPath = (p) => typeof p === "string" && p.length < 1024 && !p.split("/").includes("..") && !p.startsWith("-");
const okSegs = (s) => Array.isArray(s) && s.length >= 2 && s.length <= 12 && s.every((x) => x && SEG.test(x));
mkdirSync(REPO_BASE, { recursive: true });

// ---- Pfad = Hierarchie ----------------------------------------------------------------
// segs = [tenant, ...project(nestbar), container].  Mindestens [tenant, container].
const parseSegs = (repoPath) => (typeof repoPath === "string" ? repoPath.split("/").filter(Boolean) : []);
const splitHierarchy = (segs) => ({ tenant: segs[0], project: segs.slice(1, -1), id: segs[segs.length - 1], segs });

// Verzeichnis unterhalb von REPO_BASE anlegen + Containment prüfen (Anti-Traversal).
function ensureDir(...segs) {
  if (!segs.every((s) => s && SEG.test(s))) return null;
  const dir = join(REPO_BASE, ...segs);
  mkdirSync(dir, { recursive: true });
  const real = realpathSync(dir), base = realpathSync(REPO_BASE);
  return real === base || real.startsWith(base + "/") ? real : null;
}

// ---- Repo-Auflösung (read) + create-on-push -------------------------------------------
function resolveRepo(segs, create = false) {
  if (!okSegs(segs)) return null;
  const dirSegs = segs.slice(0, -1), id = segs[segs.length - 1];
  const p = join(REPO_BASE, ...dirSegs, `${id}.git`);
  if (!existsSync(p)) {
    if (!create) return null;
    if (!ensureDir(...dirSegs)) return null;
    execFileSync("git", ["init", "--bare", "-q", "-b", "main", p]); // Default-HEAD = main
    console.log(`+ repo angelegt (push-to-create): ${segs.join("/")}.git`);
  }
  const real = realpathSync(p), base = realpathSync(REPO_BASE);
  return real === base || real.startsWith(base + "/") ? real : null;
}
const git = (r, a) => execFileSync("git", ["--git-dir", r, ...a], { encoding: "utf8", maxBuffer: 256 << 20, timeout: 15000 }).toString();
const F = "\x1f";
// Hat das Repo überhaupt einen Commit? (frisch angelegte Container sind leer → kein HEAD)
const hasHead = (r) => { try { git(r, ["rev-parse", "--verify", "--quiet", "HEAD"]); return true; } catch { return false; } };

// Rekursiver Scan: jedes <…>.git ist ein Container; sein Pfad kodiert Tenant + Projekt(e) + Id.
function walkRepos(dir = REPO_BASE, rel = []) {
  const out = [];
  let names;
  try { names = readdirSync(dir); } catch { return out; }
  for (const name of names) {
    const full = join(dir, name);
    let st; try { st = statSync(full); } catch { continue; }
    if (!st.isDirectory()) continue;
    if (name.endsWith(".git")) {
      const id = name.slice(0, -4);
      if (!SEG.test(id)) continue;
      const segs = [...rel, id];
      out.push({ segs, tenant: segs[0], project: segs.slice(1, -1), id, path_with_namespace: segs.join("/") });
    } else if (SEG.test(name) && rel.length < 11) {
      out.push(...walkRepos(full, [...rel, name]));
    }
  }
  return out;
}
const listRepos = () => walkRepos();

// ---- hierarchy.json (Metadaten-Overlay) -----------------------------------------------
function loadHier() {
  try { return JSON.parse(readFileSync(HIER_FILE, "utf8")); } catch { return { tenants: {}, projects: {}, containers: {} }; }
}
function saveHier(h) { writeFileSync(HIER_FILE, JSON.stringify(h, null, 2)); }

// FS-Scan + Overlay -> genestete Hierarchie (Tenant -> Projekt-Baum -> Container).
function hierarchyTree() {
  const h = loadHier();
  const tName = (s) => h.tenants?.[s]?.name || s;
  const tIso = (s) => h.tenants?.[s]?.isolation || "hard";
  const pMeta = (t, path) => h.projects?.[`${t}/${path}`] || {};
  const cMeta = (pwn) => h.containers?.[pwn] || {};
  const tenants = {};
  const ensureT = (slug) => (tenants[slug] ??= { slug, name: tName(slug), isolation: tIso(slug), _p: {}, root_containers: [], container_count: 0 });
  const ensureP = (t, projPath) => {
    let level = ensureT(t)._p, node = null, accum = [];
    for (const seg of projPath.split("/")) {
      accum.push(seg);
      const path = accum.join("/");
      node = (level[seg] ??= { slug: seg, path, name: pMeta(t, path).name || seg, description: pMeta(t, path).description || null, type: pMeta(t, path).type || null, _p: {}, containers: [], container_count: 0 });
      level = node._p;
    }
    return node;
  };
  for (const r of listRepos()) {
    const t = ensureT(r.tenant); t.container_count++;
    const c = { id: r.id, path_with_namespace: r.path_with_namespace, ...cMeta(r.path_with_namespace) };
    c.type ||= null; c.title ||= null;
    if (r.project.length === 0) { t.root_containers.push(c); continue; }
    let level = t._p, node = null, accum = [];
    for (const seg of r.project) {
      accum.push(seg);
      const path = accum.join("/");
      node = (level[seg] ??= { slug: seg, path, name: pMeta(r.tenant, path).name || seg, description: pMeta(r.tenant, path).description || null, type: pMeta(r.tenant, path).type || null, _p: {}, containers: [], container_count: 0 });
      node.container_count++;
      level = node._p;
    }
    node.containers.push(c);
  }
  // leere (frisch angelegte) Projekte / Tenants aus dem Overlay einblenden
  for (const key of Object.keys(h.projects || {})) { const [t, ...rest] = key.split("/"); if (t && rest.length) ensureP(t, rest.join("/")); }
  for (const slug of Object.keys(h.tenants || {})) ensureT(slug);
  const norm = (n) => ({ slug: n.slug, path: n.path, name: n.name, description: n.description, type: n.type, container_count: n.container_count, containers: n.containers, projects: Object.values(n._p).map(norm).sort((a, b) => a.name.localeCompare(b.name)) });
  return {
    tenants: Object.values(tenants).map((t) => ({
      slug: t.slug, name: t.name, isolation: t.isolation, container_count: t.container_count,
      project_count: Object.keys(t._p).length, root_containers: t.root_containers,
      projects: Object.values(t._p).map(norm).sort((a, b) => a.name.localeCompare(b.name)),
    })).sort((a, b) => a.name.localeCompare(b.name)),
  };
}

// ---- Git-Objekt-API (read, GitLab-Format) ---------------------------------------------
const obj = {
  refs(repo) {
    const head = git(repo, ["symbolic-ref", "--short", "HEAD"]).trim();
    const raw = git(repo, ["for-each-ref", `--format=%(refname:short)\t%(objecttype)\t%(objectname)\t%(refname:rstrip=-2)`, "refs/heads", "refs/tags"]).trim();
    const branches = [], tags = [];
    for (const l of raw ? raw.split("\n") : []) {
      const [name, , sha, kind] = l.split("\t");
      const e = { name, commit: { id: sha, short_id: sha.slice(0, 8) } };
      kind === "refs/heads" ? branches.push({ ...e, default: name === head, protected: name === head }) : tags.push(e);
    }
    return { default_branch: head, branches, tags };
  },
  tree(repo, q) {
    const ref = q.ref || "HEAD", path = q.path || "";
    if (!okRef(ref) || !okPath(path)) throw 400;
    if ((ref === "HEAD" || ref === "main" || ref === "master") && !hasHead(repo)) return { ref, path, entries: [] };
    const raw = git(repo, ["ls-tree", "--long", path ? `${ref}:${path}` : ref]).trim();
    return { ref, path, entries: (raw ? raw.split("\n") : []).map((l) => {
      const [meta, name] = l.split("\t"); const [mode, type, sha, size] = meta.split(/\s+/);
      return { id: sha.slice(0, 8), name, type: type === "tree" ? "tree" : "blob", path: path ? `${path}/${name}` : name, mode, size: size === "-" ? null : +size };
    }) };
  },
  commits(repo, q) {
    const ref = q.ref || "HEAD", limit = Math.min(+q.limit || 30, 200); if (!okRef(ref)) throw 400;
    if ((ref === "HEAD" || ref === "main" || ref === "master") && !hasHead(repo)) return { ref, count: 0, commits: [] };
    const args = ["log", `--format=%H${F}%h${F}%an${F}%aI${F}%s`, `-${limit}`, ref];
    if (q.path && okPath(q.path)) args.push("--", q.path);
    const raw = git(repo, args).trim();
    const commits = (raw ? raw.split("\n") : []).map((l) => { const [id, short_id, author_name, created_at, title] = l.split(F); return { id, short_id, author_name, created_at, title }; });
    return { ref, count: commits.length, commits };
  },
  blob(repo, q) {
    const ref = q.ref || "HEAD", path = q.path; if (!okRef(ref) || !okPath(path) || !path) throw 400;
    const spec = `${ref}:${path}`, size = +git(repo, ["cat-file", "-s", spec]).trim(); const body = { file_path: path, ref, size };
    if (size <= 5 << 20) { body.encoding = "utf8"; body.content = execFileSync("git", ["--git-dir", repo, "cat-file", "blob", spec], { maxBuffer: 6 << 20 }).toString("utf8"); }
    else body.note = "too large — use /raw";
    return body;
  },
  compare(repo, q) {
    const { from, to } = q; if (!okRef(from) || !okRef(to)) throw 400;
    const cRaw = git(repo, ["log", `--format=%h${F}%s`, `${from}..${to}`]).trim();
    const commits = (cRaw ? cRaw.split("\n") : []).map((l) => { const [short_id, title] = l.split(F); return { short_id, title }; });
    const stat = git(repo, ["diff", "--numstat", `${from}..${to}`]).trim();
    const files = (stat ? stat.split("\n") : []).map((l) => { const [a, d, name] = l.split("\t"); return { path: name, additions: a === "-" ? null : +a, deletions: d === "-" ? null : +d }; });
    return { from, to, commits, files, patch: git(repo, ["diff", "--no-color", `${from}..${to}`]) };
  },
};
function commitDiff(repo, sha) {
  if (!okRef(sha)) throw 400;
  const meta = git(repo, ["show", "-s", `--format=%H${F}%h${F}%an${F}%aI${F}%s${F}%P`, sha]).trim().split(F);
  const stat = git(repo, ["show", sha, "--format=", "--numstat"]).trim();
  const files = (stat ? stat.split("\n") : []).map((l) => { const [a, d, name] = l.split("\t"); return { path: name, additions: a === "-" ? null : +a, deletions: d === "-" ? null : +d }; });
  return { commit: { id: meta[0], short_id: meta[1], author_name: meta[2], created_at: meta[3], title: meta[4], parent_ids: meta[5] ? meta[5].split(" ") : [] },
    stats: { files_changed: files.length, additions: files.reduce((a, f) => a + (f.additions || 0), 0), deletions: files.reduce((a, f) => a + (f.deletions || 0), 0) }, files, patch: git(repo, ["show", sha, "--format=", "--patch", "--no-color"]) };
}

// ---- Smart-HTTP (clone/fetch/push) ----------------------------------------------------
function smartInfoRefs(req, res, segs) {
  const service = new URL(req.url, "http://x").searchParams.get("service");
  if (service !== "git-upload-pack" && service !== "git-receive-pack") { res.writeHead(400).end("bad service"); return; }
  const repo = resolveRepo(segs, service === "git-receive-pack"); // push legt an
  if (!repo) { res.writeHead(404).end("repository not found"); return; }
  const cmd = service.replace("git-", "");
  res.writeHead(200, { "Content-Type": `application/x-${service}-advertisement`, "Cache-Control": "no-cache" });
  const head = `# service=${service}\n`;
  res.write((head.length + 4).toString(16).padStart(4, "0") + head + "0000");
  const p = spawn("git", [cmd, "--stateless-rpc", "--advertise-refs", repo]);
  p.stdout.pipe(res); p.stderr.on("data", (d) => console.error(String(d)));
}
function smartRpc(req, res, segs, service) {
  const repo = resolveRepo(segs, service === "git-receive-pack");
  if (!repo) { res.writeHead(404).end("repository not found"); return; }
  const cmd = service.replace("git-", "");
  res.writeHead(200, { "Content-Type": `application/x-${service}-result`, "Cache-Control": "no-cache" });
  const p = spawn("git", [cmd, "--stateless-rpc", repo]);
  req.pipe(p.stdin); p.stdout.pipe(res); p.stderr.on("data", (d) => console.error(String(d)));
  // Nach einem Push: HEAD, der ins Leere zeigt, auf einen existierenden Branch reparieren.
  if (service === "git-receive-pack") p.on("close", () => { repairHead(repo); maybeIngest(repo); });
}
function repairHead(repo) {
  try {
    try { git(repo, ["rev-parse", "--verify", "--quiet", "HEAD"]); return; } catch { /* HEAD dangling */ }
    const heads = git(repo, ["for-each-ref", "--format=%(refname:short)", "refs/heads"]).trim().split("\n").filter(Boolean);
    if (!heads.length) return;
    const pick = heads.includes("main") ? "main" : heads.includes("master") ? "master" : heads[0];
    execFileSync("git", ["--git-dir", repo, "symbolic-ref", "HEAD", `refs/heads/${pick}`]);
    console.log(`  HEAD → ${pick} (${repo.split("/").slice(-3).join("/")})`);
  } catch (e) { console.error("repairHead:", String(e)); }
}

// ---- Sovereign extras (opt-in via env): ingest-on-push + local semantic query ----------
// On a push, if the repo carries .gitchain/ingest.json, run the configured ingest command
// (build/atomize via the local models) — no queue, no DB. Skips our own artifact write-back.
function maybeIngest(repo) {
  if (!INGEST_CMD && !INGEST_ARGV) return;
  try {
    if (!hasHead(repo)) return;
    let cfg; try { cfg = git(repo, ["show", "HEAD:.gitchain/ingest.json"]); } catch { return; }
    if (!cfg.trim()) return;
    let author = ""; try { author = git(repo, ["log", "-1", "--format=%ae", "HEAD"]).trim(); } catch {}
    if (author === INGEST_AUTHOR_EMAIL) return; // our own commit-back — don't loop
    const argv = INGEST_ARGV || INGEST_CMD.split(/\s+/);
    const p = spawn(argv[0], [...argv.slice(1), repo], { stdio: "inherit", env: process.env });
    console.log(`  ingest → ${repo.split("/").slice(-3).join("/")}`);
    p.on("error", (e) => console.error("ingest:", String(e)));
  } catch (e) { console.error("maybeIngest:", String(e)); }
}
async function embedOne(text, model) {
  const r = await fetch(EMBED_URL, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: model || "embeddinggemma", input: [text] }) });
  if (!r.ok) throw new Error(`embed HTTP ${r.status}`);
  return (await r.json()).data[0].embedding;
}
async function chatAnswer(q, matches) {
  const ctx = matches.map((m, i) => `[${i + 1}] ${m.text}`).join("\n\n");
  const prompt = `Use ONLY the context to answer; cite sources as [n]. If it isn't in the context, say so.\n\nContext:\n${ctx}\n\nQuestion: ${q}`;
  const r = await fetch(CHAT_URL, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "local", messages: [{ role: "user", content: prompt }], temperature: 0.2, max_tokens: 512 }) });
  if (!r.ok) throw new Error(`chat HTTP ${r.status}`);
  const j = await r.json();
  return (j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content) || "";
}
// Load a container's vectors as fp32 (L2-normalized, chunk order). Prefers the raw .brain/vectors.f32
// blob; if the container is a TurboQuant store (only vectors_tq.npz), decode-on-open via numpy
// tq_decode and cache the fp32 by HEAD sha (so a container quantized at rest is still queryable).
const _vecCache = new Map(); // `${repo}|${sha}` -> Float32Array  (pro Repo nur der aktuelle HEAD; max 8 Container resident)
const VEC_CACHE_MAX = Number(process.env.VEC_CACHE_MAX || 8);
function vecCachePut(repo, key, vecs) {
  for (const k of _vecCache.keys()) if (k.startsWith(repo + "|") && k !== key) _vecCache.delete(k); // alter HEAD desselben Repos
  _vecCache.set(key, vecs);
  while (_vecCache.size > VEC_CACHE_MAX) _vecCache.delete(_vecCache.keys().next().value); // LRU: ältester Eintrag
}
const catBlob = (repo, path) =>
  execFileSync("git", ["--git-dir", repo, "cat-file", "blob", `HEAD:${path}`], { maxBuffer: 512 << 20 });
const blobExists = (repo, path) => {
  try { execFileSync("git", ["--git-dir", repo, "cat-file", "-e", `HEAD:${path}`]); return true; } catch { return false; }
};
const asF32 = (buf) => new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
function loadContainerVectors(repo, index) {
  const store = String(index.store || "f32");
  const isTq = store.startsWith("turboquant") || store.startsWith("tq");
  if (!isTq && blobExists(repo, ".brain/vectors.f32")) return asF32(catBlob(repo, ".brain/vectors.f32"));
  if (!blobExists(repo, ".brain/vectors_tq.npz"))                       // tq claimed but absent → fall back
    return blobExists(repo, ".brain/vectors.f32") ? asF32(catBlob(repo, ".brain/vectors.f32")) : new Float32Array(0);
  const sha = git(repo, ["rev-parse", "HEAD"]).trim();
  const key = `${repo}|${sha}`;
  const hit = _vecCache.get(key);
  if (hit) { _vecCache.delete(key); _vecCache.set(key, hit); return hit; } // LRU-Touch
  const cacheDir = join(REPO_BASE, ".cache"); mkdirSync(cacheDir, { recursive: true });
  const npz = join(cacheDir, `${sha.slice(0, 16)}.npz`);
  writeFileSync(npz, catBlob(repo, ".brain/vectors_tq.npz"));
  let vecs;
  try { vecs = asF32(execFileSync(PYTHON_BIN, [join(TOOLKIT_DIR, "tq_decode.py"), "--raw", npz], { maxBuffer: 512 << 20 })); }
  finally { try { unlinkSync(npz); } catch {} }
  vecCachePut(repo, key, vecs);
  console.log(`  decode-on-open: ${store} → ${vecs.length} floats  (${repo.split("/").slice(-3).join("/")})`);
  return vecs;
}

// Semantic search over ONE container's own .brain vectors (fp32, or TurboQuant decode-on-open). No index DB.
async function queryContainer(b) {
  const q = String(b.q || "").trim(); if (!q) throw 400;
  const container = String(b.container || "").trim();
  const segs = container.startsWith("0711:") ? container.slice(5).split(":") : parseSegs(container);
  const repo = resolveRepo(segs, false);
  if (!repo) return { error: "container not found" };
  if (!EMBED_URL) return { error: "EMBED_URL not configured (local embeddings)" };
  let index; try { index = JSON.parse(git(repo, ["show", "HEAD:.brain/index.json"])); }
  catch { return { error: "container not built yet (no .brain/index.json)" }; }
  const order = index.order || [], dims = index.dims || 768, chunks = {};
  try { for (const l of git(repo, ["show", "HEAD:data/chunks.jsonl"]).split("\n")) if (l.trim()) { const c = JSON.parse(l); chunks[c.id] = c; } } catch {}
  const vecs = loadContainerVectors(repo, index);
  const n = Math.min(order.length, Math.floor(vecs.length / dims));
  const qv = await embedOne(q, index.model);
  let qn = 0; for (let i = 0; i < dims; i++) qn += qv[i] * qv[i]; qn = Math.sqrt(qn) || 1;
  const k = Math.min(Math.max(1, Number(b.k) || 8), 30);
  const scored = [];
  for (let r = 0; r < n; r++) { let dot = 0; const off = r * dims; for (let i = 0; i < dims; i++) dot += qv[i] * vecs[off + i]; scored.push([r, dot / qn]); }
  scored.sort((a, b) => b[1] - a[1]);
  const matches = scored.slice(0, k).map(([r, s]) => { const id = order[r], c = chunks[id] || {}; return { score: +s.toFixed(4), id, source: c.source, text: (c.text || "").slice(0, 700) }; });
  const out = { container: segs.join("/"), q, count: n, matches };
  if (b.answer && CHAT_URL) { try { out.answer = await chatAnswer(q, matches); } catch (e) { out.answer_error = String((e && e.message) || e); } }
  return out;
}

// ---- Body-Reader für Write-Actions ----------------------------------------------------
function readJson(req) {
  return new Promise((resolve, reject) => {
    let data = ""; req.on("data", (c) => { data += c; if (data.length > 1 << 20) { req.destroy(); reject(413); } });
    req.on("end", () => { try { resolve(data ? JSON.parse(data) : {}); } catch { reject(400); } });
    req.on("error", reject);
  });
}

// ---- HTTP-Router ----------------------------------------------------------------------
createServer(async (req, res) => {
  const u = new URL(req.url, "http://x");
  const P = u.pathname;
  const send = (c, o) => { res.writeHead(c, { "Content-Type": "application/json" }); res.end(JSON.stringify(o)); };
  try {
    if (P === "/" || P === "/index.html") { res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" }); return res.end(readFileSync(HTML)); }

    // -------- Health + local semantic query (sovereign extras) --------
    if (P === "/api/v1/health") return send(200, { status: "healthy", server: "gitchain-lite", models: !!EMBED_URL, ingest: !!INGEST_CMD });
    if (P === "/api/v1/query" && req.method === "POST") return send(200, await queryContainer(await readJson(req)));

    // -------- Hierarchie-API (read) --------
    if (P === "/api/hierarchy") return send(200, hierarchyTree());
    if (P === "/api/tenants" && req.method === "GET")
      return send(200, { tenants: hierarchyTree().tenants.map(({ projects, root_containers, ...t }) => t) });
    let m = P.match(/^\/api\/tenants\/([^/]+)$/);
    if (m && req.method === "GET") {
      const t = hierarchyTree().tenants.find((x) => x.slug === m[1]);
      return t ? send(200, t) : send(404, { error: "tenant not found" });
    }
    m = P.match(/^\/api\/projects\/([^/]+)\/(.+)$/);
    if (m && req.method === "GET") {
      const t = hierarchyTree().tenants.find((x) => x.slug === m[1]);
      if (!t) return send(404, { error: "tenant not found" });
      const path = m[2]; let found = null;
      const walk = (nodes) => { for (const n of nodes) { if (n.path === path) found = n; else walk(n.projects); } };
      walk(t.projects);
      return found ? send(200, { tenant: t.slug, ...found }) : send(404, { error: "project not found" });
    }

    // -------- Actions (anlegen) — DB-less: mkdir + hierarchy.json --------
    if (P === "/api/tenants" && req.method === "POST") {
      const b = await readJson(req);
      const slug = String(b.slug || "").trim();
      if (!SEG.test(slug)) return send(400, { error: "slug muss [A-Za-z0-9._-] sein" });
      if (!ensureDir(slug)) return send(400, { error: "ungültiger Pfad" });
      const h = loadHier(); h.tenants ||= {};
      h.tenants[slug] = { name: b.name || slug, isolation: b.isolation === "soft" ? "soft" : "hard", created_at: new Date().toISOString() };
      saveHier(h);
      return send(201, { tenant: { slug, ...h.tenants[slug] } });
    }
    if (P === "/api/projects" && req.method === "POST") {
      const b = await readJson(req);
      const tenant = String(b.tenant || "").trim(), path = String(b.path || "").trim().replace(/^\/|\/$/g, "");
      const segs = path.split("/").filter(Boolean);
      if (!SEG.test(tenant) || !segs.length || !segs.every((s) => SEG.test(s))) return send(400, { error: "tenant + path (Segmente [A-Za-z0-9._-]) erforderlich" });
      if (!ensureDir(tenant, ...segs)) return send(400, { error: "ungültiger Pfad" });
      const h = loadHier(); h.projects ||= {}; h.tenants ||= {};
      h.tenants[tenant] ||= { name: tenant, isolation: "hard", created_at: new Date().toISOString() };
      h.projects[`${tenant}/${path}`] = { name: b.name || segs[segs.length - 1], description: b.description || null, type: b.type || null, created_at: new Date().toISOString() };
      saveHier(h);
      return send(201, { project: { tenant, path, name: h.projects[`${tenant}/${path}`].name } });
    }
    if (P === "/api/containers" && req.method === "POST") {
      const b = await readJson(req);
      const tenant = String(b.tenant || "").trim();
      const project = String(b.project || "").trim().replace(/^\/|\/$/g, "");
      const id = String(b.id || "").trim();
      const segs = [tenant, ...project.split("/").filter(Boolean), id];
      if (!okSegs(segs)) return send(400, { error: "tenant + id (Segmente [A-Za-z0-9._-]) erforderlich" });
      const repo = resolveRepo(segs, true); // git init --bare (leerer Container)
      if (!repo) return send(400, { error: "konnte Container nicht anlegen" });
      const pwn = segs.join("/");
      const h = loadHier(); h.containers ||= {};
      h.containers[pwn] = { type: b.type || null, title: b.title || null, created_at: new Date().toISOString() };
      saveHier(h);
      const base = `http://${req.headers.host || `localhost:${PORT}`}`;
      return send(201, { container: { path_with_namespace: pwn, ...h.containers[pwn] }, clone_url: `${base}/git/${pwn}.git`, push_url: `${base}/git/${pwn}.git` });
    }

    // -------- Katalog (flach, rückwärtskompatibel) --------
    if (P === "/git/repos") { const repos = listRepos(); return send(200, { count: repos.length, repos }); }

    // -------- Smart-HTTP + Git-Objekt-API (N-Ebenen-Pfad) --------
    if (P.startsWith("/git/")) {
      const rest = P.slice(5);
      let g;
      if ((g = rest.match(/^(.+)\.git\/info\/refs$/)) && req.method === "GET") return smartInfoRefs(req, res, parseSegs(g[1]));
      if ((g = rest.match(/^(.+)\.git\/(git-upload-pack|git-receive-pack)$/)) && req.method === "POST") return smartRpc(req, res, parseSegs(g[1]), g[2]);
      if ((g = rest.match(/^(.+)\/commits\/([^/]+)\/diff$/))) { const repo = resolveRepo(parseSegs(g[1])); return repo ? send(200, commitDiff(repo, g[2])) : send(404, { error: "repository not found" }); }
      if ((g = rest.match(/^(.+)\/(refs|tree|blob|raw|commits|compare)$/))) {
        const kind = g[2], repo = resolveRepo(parseSegs(g[1]));
        if (!repo) return send(404, { error: "repository not found" });
        const q = Object.fromEntries(u.searchParams);
        if (kind === "raw") {
          const ref = q.ref || "HEAD";
          if (!okRef(ref) || !okPath(q.path) || !q.path) throw 400; // wie blob(): kein "-"-Prefix → keine git-Option-Injection
          const buf = execFileSync("git", ["--git-dir", repo, "cat-file", "blob", `${ref}:${q.path}`], { maxBuffer: 256 << 20 });
          res.writeHead(200, { "Content-Type": "application/octet-stream" }); return res.end(buf);
        }
        return send(200, obj[kind](repo, q));
      }
    }
    send(404, { error: "not found" });
  } catch (e) {
    if (e === 400) return send(400, { error: "invalid parameter" });
    if (e === 413) return send(413, { error: "body too large" });
    console.error(e); send(500, { error: String((e && e.message) || e).slice(0, 160) });
  }
}).listen(PORT, () => {
  const t = hierarchyTree().tenants;
  console.log(`\n  gitchain-lite → http://localhost:${PORT}`);
  console.log(`  Repos:   ${REPO_BASE}  (${listRepos().length} Container in ${t.length} Tenants)`);
  console.log(`  Web-UI:  http://localhost:${PORT}/`);
  console.log(`  Hierarchie: Tenant → Project(s, nestbar) → Repo-Container → Files + Actions`);
  console.log(`  Push:    git push http://localhost:${PORT}/git/<tenant>/<projekt>/<id>.git --all  (legt beim ersten Push an)\n`);
});
