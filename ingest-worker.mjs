#!/usr/bin/env node
/**
 * gitchain-ingest-worker — hangs the container toolkit onto the git-push chain.
 *
 * git push (docs + .gitchain/ingest.json)               ← the customer, PAT-auth
 *   └─▶ service emits `git.push` on Redis stream 0711.events   (already live, unchanged)
 *        └─▶ [consume]  this worker reads the stream, and if the pushed repo carries
 *                       .gitchain/ingest.json, enqueues a migration_jobs(kind='ingest') row
 *             └─▶ [work] poll loop (FOR UPDATE SKIP LOCKED) picks the job and runs the chain:
 *                        clone bare → worktree → gitchain-build|atomize (+quantize)
 *                        → commit artifacts → push back to the SAME bare repo (local file push,
 *                          which does NOT go through the HTTP receive-pack endpoint, so it emits
 *                          NO new git.push event → no loop).
 *
 * It touches neither the live :3361 service nor the (separate) promote-worker: it only consumes an
 * event the service already emits and reuses the existing migration_jobs queue.
 *
 * The repo declares HOW it wants to be built via a committed `.gitchain/ingest.json`:
 *   { "mode": "build",  "source": "docs", "model": "embedding-model", "dims": 768, "level": "b2" }
 *   { "mode": "atomize","claims": "data/claims.jsonl", "boxes": "data/boxes.jsonl",
 *                       "fulltext": "data/fulltext.jsonl" }
 * Infra endpoints (EMBED_URL / READ_URL / TQ_STORE_PATH) come from the worker's env, never the repo.
 *
 * Run modes:
 *   node ingest-worker.mjs                     both loops (consume + work) — needs pg + redis + DB + Redis
 *   node ingest-worker.mjs --once <cid>        enqueue + run one container synchronously (needs DB)
 *   node ingest-worker.mjs --selftest <bare>   run ONLY the git+toolkit chain on a local bare repo
 *                                              (no DB, no Redis) — the end-to-end proof
 *   add --dry-run to print the toolkit commands without executing python.
 */
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import * as fssync from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const execFileP = promisify(execFile);
const HERE = path.dirname(fileURLToPath(import.meta.url));

// ── config (all env-driven; defaults mirror the :3361 service) ──────────────
const REPO_BASE = process.env.REPO_BASE_PATH || "/var/lib/gitchain/repos";
const TOOLKIT_DIR = process.env.TOOLKIT_DIR || path.resolve(HERE, "..");
const PYTHON_BIN = process.env.PYTHON_BIN || "python3";
const EMBED_URL = process.env.EMBED_URL || "";
const READ_URL = process.env.READ_URL || "";
const WORKTREE_BASE = process.env.WORKTREE_BASE || "/tmp";
const POLL_INTERVAL_MS = parseInt(process.env.POLL_INTERVAL_MS || "2000", 10);
const EVENTS_REDIS_URL =
  process.env.EVENTS_REDIS_URL || process.env.REDIS_URL || "redis://localhost:6379";
const INGEST_AUTHOR_NAME = "gitchain-ingest";
const INGEST_AUTHOR_EMAIL = process.env.INGEST_AUTHOR_EMAIL || "ingest@gitchain.local";
const CURSOR_KEY = "gitchain:ingest-worker:cursor";
const MAXBUF = 256 * 1024 * 1024;

let STOP = false;
const log0 = (m) => process.stderr.write(`[ingest-worker] ${m}\n`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const stamp = () => crypto.randomBytes(6).toString("hex");

// ── lazy singletons (so --selftest needs NO npm install of pg/redis) ────────
let _pool = null;
async function getPool() {
  if (_pool) return _pool;
  const { Pool } = await import("pg");
  _pool = process.env.GITCHAIN_DB_URL
    ? new Pool({ connectionString: process.env.GITCHAIN_DB_URL, max: 6 })
    : new Pool({
        host: process.env.DB_HOST || "localhost",
        port: parseInt(process.env.DB_PORT || "5433", 10),
        database: process.env.DB_NAME || "gitchain",
        user: process.env.DB_USER || "gitchain",
        password: process.env.DB_PASSWORD || "gitchain_password_2026",
        max: 6,
      });
  return _pool;
}
let _redis = null;
async function getRedis() {
  if (_redis) return _redis;
  const { createClient } = await import("redis");
  _redis = createClient({ url: EVENTS_REDIS_URL });
  _redis.on("error", (e) => log0(`redis error: ${e.message}`));
  await _redis.connect();
  return _redis;
}

// ── git helpers ─────────────────────────────────────────────────────────────
async function git(args, opts = {}) {
  return execFileP("git", args, { maxBuffer: MAXBUF, ...opts });
}
async function contentRef(gargs) {
  // A ref that actually has commits. Push-created bare repos default HEAD to an unborn 'master'
  // while modern clients push 'main', so HEAD often resolves to nothing — find the real branch.
  try { await git([...gargs, "rev-parse", "--verify", "-q", "HEAD^{commit}"]); return "HEAD"; } catch {}
  let heads = [];
  try {
    heads = (await git([...gargs, "for-each-ref", "--format=%(refname:short)", "refs/heads"]))
      .stdout.split("\n").map((s) => s.trim()).filter(Boolean);
  } catch {}
  return heads.find((b) => b === "main") || heads.find((b) => b === "master") || heads[0] || null;
}
function repoPathForContainer(containerId) {
  // container_id format: 0711:<type>:<namespace>:<id>
  const parts = containerId.split(":");
  if (parts.length < 4) throw new Error(`bad container_id ${containerId}`);
  const [, type, namespace, id] = parts;
  return path.join(REPO_BASE, type, namespace, `${id}.git`);
}
async function sourceKey(revAt, cfg) {
  // revAt(p) => tree/blob sha of path p at HEAD; a stable fingerprint of the INPUTS only
  if (cfg.mode === "build") return "b:" + (await revAt(cfg.source || "docs"));
  return (
    "a:" +
    (await revAt(cfg.claims || "data/claims.jsonl")) +
    ":" +
    (await revAt(cfg.boxes || "data/boxes.jsonl"))
  );
}

// ── the chain (no DB / no Redis) — clone → toolkit → commit-back ─────────────
async function runChain(barePath, { dryRun = false, log = async () => {} } = {}) {
  if (!fssync.existsSync(barePath)) throw new Error(`bare repo not found: ${barePath}`);
  const worktree = path.join(WORKTREE_BASE, `ingest-${path.basename(barePath, ".git")}-${stamp()}`);
  await log({ stage: 1, level: "info", msg: `clone ${barePath} -> ${worktree}` });
  await git(["clone", "--quiet", barePath, worktree]);
  try {
    // Resolve the branch that has commits — a push-created bare repo may leave HEAD on an unborn 'master'.
    let branch;
    const bornHead = await git(["-C", worktree, "rev-parse", "--verify", "-q", "HEAD^{commit}"])
      .then(() => true).catch(() => false);
    if (bornHead) {
      branch = (await git(["-C", worktree, "rev-parse", "--abbrev-ref", "HEAD"])).stdout.trim();
    } else {
      const rb = (await git(["-C", worktree, "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin"]))
        .stdout.split("\n").map((s) => s.trim().replace(/^origin\//, "")).filter((b) => b && b !== "HEAD");
      branch = rb.find((b) => b === "main") || rb.find((b) => b === "master") || rb[0];
      if (!branch) throw new Error("pushed repo has no branch with commits");
      await git(["-C", worktree, "checkout", "-q", "-B", branch, `origin/${branch}`]);
    }
    const cfgPath = path.join(worktree, ".gitchain", "ingest.json");
    if (!fssync.existsSync(cfgPath)) throw new Error(".gitchain/ingest.json missing in pushed repo");
    const cfg = JSON.parse(fssync.readFileSync(cfgPath, "utf8"));
    if (!["build", "atomize"].includes(cfg.mode)) throw new Error(`bad ingest mode: ${cfg.mode}`);
    await log({ stage: 2, level: "info", msg: `mode=${cfg.mode} branch=${branch}` });

    const revAt = async (p) =>
      (await git(["-C", worktree, "rev-parse", `HEAD:${p}`])).stdout.trim();
    const key = await sourceKey(revAt, cfg).catch(() => "unresolved-source"); // deterministic fallback

    // build the toolkit command list (labelled)
    const cmds = [];
    // Turbo quantize only when a GPU codec is wired in (TQ_STORE_PATH) and the repo doesn't opt out.
    // Without it, build ships an fp32 brain container — fully usable, just uncompressed (the free tier).
    const willQuantize = !!process.env.TQ_STORE_PATH && cfg.quantize !== false;
    if (cfg.mode === "build") {
      const source = path.join(worktree, cfg.source || "docs");
      if (!EMBED_URL && !dryRun) throw new Error("EMBED_URL required for build mode");
      const b = ["gitchain-build.py", "--source", source, "--container", worktree,
        "--embed-url", EMBED_URL || "http://EMBED_URL/v1/embeddings",
        "--model", cfg.model || "embedding-model", "--dims", String(cfg.dims || 768), "--no-commit"];
      if (READ_URL) b.push("--read-url", READ_URL);
      cmds.push(["build", b]);
      if (willQuantize) {
        cmds.push(["quantize", ["gitchain-quantize.py", worktree,
          "--vectors", path.join(worktree, ".brain", "vectors.f32"),
          "--dims", String(cfg.dims || 768), "--level", cfg.level || "b2", "--no-commit"]]);
      }
    } else {
      const a = ["gitchain-atomize.py",
        "--claims", path.join(worktree, cfg.claims || "data/claims.jsonl"),
        "--boxes", path.join(worktree, cfg.boxes || "data/boxes.jsonl"),
        "--container", worktree, "--no-commit"];
      if (cfg.fulltext) a.push("--fulltext", path.join(worktree, cfg.fulltext));
      cmds.push(["atomize", a]);
    }

    for (const [label, args] of cmds) {
      await log({ stage: 3, level: "info", msg: `${dryRun ? "[dry-run] " : ""}${label}: ${PYTHON_BIN} ${args.join(" ")}` });
      if (dryRun) continue;
      const r = await execFileP(PYTHON_BIN, args, {
        cwd: TOOLKIT_DIR, maxBuffer: MAXBUF,
        env: { ...process.env, TQ_STORE_PATH: process.env.TQ_STORE_PATH || "" },
      });
      const last = r.stdout.trim().split("\n").pop() || "";
      await log({ stage: 3, level: "info", msg: `${label} ok: ${last.slice(0, 200)}` });
    }

    if (cfg.mode === "build" && willQuantize && !cfg.keep_fp32 && !dryRun) {
      const f32 = path.join(worktree, ".brain", "vectors.f32");
      if (fssync.existsSync(f32)) fssync.rmSync(f32); // quantized store shipped; drop the fp32 source
    }

    // idempotency marker (the INPUT fingerprint) so a re-push of the same source is a no-op
    fssync.mkdirSync(path.join(worktree, ".gitchain"), { recursive: true });
    fssync.writeFileSync(path.join(worktree, ".gitchain", "ingest.built"), key + "\n");

    await git(["-C", worktree, "add", "-A"]);
    const dirty = (await git(["-C", worktree, "status", "--porcelain"])).stdout.trim();
    if (!dirty) {
      await log({ stage: 4, level: "info", msg: "no changes (already built)" });
      return { mode: cfg.mode, branch, committed: false, sourceKey: key };
    }
    if (dryRun) {
      await log({ stage: 4, level: "info", msg: `[dry-run] would commit+push to ${branch}` });
      return { mode: cfg.mode, branch, committed: false, dryRun: true, sourceKey: key };
    }
    const env = { ...process.env,
      GIT_AUTHOR_NAME: INGEST_AUTHOR_NAME, GIT_AUTHOR_EMAIL: INGEST_AUTHOR_EMAIL,
      GIT_COMMITTER_NAME: INGEST_AUTHOR_NAME, GIT_COMMITTER_EMAIL: INGEST_AUTHOR_EMAIL };
    await git(["-C", worktree, "commit", "-q", "-m",
      `ingest(${cfg.mode}): build artifacts from git push [source ${key.slice(0, 14)}]`], { env });
    const sha = (await git(["-C", worktree, "rev-parse", "HEAD"])).stdout.trim();
    // local file push back to the bare repo — NOT via HTTP receive-pack → emits no git.push event
    await git(["-C", worktree, "push", "origin", `HEAD:refs/heads/${branch}`]);
    // make the container cloneable-by-default: point the bare HEAD at the content branch
    // (push-created bare repos otherwise leave HEAD on an unborn 'master'). Best-effort.
    try { await git(["--git-dir", barePath, "symbolic-ref", "HEAD", `refs/heads/${branch}`]); } catch {}
    await log({ stage: 4, level: "info", msg: `committed ${sha.slice(0, 12)} + pushed to ${branch} (local, no event)` });
    return { mode: cfg.mode, branch, committed: true, commit: sha, sourceKey: key };
  } finally {
    try { fssync.rmSync(worktree, { recursive: true, force: true }); } catch {}
  }
}

// ── enqueue on git.push (reads the BARE repo, no worktree) ──────────────────
async function maybeEnqueue(containerId) {
  let bare;
  try { bare = repoPathForContainer(containerId); } catch (e) { return { skipped: "bad-id" }; }
  if (!fssync.existsSync(bare)) return { skipped: "no-repo" };
  const ref = await contentRef(["--git-dir", bare]); // not necessarily HEAD (see contentRef)
  if (!ref) return { skipped: "empty-repo" };
  const show = async (p) => (await git(["--git-dir", bare, "show", `${ref}:${p}`])).stdout;
  let cfg;
  try { cfg = JSON.parse(await show(".gitchain/ingest.json")); }
  catch { return { skipped: "no-ingest-config" }; }
  if (!["build", "atomize"].includes(cfg.mode)) return { skipped: "bad-mode" };

  // loop guard #1: don't re-ingest our own write-back commit
  try {
    const ae = (await git(["--git-dir", bare, "log", "-1", "--format=%ae", ref])).stdout.trim();
    if (ae === INGEST_AUTHOR_EMAIL) return { skipped: "own-writeback" };
  } catch {}

  // loop guard #2 + idempotency: skip malformed configs (source path missing → every re-push would
  // otherwise spawn a doomed job) and skip when the INPUT fingerprint is unchanged since last build.
  let key;
  try {
    const revAt = async (p) => (await git(["--git-dir", bare, "rev-parse", `${ref}:${p}`])).stdout.trim();
    key = await sourceKey(revAt, cfg);
  } catch {
    return { skipped: "source-path-missing" }; // config points at a path that isn't in the repo
  }
  let built = "";
  try { built = (await show(".gitchain/ingest.built")).trim(); } catch {}
  if (built && built === key) return { skipped: "unchanged-source" };

  const pool = await getPool();
  try {
    const r = await pool.query(
      `INSERT INTO migration_jobs (container_id, kind, triggered_by, status, total_steps)
         SELECT $1, 'ingest', 'git.push', 'queued', 4
         WHERE NOT EXISTS (
           SELECT 1 FROM migration_jobs WHERE container_id = $1 AND status IN ('queued','running'))
       RETURNING id`,
      [containerId]
    );
    return r.rowCount ? { enqueued: r.rows[0].id } : { skipped: "active-job-exists" };
  } catch (e) {
    // A concurrent worker won the race: the partial-unique index (container_id WHERE queued/running)
    // rejects the second insert. The job exists — treat as a benign skip, not an error.
    if (e && e.code === "23505") return { skipped: "race-lost" };
    throw e;
  }
}

// ── job execution (DB wrapper around runChain) ──────────────────────────────
async function appendLog(jobId, entry) {
  const pool = await getPool();
  const e = { ts: new Date().toISOString(), ...entry };
  await pool.query(
    `UPDATE migration_jobs SET log = log || $1::jsonb, current_step = COALESCE($2, current_step) WHERE id = $3`,
    [JSON.stringify([e]), entry.stage ?? null, jobId]
  );
  log0(`job=${jobId.slice(0, 8)} stage=${entry.stage} ${entry.level}: ${entry.msg}`);
}
async function setStatus(jobId, status, extra = {}) {
  const pool = await getPool();
  const fields = ["status = $2"];
  const values = [jobId, status];
  let n = 3;
  if (extra.error !== undefined) { fields.push(`error = $${n++}`); values.push(extra.error); }
  if (extra.output !== undefined) { fields.push(`output = $${n++}::jsonb`); values.push(JSON.stringify(extra.output)); }
  if (extra.finished) {
    fields.push("finished_at = now()");
    fields.push("duration_seconds = EXTRACT(EPOCH FROM (now() - started_at))");
  }
  if (status === "running") fields.push("started_at = COALESCE(started_at, now())");
  await pool.query(`UPDATE migration_jobs SET ${fields.join(", ")} WHERE id = $1`, values);
}
async function runIngest(jobId) {
  const pool = await getPool();
  const j = await pool.query(`SELECT container_id FROM migration_jobs WHERE id = $1`, [jobId]);
  if (j.rowCount !== 1) throw new Error(`job ${jobId} not found`);
  const containerId = j.rows[0].container_id;
  await setStatus(jobId, "running");
  await appendLog(jobId, { stage: 0, level: "info", msg: `ingest ${containerId}` });
  try {
    const bare = repoPathForContainer(containerId);
    const res = await runChain(bare, { log: (e) => appendLog(jobId, e) });
    await setStatus(jobId, "sealed", { output: { kind: "ingest", ...res }, finished: true });
    return res;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    await appendLog(jobId, { stage: 0, level: "error", msg: `ingest failed: ${msg}` });
    await setStatus(jobId, "failed", { error: msg, finished: true });
    throw err;
  }
}
async function pickQueued() {
  const pool = await getPool();
  const r = await pool.query(
    `WITH next AS (
       SELECT id FROM migration_jobs
        WHERE kind = 'ingest' AND status = 'queued'
        ORDER BY triggered_at LIMIT 1
        FOR UPDATE SKIP LOCKED)
     UPDATE migration_jobs SET status = 'running', started_at = now()
      WHERE id IN (SELECT id FROM next) RETURNING id`
  );
  return r.rows[0]?.id ?? null;
}

// ── loops ───────────────────────────────────────────────────────────────────
async function consume() {
  const redis = await getRedis();
  let lastId = (await redis.get(CURSOR_KEY)) || "$";
  log0(`[consume] 0711.events from ${lastId}`);
  while (!STOP) {
    let res;
    try {
      res = await redis.xRead({ key: "0711.events", id: lastId }, { BLOCK: 5000, COUNT: 50 });
    } catch (e) { log0(`[consume] xRead: ${e.message}`); await sleep(2000); continue; }
    if (!res) continue;
    for (const stream of res) {
      for (const m of stream.messages) {
        lastId = m.id;
        const f = m.message || {};
        if (f.type === "git.push" && f.container_id) {
          try {
            const r = await maybeEnqueue(f.container_id);
            log0(`[consume] git.push ${f.container_id} -> ${JSON.stringify(r)}`);
          } catch (e) { log0(`[consume] enqueue error: ${e.message}`); }
        }
      }
    }
    try { await redis.set(CURSOR_KEY, lastId); } catch {}
  }
}
async function work() {
  log0(`[work] poll migration_jobs(kind=ingest) every ${POLL_INTERVAL_MS}ms`);
  while (!STOP) {
    let jobId = null;
    try { jobId = await pickQueued(); } catch (e) { log0(`[work] pick: ${e.message}`); await sleep(POLL_INTERVAL_MS); continue; }
    if (jobId) { try { await runIngest(jobId); } catch (e) { log0(`[work] job ${jobId} failed: ${e.message}`); } }
    else await sleep(POLL_INTERVAL_MS);
  }
}

// ── entry ────────────────────────────────────────────────────────────────────
async function main() {
  const argv = process.argv.slice(2);
  const dryRun = argv.includes("--dry-run");
  const selfIdx = argv.indexOf("--selftest") !== -1 ? argv.indexOf("--selftest") : argv.indexOf("--ingest");
  const onceIdx = argv.indexOf("--once");

  if (selfIdx !== -1) {
    const bare = argv[selfIdx + 1];
    if (!bare) { console.error("usage: --selftest <bare-repo-path> [--dry-run]"); process.exit(2); }
    const res = await runChain(path.resolve(bare), {
      dryRun,
      log: async (e) => log0(`stage ${e.stage} ${e.level}: ${e.msg}`),
    });
    console.log(JSON.stringify({ selftest: true, ...res }, null, 2));
    return;
  }

  const chkIdx = argv.indexOf("--check-enqueue");
  if (chkIdx !== -1) {
    // Ops/test hook: report the enqueue decision (incl. loop guards). Only touches the DB if the
    // guards pass — a skipped result (no-repo / no-ingest-config / own-writeback / unchanged-source)
    // returns before any DB connection, so this runs without pg/DB in those cases.
    const cid = argv[chkIdx + 1];
    if (!cid) { console.error("usage: --check-enqueue <container-id>"); process.exit(2); }
    const r = await maybeEnqueue(cid);
    console.log(JSON.stringify({ checkEnqueue: cid, ...r }));
    if (_pool) await _pool.end();
    return;
  }

  if (onceIdx !== -1) {
    const cid = argv[onceIdx + 1];
    if (!cid) { console.error("usage: --once <container-id>"); process.exit(2); }
    const enq = await maybeEnqueue(cid);
    log0(`[once] enqueue ${cid} -> ${JSON.stringify(enq)}`);
    const jobId = await pickQueued();
    if (jobId) { const r = await runIngest(jobId); console.log(JSON.stringify({ once: cid, job: jobId, ...r }, null, 2)); }
    else log0("[once] nothing to run");
    await (_pool && _pool.end());
    return;
  }

  process.on("SIGINT", () => { STOP = true; log0("SIGINT — draining"); });
  process.on("SIGTERM", () => { STOP = true; });
  const loops = [work()];
  if (!argv.includes("--work-only")) loops.push(consume());
  await Promise.all(loops);
}

main().catch((e) => { log0(`fatal: ${e.stack || e.message}`); process.exit(1); });
