#!/usr/bin/env node
/** Baut aus einem docs-Ordner (.md/.txt) einen MODELL-FREIEN Container-Worktree:
 *  data/chunks.jsonl (ein Atom je Absatz, sha-adressiert) + .brain/index.json (store:"none").
 *  Kein Python, keine Embeddings — die Suche laeuft ueber den BM25-Pfad des Servers. */
import { readdirSync, readFileSync, writeFileSync, mkdirSync, copyFileSync } from "node:fs";
import { join, basename } from "node:path";
import { createHash } from "node:crypto";

const [src, out] = process.argv.slice(2);
if (!src || !out) { console.error("usage: build-lite.mjs <docsDir> <outDir>"); process.exit(1); }
mkdirSync(join(out, "docs"), { recursive: true });
mkdirSync(join(out, "data"), { recursive: true });
mkdirSync(join(out, ".brain"), { recursive: true });

const chunks = [];
for (const f of readdirSync(src).filter((f) => /\.(md|txt)$/i.test(f)).sort()) {
  copyFileSync(join(src, f), join(out, "docs", f));
  const paras = readFileSync(join(src, f), "utf8").split(/\n\s*\n/).map((p) => p.trim()).filter((p) => p.length > 20);
  for (const text of paras) {
    const id = createHash("sha1").update(text).digest("hex").slice(0, 16);
    if (!chunks.some((c) => c.id === id)) chunks.push({ id, source: basename(f), text });
  }
}
writeFileSync(join(out, "data", "chunks.jsonl"), chunks.map((c) => JSON.stringify(c)).join("\n") + "\n");
writeFileSync(join(out, ".brain", "index.json"), JSON.stringify(
  { model: null, dims: 0, count: chunks.length, store: "none", order: chunks.map((c) => c.id) }, null, 2));
console.log(`${chunks.length} Atome aus ${src} -> ${out} (modell-frei, BM25-suchbar)`);
