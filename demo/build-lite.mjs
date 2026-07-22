#!/usr/bin/env node
/** Baut aus einem docs-Ordner einen MODELL-FREIEN Container-Worktree — jedes Atom mit Provenienz:
 *  - .md/.txt : ein Atom je Absatz, Provenienz = Datei + Zeilenbereich (reader "lite-text")
 *  - .pdf     : Glyph-Text-Layer-Leser (toolkit/glyph_pdf.py), Provenienz = Datei + SEITE + y
 *               (reader "glyph-textlayer") — braucht python3 + numpy + pymupdf, KEINE Modelle.
 *  Ausgabe: data/chunks.jsonl (sha-adressiert) + .brain/index.json (store:"none", BM25-suchbar). */
import { readdirSync, readFileSync, writeFileSync, mkdirSync, copyFileSync, existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { join, dirname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";

const HERE = dirname(fileURLToPath(import.meta.url));
const TOOLKIT = process.env.TOOLKIT_DIR || join(HERE, "..", "toolkit");
const PY = process.env.PYTHON_BIN || "python3";
const [src, out] = process.argv.slice(2);
if (!src || !out) { console.error("usage: build-lite.mjs <docsDir> <outDir>"); process.exit(1); }
for (const d of ["docs", "data", ".brain"]) mkdirSync(join(out, d), { recursive: true });

const chunks = [];
const push = (c) => {
  c.id = createHash("sha1").update(c.text).digest("hex").slice(0, 16);
  if (!chunks.some((x) => x.id === c.id)) chunks.push(c);
};

const PDF_READER = `
import sys, json
sys.path.insert(0, ${JSON.stringify(TOOLKIT)})
import glyph_pdf, fitz
path = sys.argv[1]
doc = fitz.open(path)
if not "".join(p.get_text() for p in doc).strip():
    print(json.dumps({"scan": True, "rows": []})); raise SystemExit
rows = []
for pno, page in enumerate(doc):
    g, recs = glyph_pdf.from_pdf(page)
    H, W = g.shape
    for row in glyph_pdf.detect_table(recs, W, H).get("rows", []):
        cells = [c for c in row.get("cells", []) if c]
        if not cells: continue
        t = (" | ".join(cells) if len(cells) > 1 else cells[0]).strip()
        if (" | " in t) or len(t) >= 25 or any(ch.isdigit() for ch in t):
            rows.append({"page": pno + 1, "y": int(row.get("y", 0)), "text": t})
print(json.dumps({"scan": False, "rows": rows}))
`;

let pdfOk = null;
for (const f of readdirSync(src).sort()) {
  const full = join(src, f);
  if (/\.(md|txt)$/i.test(f)) {
    copyFileSync(full, join(out, "docs", f));
    const lines = readFileSync(full, "utf8").split("\n");
    let start = 0, buf = [];
    const flush = (endIdx) => {
      const text = buf.join("\n").trim();
      if (text.length > 20)
        push({ source: f, line_from: start + 1, line_to: endIdx, reader: "lite-text", text });
      buf = [];
    };
    lines.forEach((l, i) => {
      if (l.trim() === "") { flush(i); start = i + 1; } else { if (!buf.length) start = i; buf.push(l); }
    });
    flush(lines.length);
  } else if (/\.pdf$/i.test(f)) {
    if (pdfOk === null) {
      try { execFileSync(PY, ["-c", "import numpy, fitz"], { stdio: "ignore" }); pdfOk = existsSync(join(TOOLKIT, "glyph_pdf.py")); }
      catch { pdfOk = false; }
      if (!pdfOk) console.error(`! PDFs uebersprungen — fuer Glyph-Provenienz: pip3 install numpy pymupdf (Reader: ${TOOLKIT}/glyph_pdf.py)`);
    }
    if (!pdfOk) continue;
    copyFileSync(full, join(out, "docs", f));
    try {
      const r = JSON.parse(execFileSync(PY, ["-c", PDF_READER, full], { encoding: "utf8", maxBuffer: 64 << 20 }));
      if (r.scan) { console.error(`! ${f}: kein Text-Layer (Scan) — im Lite-Modus uebersprungen`); continue; }
      for (const row of r.rows)
        for (let i = 0; i < row.text.length; i += 460)
          push({ source: f, page: row.page, y: row.y, reader: "glyph-textlayer", text: row.text.slice(i, i + 480) });
    } catch (e) { console.error(`! ${f}: ${String((e && e.message) || e).slice(0, 120)}`); }
  }
}
writeFileSync(join(out, "data", "chunks.jsonl"), chunks.map((c) => JSON.stringify(c)).join("\n") + "\n");
writeFileSync(join(out, ".brain", "index.json"), JSON.stringify(
  { model: null, dims: 0, count: chunks.length, store: "none", reader: "glyph", order: chunks.map((c) => c.id) }, null, 2));
const pdfs = chunks.filter((c) => c.reader === "glyph-textlayer").length;
console.log(`${chunks.length} Atome -> ${out} (modell-frei, BM25-suchbar; davon ${pdfs} Glyph-seitenzitiert)`);
