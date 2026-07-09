#!/usr/bin/env python3
"""gitchain-atomize — create a *fact* container of evidence-verified atoms.

Grounding mode 2 of 2 (see gitchain-build.py for mode 1 — a semantic brain).
Where `build` embeds unstructured text for cosine recall, `atomize` takes STRUCTURED
claims and checks each one against document word-boxes, so every fact carries its own
page/box proof. Recall is exact fact + evidence, not similarity.

For each claim it emits one atom with a verdict:
  verified              value found in a box within the label's y-band (proof attached)
  value-not-at-label    label appears but the value is not on the same line/band
  in-text-not-in-line   value is somewhere in the doc text but not near the label
  not-in-doc            label word not found in any evidence doc
  no-evidence-doc       claim has no evidence documents
  value-null-suspect    numeric claim whose value is 0 (a data-quality finding, unverifiable)
  unverifiable-text     free-text claim (kind "L") — not machine-checkable in v1

Inputs are plain JSONL (no database coupling):
  --claims   claims.jsonl   one per line:
             {id?, subject, attribute, value, value2?, unit?, kind?("A"|"N"|"R"|"L"), docs:[doc_id,...]}
             kind: A=enumerated/text value, N=number, R=range (value..value2), L=free text.
  --boxes    boxes.jsonl    word-boxes from your reader:
             {doc, page, x0, y0, x1, y1, text}   (coordinates in points, origin top-left)
  --fulltext fulltext.jsonl (optional) {doc, text} for the cheap "is the value in the doc" prefilter;
             if omitted it is derived by joining that doc's boxes.

Outputs a git container:
  data/atoms.jsonl   one atom per claim (with proof where verified)
  report.json        verdict counts + method
  container.json     manifest

Usage:
  gitchain-atomize.py --claims claims.jsonl --boxes boxes.jsonl \
      [--fulltext fulltext.jsonl] [--container out-dir] [--id container:my-facts] [--no-commit]
"""
import argparse, collections, hashlib, json, os, re, subprocess, sys, time


def num_variants(v):
    v = (v or "").strip()
    if not v:
        return []
    out = {v}
    if v.endswith(".0"):
        out.add(v[:-2])
    out.add(v.replace(".", ","))
    if re.fullmatch(r"\d+\.\d+", v):
        out.add(v.split(".")[0] + "," + v.split(".")[1])
    return [x for x in out if x]


def text_has(text, variants, numeric):
    if numeric:
        return any(re.search(r"(?<![\d,.])" + re.escape(x) + r"(?![\d])", text) for x in variants)
    return any(x.lower() in text for x in variants)


def find_box_proof(boxes_by_doc, doc, distinctive, variants, unit):
    """Find a label box, then the value box in its y-band. Short values (<=2 chars) require a
    unit in the band or the value directly to the right of the label on the same line."""
    boxes = boxes_by_doc.get(doc, [])
    label_hits = [b for b in boxes if distinctive.lower() in b["text"].lower()][:20]
    for lb in label_hits:
        pg, ly0, lx1 = lb["page"], lb["y0"], lb["x1"]
        band = [b for b in boxes if b["page"] == pg and ly0 - 4 <= b["y0"] <= ly0 + 30]
        band.sort(key=lambda b: (b["y0"], b["x0"]))
        band_words = {b["text"] for b in band}
        for wb in band:
            wt = wb["text"]
            if not any(re.fullmatch(r"(?:\D?)" + re.escape(x) + r"(?:\D?)", wt) or wt == x
                       for x in variants):
                continue
            if min(len(x) for x in variants) <= 2:
                same_line_right = abs(wb["y0"] - ly0) < 4 and wb["x0"] >= lx1 and (wb["x0"] - lx1) < 250
                unit_in_band = unit and unit in band_words
                if not (unit_in_band or same_line_right):
                    continue
            return {"page": pg, "line": " ".join(b["text"] for b in band)[:140],
                    "label_box": [round(lb[k], 1) for k in ("x0", "y0", "x1", "y1")],
                    "label_word": lb["text"],
                    "value_box": [round(wb[k], 1) for k in ("x0", "y0", "x1", "y1")],
                    "value_word": wt}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--fulltext", default="")
    ap.add_argument("--container", default=".")
    ap.add_argument("--id", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--no-commit", action="store_true")
    args = ap.parse_args()
    cdir = os.path.abspath(args.container)
    os.makedirs(f"{cdir}/data", exist_ok=True)

    claims = [json.loads(l) for l in open(args.claims, encoding="utf-8") if l.strip()]
    boxes_by_doc = collections.defaultdict(list)
    for l in open(args.boxes, encoding="utf-8"):
        if l.strip():
            b = json.loads(l); boxes_by_doc[b["doc"]].append(b)
    if args.fulltext:
        texts = {j["doc"]: (j.get("text") or "").lower()
                 for j in (json.loads(l) for l in open(args.fulltext, encoding="utf-8") if l.strip())}
    else:
        texts = {d: " ".join(b["text"] for b in bs).lower() for d, bs in boxes_by_doc.items()}
    print(f"gitchain-atomize: {len(claims)} claims · {len(boxes_by_doc)} docs with boxes", flush=True)

    t0 = time.perf_counter()
    atoms, stats = [], collections.Counter()
    for n, c in enumerate(claims, 1):
        kind = c.get("kind") or ("N" if str(c.get("value", "")).replace(".", "").isdigit() else "A")
        val, val2, unit = str(c.get("value", "")), c.get("value2"), c.get("unit")
        docs = c.get("docs", [])
        atom = {"typ": "fact-atom", "subject": c.get("subject"), "attribute": c.get("attribute"),
                "value": c.get("value"), "value2": val2, "unit": unit, "kind": kind}
        if kind == "L":
            atom["verdict"] = "unverifiable-text"
        elif kind in ("N", "R") and val.strip() in ("0", "0.0", "0,0"):
            atom["verdict"] = "value-null-suspect"
        elif not docs:
            atom["verdict"] = "no-evidence-doc"
        else:
            numeric = kind in ("N", "R")
            variants = (num_variants(val) + (num_variants(val2) if kind == "R" else [])) if numeric \
                else ([val] if val else [])
            words = re.findall(r"[A-Za-zÄÖÜäöüß]{4,}", c.get("attribute", ""))
            distinctive = max(words, key=len) if words else (c.get("attribute") or "")
            verdict, proof = "not-in-doc", None
            for doc in docs:
                text = texts.get(doc, "")
                if distinctive and distinctive.lower() not in text:
                    continue
                if verdict == "not-in-doc":
                    verdict = "value-not-at-label"
                if not text_has(text, variants, numeric):
                    continue
                p = find_box_proof(boxes_by_doc, doc, distinctive, variants, unit)
                if p:
                    verdict = "verified"
                    proof = {"doc": doc, **p, "value_source": "claims"}
                    break
                verdict = "in-text-not-in-line"
            atom["verdict"] = verdict
            if proof:
                atom["proof"] = proof
        atom["atom_id"] = hashlib.sha256(
            json.dumps(atom, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        atoms.append(atom); stats[atom["verdict"]] += 1
        if n % 3000 == 0:
            print(f"  {n}/{len(claims)}  {dict(stats)}", flush=True)

    with open(f"{cdir}/data/atoms.jsonl", "w") as f:
        for a in atoms:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    report = {"claims": len(claims), "verdicts": dict(stats),
              "runtime_s": round(time.perf_counter() - t0, 1),
              "method": "claim value × document word-boxes; label = longest attribute word, "
                        "value in y-band label-4 .. label+30 pt"}
    json.dump(report, open(f"{cdir}/report.json", "w"), ensure_ascii=False, indent=2)
    json.dump({"id": args.id or f"container:{os.path.basename(cdir)}", "type": "atoms",
               "title": args.title or os.path.basename(cdir),
               "claims_source": os.path.basename(args.claims),
               "evidence_source": os.path.basename(args.boxes),
               "atom_count": len(atoms), "verdicts": dict(stats),
               "recall": "exact fact + attached page/box evidence"},
              open(f"{cdir}/container.json", "w"), ensure_ascii=False, indent=2)

    if not args.no_commit:
        if not os.path.isdir(f"{cdir}/.git"):
            subprocess.run(["git", "-C", cdir, "init", "-q"], check=True)
            subprocess.run(["git", "-C", cdir, "symbolic-ref", "HEAD", "refs/heads/main"], check=True)
        subprocess.run(["git", "-C", cdir, "add", "-A"], check=True)
        subprocess.run(["git", "-C", cdir, "-c", "user.name=gitchain-atomize",
                        "-c", "user.email=atomize@localhost", "commit", "-q", "-m",
                        f"atoms: {len(atoms)} claims verified against document boxes "
                        f"({stats.get('verified', 0)} verified)"], check=True)

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"DONE — container: {cdir}", flush=True)


if __name__ == "__main__":
    main()
