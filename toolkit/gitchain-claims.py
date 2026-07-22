#!/usr/bin/env python3
"""gitchain-claims — typisierte Claims aus Text-Layer-PDFs (LStB-Profil) als Container-Worktree.

Liest jedes PDF ueber fitz rawdict (derselbe Glyph-Textlayer-Pfad wie glyph_pdf, 200dpi-Skala
sc=200/72), baut Zeilenobjekte {text,x,y,x2,y2,page} und extrahiert getypte Claims mit der aus
claims_eval2 validierten LStB-Logik: Komma-Reparatur (RE_FIX), Euro/Cent-Merge, Zeilen-Anker
(RE_ROW, akzeptiert [.,] nach der Nummer), Keyword-Paarung mit Ausschluessen, positionale
Paarung ueber die RECHTE Kante (x2) der Wertspalte, Arbeitgeber-Anker (employer_of).

Zeugen-Merge ist steckbar: merge_witnesses(witnesses) nimmt eine Liste [(source_name, vals_dict)]
in Prioritaetsreihenfolge — der erste Zeuge liefert den Wert, jeder weitere mit Ziffern-Match
(exakt oder Suffix >= 4 Ziffern) zaehlt als Bestaetigung (witnesses += 1). Zusaetzliche Zeugen
(OCR, Glypher-Recs, ...) haengen spaeter einfach weitere (name, vals)-Tupel an die Liste.

Usage:  gitchain-claims.py <pdfDir> <outDir> [--profile lstb]
Output: <outDir>/docs/*.pdf (Kopien), data/chunks.jsonl (1 Claim-Atom pro Zeile),
        .brain/index.json {"model":null,"dims":0,"count":N,"store":"none","kind":"claims","order":[ids]}.
Scans (kein Text-Layer) werden mit stderr-Notiz uebersprungen."""
import argparse, glob, hashlib, json, os, re, shutil, sys

import fitz

SC = 200 / 72.0
GAP = 30  # px @200dpi: Spalten-Gutter trennt Segmente, Wortabstaende (~4-8px) nicht

CODE2LABEL = {"E0200204": "Bruttoarbeitslohn", "E0200304": "einbehaltene Lohnsteuer",
              "E0200404": "Solidaritaetszuschlag", "E0200504": "Kirchensteuer",
              "E2000401": "Rentenversicherung Arbeitnehmeranteil",
              "E2001203": "Krankenversicherung Arbeitnehmerbeitraege",
              "E2001505": "Pflegeversicherung Arbeitnehmerbeitraege",
              "E2004403": "Arbeitslosenversicherung Arbeitnehmerbeitraege"}
ROW2CODE = {"3": "E0200204", "4": "E0200304", "5": "E0200404", "6": "E0200504", "7": "E0200604",
            "23": "E2000401", "25": "E2001203", "26": "E2001505", "27": "E2004403"}
ORDINAL = ["3", "4", "5", "6", "7"]          # Zeile 3-7 = Kopfblock in Formularreihenfolge
RE_FIX = re.compile(r"\b(\d{1,3}(?:\.\d{3})+)(\d{2})\b")
RE_FIXSP = re.compile(r"\b(\d{1,3}(?:\.\d{3})+) (\d{2})\b")  # Euro/Cent im selben Segment
RE_CENTSEG = re.compile(r"^(\d{1,5}) (\d{2})$")              # nacktes "150 00"-Segment
RE_EURO = re.compile(r"^\d{1,3}(?:\.\d{3})*$|^\d{1,5}$")
RE_CENT = re.compile(r"^\d{2}$")
RE_ROW = re.compile(r"^\s*(\d{1,2})[.,]\s*[A-Za-zÄÖÜäöü]")  # Synthetik-Fonts rendern "3," statt "3."
RE_MONEYTOK = re.compile(r"^\d{1,3}(?:\.\d{3})*,\d{2}$|^\d{1,5},\d{2}$")
RE_FIRM = re.compile(r"([A-ZÄÖÜ][\w.&ÄÖÜäöüß-]*(?:\s+[A-ZÄÖÜ&][\w.&ÄÖÜäöüß-]*){0,5}\s+(?:GmbH(?:\s*&\s*Co\.?\s*KG)?|OHG|AG|KG|UG|e\.V\.|BV|SE))")
RE_YEAR = re.compile(r"Lohnsteuerbescheinigung\D{0,30}\b(20\d{2})\b")


class Line:
    __slots__ = ("text", "x", "y", "x2", "y2", "page")
    def __init__(self, text, x, y, x2, y2, page):
        self.text = text; self.x = x; self.y = y; self.x2 = x2; self.y2 = y2; self.page = page


class Doc:
    def __init__(self, file, lines, npages):
        self.file = file; self.lines = lines; self.npages = npages; self.text = ""
    def page_lines(self, pg):
        return [l for l in self.lines if l.page == pg]


def nd(v):
    return re.sub(r"\D", "", str(v or ""))


def read_doc(path):
    """PDF -> Doc: rawdict-Zeilen, an x-Luecken > GAP in Segmente geteilt (Wertspalten werden
    eigene Zeilenobjekte, Labels bleiben ganz). Boxen aus Zeichen-bboxen, 200dpi. None = Scan."""
    pdf = fitz.open(path)
    if not "".join(p.get_text() for p in pdf).strip():
        return None
    lines = []
    for pno, page in enumerate(pdf, 1):
        for b in page.get_text("rawdict")["blocks"]:
            for l in b.get("lines", []):
                segs = []  # (text, x, y, x2, y2)
                for s in l.get("spans", []):
                    cs = [c for c in s.get("chars", []) if c["c"].strip()]
                    if not cs:
                        continue
                    t = "".join(c["c"] for c in s.get("chars", [])).strip()
                    x = min(c["bbox"][0] for c in cs) * SC; x2 = max(c["bbox"][2] for c in cs) * SC
                    y = min(c["bbox"][1] for c in cs) * SC; y2 = max(c["bbox"][3] for c in cs) * SC
                    if segs and x - segs[-1][3] <= GAP:
                        p = segs[-1]
                        segs[-1] = (p[0] + " " + t, p[1], min(p[2], y), max(p[3], x2), max(p[4], y2))
                    else:
                        segs.append((t, x, y, x2, y2))
                for t, x, y, x2, y2 in segs:
                    lines.append(Line(t, int(x), int(y), int(x2), int(y2), pno))
    return Doc(os.path.basename(path), lines, len(pdf))


def repair(d):
    for l in d.lines:
        l.text = RE_FIX.sub(r"\1,\2", l.text)
        l.text = RE_FIXSP.sub(r"\1,\2", l.text)
        m = RE_CENTSEG.match(l.text.strip())
        if m:
            l.text = f"{m.group(1)},{m.group(2)}"
    for pg in range(1, d.npages + 1):
        pls = d.page_lines(pg)
        cents = [l for l in pls if RE_CENT.fullmatch(l.text.strip())]
        for e in pls:
            if not RE_EURO.fullmatch(e.text.strip()):
                continue
            for c in cents:
                if c is not e and c.x > e.x and abs(c.y - e.y) <= 8 and (e.x2 is None or 0 < c.x - e.x2 <= 130):
                    e.text = f"{e.text.strip()},{c.text.strip()}"; c.text = ""
                    break
    d.text = "\n".join(l.text for l in d.lines)


def employer_of(d):
    """Arbeitgeber deterministisch. a) Anker 'des Arbeitgebers' -> naechste Zeile(n) darunter.
    b) Tabellenform 'Name des Arbeitgebers | X' in derselben Zeile. c) RE_FIRM."""
    for pg in range(1, d.npages + 1):
        pls = sorted(d.page_lines(pg), key=lambda l: (l.y, l.x))
        for ln in pls:
            t = ln.text
            if "Name des Arbeitgebers" in t:
                after = t.split("Name des Arbeitgebers", 1)[1].strip(" |:")
                if len(after) > 3:
                    return after[:60]
                right = [o for o in pls if abs(o.y - ln.y) <= 12 and o.x > ln.x + 40 and len(o.text.strip()) > 3]
                if right:
                    return sorted(right, key=lambda o: o.x)[0].text.strip()[:60]
            if "Anschrift und Steuernummer des Arbeitgebers" in t or re.search(r"Anschrift.*Arbeitgebers", t):
                below = [o for o in pls if ln.y < o.y <= ln.y + 90 and abs(o.x - ln.x) < 250 and len(o.text.strip()) > 3]
                for o in sorted(below, key=lambda o: o.y):
                    cand = o.text.strip()
                    if not re.match(r"^\d", cand) and "Steuernummer" not in cand:
                        return cand[:60]
    m = RE_FIRM.search(d.text)
    return m.group(1)[:60] if m else None


def positional_pairing(d, vals):
    """Merged-Row-Variante — Zeilen-Labels 3..7 monotone y-Paarung mit der EUR-Wertspalte
    (dominante x-Spalte reiner Geldtoken). Betragsspalte ist RECHTSBUENDIG -> nach rechter
    Kante (x2) clustern, nicht nach x. Nur fuer Codes ohne Wert."""
    need = [n for n in ORDINAL if ROW2CODE.get(n) and ROW2CODE[n] not in vals]
    if not need:
        return
    for pg in range(1, d.npages + 1):
        pls = d.page_lines(pg)
        labels = []
        for ln in pls:
            m = RE_ROW.match(ln.text)
            if m and m.group(1) in ORDINAL:
                labels.append((m.group(1), ln.y))
        moneys = [(l.y, (l.x2 if l.x2 is not None else l.x), l.text.strip())
                  for l in pls if RE_MONEYTOK.fullmatch(l.text.strip())]
        if len(labels) < 3 or len(moneys) < 3:
            continue
        xs = sorted(m[1] for m in moneys)
        col_x = xs[len(xs) // 2]
        col = sorted([m for m in moneys if abs(m[1] - col_x) <= 40], key=lambda m: m[0])
        labels.sort(key=lambda l: l[1])
        used = set()
        for num, ly in labels:
            code = ROW2CODE[num]
            if code in vals:
                continue
            best = None
            for j, (my, mx, mt) in enumerate(col):
                if j in used or my < ly - 15:
                    continue
                dy = abs(my - ly)
                if dy <= 220 and (best is None or dy < best[0]):
                    best = (dy, j, mt, my)
            if best:
                used.add(best[1])
                vals[code] = (best[2], pg, best[3])
        return


def keyword_pairing(d, vals):
    """Variante ohne (intakte) Zeilennummern: tolerante Label-Keywords -> naechster
    Geldwert rechts/darunter. Ausschluss der Zeilen 11/12 (von 9. und 10.)."""
    KW = [("E0200204", ("rutto",), ()), ("E0200304", ("ohnsteuer",), ("und10", "9und")),
          ("E0200404", ("olidar",), ("und10", "9und")),
          ("E0200504", ("irchensteuer",), ("hegatt", "ebenspartner"))]
    compact = lambda t: re.sub(r"\s+", "", t.lower())
    for pg in range(1, d.npages + 1):
        pls = d.page_lines(pg)
        moneys = [l for l in pls if RE_MONEYTOK.fullmatch(l.text.strip())]
        if not moneys:
            continue
        used = set()
        for code, kws, excl in KW:
            if code in vals:
                continue
            labs = [ln for ln in pls if all(k in compact(ln.text) for k in kws)
                    and not any(x in compact(ln.text) for x in excl)]
            for ln in sorted(labs, key=lambda l: l.y):
                best = None
                for m in moneys:
                    if id(m) in used or m.x < ln.x + 150:
                        continue
                    dy = m.y - ln.y
                    if -15 <= dy <= 70 and (best is None or abs(dy) < best[0]):
                        best = (abs(dy), m)
                if best:
                    used.add(id(best[1]))
                    vals[code] = (best[1].text.strip(), pg, best[1].y)
                    break


def extract_vals(d):
    """Alle drei Paarungswege in Reihenfolge Zeilen-Anker -> positional -> Keyword.
    vals: {code: (raw, page, y)} — erster Treffer je Code gewinnt."""
    vals = {}
    for pg in range(1, d.npages + 1):
        pls = d.page_lines(pg)
        ms = [l for l in pls if RE_MONEYTOK.fullmatch(l.text.strip())]
        for ln in pls:
            mm = RE_ROW.match(ln.text)
            code = ROW2CODE.get(mm.group(1)) if mm else None
            if not code or code in vals:
                continue
            cands = [v for v in ms if v.x > ln.x + 60 and abs(v.y - ln.y) <= 14]
            if len(cands) == 1:
                vals[code] = (cands[0].text.strip(), pg, cands[0].y)
            else:
                toks_ = [t for t in ln.text.split() if RE_MONEYTOK.fullmatch(t)]
                if len(toks_) == 1:
                    vals[code] = (toks_[0], pg, ln.y)
    positional_pairing(d, vals)
    keyword_pairing(d, vals)
    return vals


def merge_witnesses(witnesses):
    """Steckbarer Zeugen-Merge. witnesses = [(source_name, vals_dict)] in Prioritaets-
    reihenfolge, vals_dict = {code: (raw, page, y)}. Politik (empirisch beste aus
    claims_eval2): erster Zeuge traegt den Wert ein, jeder weitere mit Ziffern-Match
    (exakt oder Suffix >= 4 Ziffern) bestaetigt nur (witnesses += 1), ueberschreibt nie.
    Rueckgabe {code: {"value","page","y","witnesses","sources"}}. Weitere Zeugen (OCR,
    Glypher) haengen spaeter einfach zusaetzliche (name, vals)-Tupel an die Liste."""
    out = {}
    for src, vals in witnesses:
        for code, (raw, page, y) in vals.items():
            e = out.get(code)
            if e is None:
                out[code] = {"value": raw, "page": page, "y": y, "witnesses": 1, "sources": [src]}
            else:
                g = nd(raw)
                if nd(e["value"]) == g or (len(g) >= 4 and nd(e["value"]).endswith(g)):
                    e["witnesses"] += 1; e["sources"].append(src)
    return out


def main():
    ap = argparse.ArgumentParser(description="Typed-Claims-Container aus Text-Layer-PDFs (LStB)")
    ap.add_argument("pdfDir"); ap.add_argument("outDir")
    ap.add_argument("--profile", default="lstb", choices=["lstb"])
    args = ap.parse_args()
    cdir = os.path.abspath(args.outDir)
    os.makedirs(f"{cdir}/docs", exist_ok=True)
    os.makedirs(f"{cdir}/data", exist_ok=True)
    os.makedirs(f"{cdir}/.brain", exist_ok=True)

    pdfs = sorted(glob.glob(os.path.join(args.pdfDir, "*.pdf")))
    claims, scans, wage_docs = [], 0, 0
    for path in pdfs:
        d = read_doc(path)
        if d is None:
            print(f"SKIP (Scan, kein Text-Layer): {os.path.basename(path)}", file=sys.stderr)
            scans += 1
            continue
        repair(d)
        emp = employer_of(d)
        merged = merge_witnesses([("glyph-textlayer", extract_vals(d))])
        shutil.copy2(path, f"{cdir}/docs/{d.file}")
        my = RE_YEAR.search(d.text)
        year = my.group(1) if my else "2024"
        wage_docs += "E0200204" in merged
        for code in sorted(merged):
            label = CODE2LABEL.get(code)
            if not label:  # z.B. E0200604 (Zeile 7) nicht im Feld-Mapping
                continue
            e = merged[code]
            text = f"{emp or 'Arbeitgeber unbekannt'} — Lohnsteuerbescheinigung {year} — {label} ({code}): {e['value']} EUR"
            claims.append({"id": hashlib.sha1(text.encode()).hexdigest()[:16], "text": text,
                           "source": d.file, "page": int(e["page"]), "y": int(e["y"]),
                           "reader": "glyph-textlayer", "code": code, "value": e["value"],
                           "employer": emp, "witnesses": e["witnesses"]})

    with open(f"{cdir}/data/chunks.jsonl", "w") as f:
        for c in claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    json.dump({"model": None, "dims": 0, "count": len(claims), "store": "none",
               "kind": "claims", "order": [c["id"] for c in claims]},
              open(f"{cdir}/.brain/index.json", "w"))
    print(f"gitchain-claims: {len(pdfs)} PDFs, {scans} Scans uebersprungen, "
          f"{len(claims)} Claims, {wage_docs} Docs mit Bruttoarbeitslohn -> {cdir}", flush=True)


if __name__ == "__main__":
    main()
