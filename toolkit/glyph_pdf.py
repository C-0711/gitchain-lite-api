#!/usr/bin/env python3
"""Glyphs Text-Layer-Reader — EINE Quelle, von server.py UND vom Batch genutzt.
Das IST Glyphs `from_pdf` (Zeichen-Boxen aus dem PDF-Text-Layer) + `detect_table` (Box-basierte
Tabellen-Rekonstruktion). Kein Substitut-Reader: Glyph nutzt für den Text-Layer ohnehin fitz —
dieser Code ist genau dieser Glyph-Pfad, nur importierbar (so parallelisierbar, ohne Glyph zu umgehen).
Das MLP (`from_image`) bleibt in server.py und greift nur bei Scans/Fotos."""
import numpy as np, fitz

DPI = 200

def from_pdf(page, dpi=DPI):
    sc=dpi/72.0
    pm=page.get_pixmap(dpi=dpi);g=np.frombuffer(pm.samples,np.uint8).reshape(pm.h,pm.w,pm.n)[:,:,0];ink=(g<160).astype(np.uint8);recs=[]
    for bi,b in enumerate(page.get_text("rawdict")["blocks"]):
        for li,l in enumerate(b.get("lines",[])):
            flat=[(c,s) for s in l.get("spans",[]) for c in s.get("chars",[]) if c["c"].strip()];wpos=0
            for k,(c,s) in enumerate(flat):
                ch=c["c"];x0,y0,x1,y1=[v*sc for v in c["bbox"]];cr=ink[int(y0):int(y1)+1,int(x0):int(x1)+1]
                # Ink-Check verwirft NUR unsichtbaren Text — bewiesen (Audit): weiße/Hint-/OCR-Overlay-Glyphen
                # haben inkSum==0 (invisible Overlays 600/600 + 214/214 = 0). Schwelle daher >=1 statt >=3:
                # mit >=3 fielen sichtbare Kleinst-Glyphen heraus (Dezimal-/Datumspunkte, ~2 Tinten-Pixel) —
                # '.' war mit 7167 das häufigste fehlende Zeichen, ~6000 davon echte Dezimalpunkte (Datenintegrität).
                # Null-Breite-Ligatur-Komponenten (Breite/Höhe <1px) überspringen den Check ohnehin und bleiben.
                if (x1-x0)>=1.0 and (y1-y0)>=1.0 and not(cr.size and cr.sum()>=1):continue
                if k>0 and (flat[k][0]["bbox"][0]-flat[k-1][0]["bbox"][2])*sc>0.18*(y1-y0): wpos=0
                recs.append({"label":ch,"x":int(x0),"y":int(y0),"w":int(x1-x0),"h":int(y1-y0),"block":bi,"zeile":li,"wpos":wpos,
                    "kpi":{"label":ch,"font":s["font"].split("+")[-1],"size_pt":round(s["size"],1),
                           "links":(flat[k-1][0]["c"] if k>0 else "-"),"rechts":(flat[k+1][0]["c"] if k+1<len(flat) else "-"),"quelle":"Text-Layer"}})
                wpos+=1
    # Leerzeichen: Wortgrenzen aus den WORTBOXEN (page.get_text("words") — dieselbe Text-Layer-Quelle,
    # die Glyph ohnehin nutzt). Robust auch bei condensed Fonts (Zeichen-Ebene hat keine Space-Glyphen,
    # x-Lücken zu klein). Pro Wortbox: linkestes Zeichen -> wpos=0 (Wortanfang), Rest -> wpos=1.
    if recs:
        rx=np.array([r["x"] for r in recs],np.float32); ryc=np.array([r["y"]+r["h"]/2.0 for r in recs],np.float32)
        for w in page.get_text("words"):
            wx0,wy0,wx1,wy1=[v*sc for v in w[:4]]
            idx=np.nonzero((rx>=wx0-1)&(rx<=wx1+1)&(ryc>=wy0-2)&(ryc<=wy1+2))[0]
            if idx.size:
                st=int(idx[np.argmin(rx[idx])])
                for j in idx: recs[int(j)]["wpos"]=1
                recs[st]["wpos"]=0
    # Sub-/Superskript (additiv): from_pdf hat size_pt + Box je Glyph. Kleine, vertikal
    # versetzte Glyphen als 'sub'/'sup' markieren — reine Layout-Heuristik pro Zeile,
    # Label + Reihenfolge bleiben unverändert (null Regressionsrisiko für Belege).
    import collections as _c
    _ln=_c.defaultdict(list)
    for _i,_r in enumerate(recs): _ln[(_r["block"],_r["zeile"])].append(_i)
    for _idxs in _ln.values():
        _sz=[recs[i]["kpi"]["size_pt"] for i in _idxs]
        _med=float(np.median(_sz)) if _sz else 0.0
        _big=[i for i in _idxs if recs[i]["kpi"]["size_pt"]>=0.85*_med] or _idxs
        _base=float(np.median([recs[i]["y"]+recs[i]["h"] for i in _big]))     # Grundlinie (Unterkante)
        _mh=float(np.median([recs[i]["h"] for i in _big])) or 1.0
        for i in _idxs:
            r=recs[i]; r["kpi"]["hoch"]=""
            if _med and r["kpi"]["size_pt"]>=0.85*_med: continue               # normale Größe
            gb=r["y"]+r["h"]; gt=r["y"]
            if gb<=_base-0.28*_mh:   r["kpi"]["hoch"]="sup"                     # Unterkante klar über Grundlinie
            elif gt>=_base-0.62*_mh: r["kpi"]["hoch"]="sub"                     # Oberkante tief → Tiefstellung
    return g,recs

def detect_table(glyphs, W, H):
    """Tabellen-Rekonstruktion AUS DEN GLYPH-BOXEN (souverän, kein zeile-Clustering):
    Zeilen per horizontaler y-Projektion, globale Spalten per x-Gap-Cluster, Zelltext nach
    (y-Subzeile, x) geordnet (umbrochene Labels verschränken nicht). Auf Text-Layer ~Identität,
    auf Foto/Scan korrigierend. -> {'ncols':K,'rows':[{'y':int,'full':bool,'cells':[str,...]}]}"""
    gs=[g for g in glyphs if g.get("label")]
    if len(gs)<4: return {"ncols":1,"rows":[]}
    Hh=int(H)
    mh=float(np.median([g["h"] for g in gs])) or 12.0
    mw=float(np.median([g["w"] for g in gs])) or 8.0
    occ=np.zeros(Hh+2,np.int32)                                  # horizontale Projektion -> belegte y-Bänder
    for g in gs:
        a=max(0,int(g["y"])); b=min(Hh,int(g["y"]+g["h"])); occ[a:b+1]+=1
    rows=[]; ys=None; ye=0
    for y in range(Hh+1):
        if occ[y]>0:
            if ys is None: ys=y
            ye=y
        elif ys is not None: rows.append((ys,ye)); ys=None
    if ys is not None: rows.append((ys,ye))
    if not rows: return {"ncols":1,"rows":[]}
    def rowof(g):
        yc=g["y"]+g["h"]/2.0
        for i,(a,b) in enumerate(rows):
            if a-2<=yc<=b+2: return i
        return -1
    R={}
    for g in gs: R.setdefault(rowof(g),[]).append(g)
    rowsegs={}; starts=[]                                        # pro Zeile per x-Gap segmentieren
    for i in sorted(k for k in R if k>=0):
        row=sorted(R[i],key=lambda g:g["x"]); cells=[[row[0]]]
        for k in range(1,len(row)):
            if row[k]["x"]-(row[k-1]["x"]+row[k-1]["w"])>2.2*mw: cells.append([row[k]])
            else: cells[-1].append(row[k])
        rowsegs[i]=cells
        for c in cells: starts.append(c[0]["x"])
    if not starts: return {"ncols":1,"rows":[]}
    starts.sort(); anchors=[[starts[0]]]                         # Segment-Starts global clustern -> Spalten
    for x in starts[1:]:
        if x-anchors[-1][-1]<=3*mw: anchors[-1].append(x)
        else: anchors.append([x])
    anchors=[a for a in anchors if len(a)>=2] or anchors         # einmalige Starts (z.B. Notiz-Zeile) sind keine Spalte
    centers=[float(np.mean(a)) for a in anchors]
    if len(centers)>1:                                          # zu nahe Anker verschmelzen — echte Spalten-Gutter sind groß (>=5*mw)
        mc=[centers[0]]
        for c in centers[1:]:
            if c-mc[-1]<5.0*mw: mc[-1]=(mc[-1]+c)/2.0
            else: mc.append(c)
        centers=mc
    K=len(centers)
    bnds=[(centers[i]+centers[i+1])/2.0 for i in range(K-1)]
    def colof(x):
        for i,b in enumerate(bnds):
            if x<b: return i
        return K-1
    def cell_text(seg):
        sub=sorted(seg,key=lambda g:(round((g["y"]+g["h"]/2.0)/(0.7*mh)),g["x"])); t=""
        for k,g in enumerate(sub):
            # Leerzeichen an Glyphs eigenen Wortgrenzen (wpos==0) ODER an sichtbarer x-Lücke
            if k>0 and (g.get("wpos")==0 or (g["x"]-(sub[k-1]["x"]+sub[k-1]["w"]))>0.55*mw): t+=" "
            t+=g["label"]
        return t.strip()
    out_rows=[]
    for i in sorted(k for k in R if k>=0):
        cells=[""]*K
        for seg in rowsegs[i]:
            x0=min(g["x"] for g in seg); ci=colof(x0)            # Zuweisung per Start-x (wie die Anker), nicht Mittelpunkt
            cells[ci]=(cells[ci]+" "+cell_text(seg)).strip() if cells[ci] else cell_text(seg)
        nz=[j for j,c in enumerate(cells) if c]
        if not nz: continue
        out_rows.append({"y":int(rows[i][0]),"full":bool(nz==[0] and len(cells[0])>40),"cells":cells})
    return {"ncols":int(K),"rows":out_rows}

def read_pdf_chunks(path, snr=None, prefix=""):
    """Ein PDF komplett über GLYPHs Reader -> (snr, scan, chunks). Text-Layer = from_pdf+detect_table.
    Kein Text-Layer (Scan) -> scan=True (Glyph-MLP-Route). Optionaler Produkt-`prefix` (Name·Marke) macht
    jede Zeile produktbewusst; `snr` überschreibt die Dateinummer (echte Produkt-SNR via doc->snr-Brücke)."""
    import os
    b=os.path.basename(path)
    if not snr: snr=b.split("_")[1].split("-")[0] if "_" in b else os.path.splitext(b)[0]
    doc=fitz.open(path)
    if not "".join(p.get_text() for p in doc).strip():
        return snr, True, []
    pre=(prefix+" — ") if prefix else ""
    chunks=["Datenblatt SNR %s%s"%(snr, (" · "+prefix) if prefix else "")]; seen=set()
    for page in doc:
        g,recs=from_pdf(page)
        H,Wd=g.shape
        for row in detect_table(recs,Wd,H).get("rows",[]):
            cells=[c for c in row.get("cells",[]) if c]
            if not cells: continue
            t=(" | ".join(cells) if len(cells)>1 else cells[0]).strip()
            if ((" | " in t) or len(t)>=25) and t not in seen: seen.add(t); chunks.append((pre+t)[:480])
    return snr, False, chunks
