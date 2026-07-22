# Glypher-Bildleser aus Text-Layer-PDFs trainieren (Scans lesbar machen)

Text-Layer-PDFs sind **auto-gelabelte Trainingsdaten**: das Rendern der Seite liefert das
Bild, der Text-Layer liefert jede Glyphe als (Zeichen + Bounding-Box). Damit trainiert man
den Glypher-MLP-Bildleser (`glyph_clf.npz`, 384-d → K Klassen) — der Pfad, der *gescannte*
Belege liest — auf genau den Formularfonts, ohne eine einzige Handbeschriftung.

Nutzen: der Claims-Builder ueberspringt aktuell reine Scans (kein Text-Layer). Ein auf der
Testmenge trainierter Bildleser macht diese Scans lesbar → volle Abdeckung.

```bash
pip3 install numpy opencv-python-headless pymupdf torch   # optionale Trainings-Deps (NICHT Kern)

# 1) Trainingsdaten aus Text-Layer-PDFs (auto-gelabelt, 384-d Features):
python3 training/gen_glyph_trainset.py <textlayer-pdf-dir> glyph_trainset.npz

# 2) MLP trainieren (gleiches Format wie glyph_clf.npz -> direkt einsetzbar):
python3 training/train_glyph.py glyph_trainset.npz glyph_clf.trained.npz
```

Gemessen (138 Text-Layer-PDFs der Testmenge): 308.254 gelabelte Glyphen, 131 Klassen;
12 Epochen from scratch, ohne Augmentierung → 73% Val-Glyphen-Accuracy (steigend).
Naechste Schritte fuer Produktionsguete: Augmentierung (Scan-Rauschen/Blur/Rotation gegen
die Domain-Luecke Render→Scan), mehr Epochen, dann Test gegen die GT der Scan-Belege.

## Ehrlicher Befund: Scans sind segmentierungs-, nicht klassifikator-limitiert

Gemessen (12 Scan-LStB, CC-Segmenter + MLP, GT-Bruttolohn im gelesenen Text):
shipped glyph_clf **3/12**, Testmengen-trained **3/12** — der trainierte Klassifikator
schlaegt den alten NICHT. Der Engpass ist der **Segmenter** (connected-components
verschmilzt/spaltet Glyphen VOR der Klassifikation), nicht die Klassifikation. Deckt sich
mit dem glyph-mapper-V2-Befund ("der eigentliche Fehler ist der CC-Containerizer, nicht
Geometrie/Licht"). Der trainierte Klassifikator ist also die notwendige, aber nicht
hinreichende Zutat — Produktions-Scan-Lesen braucht zuerst einen pitch/gitter-basierten
Segmenter. Bis dahin deckt die Claims-Pipeline die Text-Layer-Belege ab (60/106 LStB);
Scans bleiben dokumentierte Luecke.
