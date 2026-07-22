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
