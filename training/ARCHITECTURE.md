# Glypher-Readout: die vollstaendige Lese→Korrektur→Lernen-Pipeline

Ziel: Belege aller Typen (Lohnsteuerbescheinigung, KAP, Rechnung, Kassenbon, Ticket)
bis nahe 100% lesen — deterministisch wo moeglich, LLM nur super-gezielt auf Restfaelle.

## Die Schichten (jede zahlt nur, was die vorige nicht loesen konnte)

1. **CNN-Glyph-Leser** (`train_cnn.py`) — 2-Kanal (ink+grau) CNN auf 32×24-Glyphen,
   trainiert auf allen 5 Typen. Gemessen (doc-held-out): KAP/Rechnung ~100%, Ticket 97%,
   LStB 92%, Kassenbon 65%; gesamt 92.7%. Restfehler sind INTRINSISCH mehrdeutige Glyphen
   (i↔l, ,↔., 0↔8) — kein Klassifikator loest die isoliert.

2. **Multi-ASCII-MLP** (`train_final_mlp.py`) — der Drop-in fuer den Glypher-from_image-Pfad
   (`glyph_clf.npz`-Format, W1/b1/W2/b2/classes), auf ALLEN Glyphen gelernt ("learn all").

3. **Duden + Levenshtein** (`duden_correct.py`) — 356k deutsche Vollformen fuellen die
   Luecken. Ein gelesenes Wort, das nicht im Duden steht, wird durch den EINDEUTIG naechsten
   Duden-Kandidaten (Levenshtein ≤2, inkl. fehlender/zusaetzlicher Buchstaben) ersetzt.
   Gesetz (wie glypher `duden.py`): der Duden schlaegt vor, die Bank/Glyph-Evidenz beweist.

4. **300m-Lexikon-Container** (`build_lexicon.py`) — jedes Wort aller Belege + ELSTER-Katalog,
   mit embeddinggemma (300m) eingebettet, im gitchain-lite-Container-Format. Liefert die
   Kandidaten, die Schicht 5 als Kontext bekommt.

5. **gemma4-mm (31B), super-gezielt** — NUR auf Restfaelle (fehlende Werte, mehrdeutige/
   Fehler-Woerter). Prompt = Roh-Lesung + Lexikon-Kandidaten + Dokumentkontext; vLLM
   prefix-cached den Kontext. Validiert (liefert praezise):
     "Bruttoarbeitsiohn" → Bruttoarbeitslohn ·  "Soldaritatszuschlag" → Solidaritätszuschlag
     (fehlende Buchstaben + Umlaut) ·  "25.300 00" → 25.300,00 EUR
   TurboQuant ist hier der KV-Cache-Quantisierer (mehr Kontext pro GPU), NICHT eine Art,
   Embeddings in den KV-Cache zu laden (Embeddings ≠ KV-Tensoren).

6. **Final read + self-training** — die korrigierte Ausgabe ist hochsichere Wahrheit; sie
   erweitert (v.a. fuer Scans, die keinen Text-Layer haben) die Trainingsdaten → Schicht 1/2
   neu lernen → besserer Leser. Text-Layer-Belege sind bereits perfekt gelabelt.

## Ehrliche Grenzen
- Kassenbon (65%): 32 Belege decken den Retailer-Font-Raum nicht; braucht mehr Dokumente.
- Scans: Segmentierung (nicht Klassifikation) ist der Engpass; braucht pitch/gitter-Segmenter.
- Deps (torch/opencv/pymupdf) sind Trainings-only, NICHT im zero-dep-Kern.
