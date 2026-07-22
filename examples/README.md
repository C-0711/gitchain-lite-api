# Beispiel-Goldsets
Format (JSONL, eine Frage je Zeile): `{"q": Frage, "doc": erwartete Quelldatei, "value": erwarteter Wert}` — Treffer per Ziffernvergleich (Atom-source enthaelt doc, Atom-Text enthaelt den Wert ziffernweise).
Auswertung: `toolkit/gitchain-eval.py <containerDir> examples/goldset-lstb.jsonl [--embed-url URL]` — recall@1/@3 fuer BM25 (plus dense/RRF mit Embeddings-Endpoint).
