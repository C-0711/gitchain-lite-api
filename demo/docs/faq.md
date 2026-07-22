# FAQ (Demo-Wissen)

Frage: Braucht gitchain-lite eine Datenbank? Antwort: Nein. Das Dateisystem mit bare Git-Repositories ist die Wahrheit; hierarchy.json ist nur ein optionales Metadaten-Overlay.

Frage: Wie sichere ich Provenienz? Antwort: Jedes Atom ist nach dem Hash seines Inhalts benannt (content-addressed), jede Änderung ist ein Commit mit Autor und Zeitpunkt — signierte Commits machen die Herkunft kryptografisch beweisbar.

Frage: Welche Voraussetzungen hat der reine Server? Antwort: Nur git und Node ab Version 18. Python mit numpy und pymupdf wird ausschließlich für den PDF-Ingest gebraucht.
