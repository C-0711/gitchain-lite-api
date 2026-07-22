# gitchain-lite Handbuch (Demo-Wissen)

gitchain-lite ist ein souveräner, git-nativer Container-Store. Der Standard-Port des Servers ist 7420 und lässt sich über die Umgebungsvariable PORT ändern.

Ein Container entsteht durch push-to-create: der erste git push auf einen neuen Pfad legt das bare Repository automatisch an — kein Admin-Schritt nötig.

Die Hierarchie ist der Pfad: Tenant, dann beliebig verschachtelte Projekte, dann der Container — genau wie group/subgroup/project.git bei GitLab.

TurboQuant komprimiert die Vektoren eines Containers etwa 23-fach bei ungefähr 0,998 Recall; beim ersten Query wird der Store einmalig dekodiert (decode-on-open).

Suche funktioniert auch komplett ohne Modelle: der BM25-Pfad beantwortet Anfragen lexikalisch. Ist EMBED_URL gesetzt, werden dichte und lexikalische Treffer per Reciprocal-Rank-Fusion kombiniert.
