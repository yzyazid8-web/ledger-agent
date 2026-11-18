# ledger-agent

Assistant d'analyse financière sur les rapports annuels américains (10-K, 10-Q), conçu pour que
**chaque chiffre d'une réponse soit traçable** : il provient d'un fait XBRL officiel, d'un passage
cité ou d'un calcul journalisé. Sinon, le système s'abstient.

Le projet est évalué sur [FinanceBench](https://github.com/patronus-ai/financebench) avec une
méthodologie stricte : jeux dev/test séparés par document, intervalles de confiance, tests
statistiques appariés et juge LLM validé.

> **État : en construction.** Ce README décrit uniquement ce qui existe aujourd'hui.

## Avancement

| Jalon | Contenu | État |
|---|---|---|
| 0 | Socle : monorepo, outillage, CI | ✅ |
| 1 | Ingestion : faits XBRL (SEC EDGAR) dans DuckDB, PDF FinanceBench indexés (Qdrant + BM25) | ✅ |
| 2 | RAG naïf de référence et harnais d'évaluation | ✅ |
| 3 | Outils : recherche hybride filtrée, faits XBRL typés, calculatrice sûre | ✅ |
| 3 bis | Rigueur de l'évaluation, suite à un audit externe | ✅ |
| 4 | Agent LangGraph (routage, planification, vérification des chiffres) | ⏳ à venir |
| 5 | Étude d'ablation complète sur le jeu test | ⏳ à venir |
| 6–7 | API FastAPI et interface Next.js | ⏳ à venir |
| 8–9 | Docker, CI/CD et déploiement sur Azure | ⏳ à venir |

## Ce qui fonctionne aujourd'hui

**Ingestion** (`backend/src/ledger_agent/ingestion/`)
- **Chiffres XBRL** de 32 entreprises, issus de l'API SEC EDGAR : environ 896 000 faits.
  L'exercice de chaque chiffre est recalculé à partir de sa date de fin de période, car le champ
  `fy` de la SEC décrit le dépôt et non le chiffre.
- **84 PDF FinanceBench**, soit 12 013 pages et 30 340 chunks. Chaque chunk reste dans une seule
  page et porte un en-tête de contexte (entreprise, document, exercice, page).
- **Index vectoriel** Qdrant (`bge-small-en-v1.5`, sur CPU) synchronisé de façon incrémentale,
  et **index lexical** BM25.

**Recherche** (`retrieval/`) : fusion vecteurs + BM25 (RRF), reranker optionnel, et filtres
entreprise/exercice repérés dans la question. Les filtres sont élargis si les résultats manquent,
et **cet élargissement est signalé**, jamais silencieux.

**Outils de l'agent** (`tools/`)
- `search_concepts` / `get_financial_fact` : accès typé à DuckDB. L'agent n'écrit jamais de SQL :
  les requêtes sont paramétrées et la connexion est en lecture seule. Ces outils signalent aussi
  les chiffres retraités après publication.
- `calculate` : calculatrice qui analyse l'expression en arbre syntaxique, sans jamais l'exécuter.
  Chaque calcul reçoit un identifiant de source.

**Évaluation** (`evaluation/`) : voir la méthodologie ci-dessous.

## Méthodologie d'évaluation

- **Corpus** : les 84 documents cités par les 150 questions publiques de FinanceBench, dans un
  index unique partagé. C'est un réglage intermédiaire entre les deux de l'article FinanceBench
  (un index par document, ou un index commun aux 361 documents). La recherche y est **plus
  facile** que sur le corpus complet. L'écart sera mesuré plus tard.
- **Découpage dev/test groupé par document** : 50 questions dev (27 documents) et 100 questions
  test (57 autres documents), sans aucun document commun. Le découpage est figé dans
  `evaluation/splits.json`. Tous les réglages se font sur dev, et le jeu test n'est utilisé
  qu'une fois par configuration.
- **Réponses chiffrées** : notées par règles, sur la seule réponse finale, avec le signe, l'échelle
  (million, milliard) et une tolérance d'arrondi.
- **Réponses rédigées** : notées par un juge LLM (Gemini) d'une **autre famille** que le
  générateur (gpt-oss). Le juge note aussi les réponses chiffrées, ce qui permet de **mesurer son
  accord avec les règles** (kappa de Cohen).
- **Incertitude** : chaque exactitude est publiée avec son intervalle de confiance à 95 %
  (bootstrap). Deux configurations sont comparées sur les mêmes questions avec un **test de
  McNemar exact**.
- **Prudence** : une question en erreur ou non jugée compte comme non réussie.
- **Reproductibilité** : chaque exécution enregistre le commit, l'empreinte de la configuration,
  du prompt, du corpus et du découpage. Une reprise avec une configuration modifiée est refusée.

## Résultats

Les résultats sur le nouveau découpage sont en cours de production. Ils seront publiés ici avec
leurs intervalles de confiance, et le détail se trouve dans `evaluation/runs/` et
`evaluation/reports/`.

## Installation et reproduction

Prérequis : Python 3.12, [uv](https://docs.astral.sh/uv/), des clés gratuites
[Groq](https://console.groq.com) et [Gemini](https://aistudio.google.com).

```bash
cp .env.example .env            # renseigner SEC_USER_AGENT, GROQ_API_KEY, GEMINI_API_KEY
cd backend
uv sync
uv run poe check                # lint, formatage, typage, tests

uv run poe ingest-facts         # chiffres XBRL -> DuckDB
uv run poe ingest-docs          # PDF -> chunks -> Qdrant + BM25 (environ 2 h sur CPU)

uv run poe eval-retrieval hybrid --split dev   # recherche seule, gratuite
uv run poe eval naive --split dev              # génération + notation
uv run poe eval hybrid --split dev
uv run poe eval-compare naive hybrid --split dev
```

Les offres gratuites limitent le débit : Groq accepte 200 000 tokens par jour et par modèle.
L'évaluation s'arrête proprement quand le quota est atteint, et reprend là où elle s'est arrêtée
en relançant la même commande. Le cache évite de repayer les appels déjà faits.

## Structure du dépôt

| Dossier | Contenu |
|---|---|
| `backend/src/ledger_agent/` | Paquet Python : ingestion, recherche, outils, LLM, évaluation |
| `backend/tests/` | Tests unitaires et tests de bout en bout (sans réseau) |
| `evaluation/configs/` | Configurations évaluées (`naive`, `hybrid`, `hybrid_rerank`) |
| `evaluation/runs/` | Prédictions, scores et résumés de chaque exécution |
| `evaluation/reports/` | Comparaisons statistiques entre configurations |
