# ledger-agent

Assistant d'analyse financière sur les rapports annuels américains (10-K, 10-Q), conçu pour que
**chaque chiffre d'une réponse soit traçable** : il provient d'un fait XBRL officiel, d'un passage
cité ou d'un calcul journalisé. Sinon, le système s'abstient.

Le projet est évalué sur [FinanceBench](https://github.com/patronus-ai/financebench) avec une
méthodologie stricte : jeux dev/test séparés par document, intervalles de confiance, tests
statistiques appariés et juge LLM d'une autre famille que le générateur.

> **État : en construction.** Ce README décrit uniquement ce qui existe aujourd'hui.

## Avancement

| Jalon | Contenu | État |
|---|---|---|
| 0 | Socle : monorepo, outillage, CI | ✅ |
| 1 | Ingestion : faits XBRL (SEC EDGAR) dans DuckDB, PDF FinanceBench indexés (Qdrant + BM25) | ✅ |
| 2 | RAG naïf de référence et harnais d'évaluation | ✅ |
| 3 | Outils : recherche hybride filtrée, faits XBRL typés, calculatrice sûre | ✅ |
| 3 bis | Rigueur de l'évaluation, suite à un audit externe | ✅ |
| 4 | Agent ReAct LangGraph : le modèle choisit ses outils, lit leurs résultats, est vérifié | ✅ |
| 5 | Étude d'ablation (dev en cours, puis passage unique sur le jeu test) | 🔄 en cours |
| 6 | API FastAPI : étapes de l'agent diffusées en direct (SSE) | ✅ |
| 7 | Interface Next.js : raisonnement en direct, sources, origine de chaque chiffre | ✅ |
| 8 | Images Docker construites, testées et publiées par la CI (GHCR) | ✅ |
| 9 | Déploiement continu sur Azure Container Apps ([guide](docs/deploiement.md)) | ✅ |

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
- `search_filings` : recherche hybride dans les rapports, filtrée par entreprise et exercice.
- `find_concepts` / `get_fact` : accès typé à DuckDB. L'agent n'écrit jamais de SQL :
  les requêtes sont paramétrées et la connexion est en lecture seule. Ces outils signalent aussi
  les chiffres retraités après publication.
- `calculate` : calculatrice qui analyse l'expression en arbre syntaxique, sans jamais l'exécuter.
  Chaque calcul reçoit un identifiant de source.

**Agent** (`agent/`) : un agent **ReAct** construit avec LangGraph. À chaque tour, le modèle
(`gpt-oss-120b`) **choisit lui-même** l'outil à appeler et ses arguments, **lit le résultat**, puis
décide de la suite : chercher autrement, changer d'exercice, calculer, répondre ou s'abstenir.

```
question ─► agent ──appels d'outils──► outils ──observations──┐
              ▲  │                                            │
              │  └──────────────────◄─────────────────────────┘
              │  submit_answer                    abstain
              │     ▼                                ▼
              └─ vérifier (règles) ─► réponse    abstention
               chiffre refusé
```

- **Outils exposés au modèle** : `search_filings`, `find_concepts`, `get_fact`, `calculate`,
  `submit_answer` et `abstain`. Une erreur d'outil (concept inconnu, exercice absent) est
  renvoyée comme observation : le modèle peut se corriger au tour suivant.
- Le modèle **ne fait aucune arithmétique** : chaque entrée d'un calcul cite sa source (une preuve
  `E3` ou un calcul précédent `C1`) et sa valeur est contrôlée avant exécution.
- Le **vérificateur**, par règles, exige que chaque nombre de la réponse vienne d'une preuve ou
  d'un calcul, **avec le bon signe**. Sinon, la réponse est renvoyée à l'agent, qui peut chercher
  ce qui manque ; au second refus, il s'abstient.
- **Traçable ne veut pas dire exact** : le vérificateur garantit que chaque nombre a une source,
  pas que le raisonnement est juste. Une formule mal choisie (par exemple une moyenne non
  calculée) produit un chiffre traçable mais faux ; c'est ce que mesure l'évaluation.
- **Budget d'étapes** (12 tours) : au dernier tour, seuls `submit_answer` et `abstain` sont
  proposés, ce qui garantit l'arrêt. Les sorties mal formées du fournisseur (appel d'outil
  invalide) ne consomment pas ce budget, dans une limite à part.
- **Point de comparaison** : `workflow.toml` garde un graphe à étapes fixes (planifier, collecter,
  juger la suffisance, synthétiser, vérifier). L'étude d'ablation mesure ce qu'apporte le fait de
  laisser le modèle piloter sa recherche.

**API** (`api/`) : `POST /chat` diffuse chaque étape de l'agent en direct (Server-Sent Events),
puis la réponse avec ses preuves et ses calculs. `POST /ask` renvoie le tout en une seule réponse
JSON, et `GET /companies` liste les entreprises couvertes.

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
- **Sous-ensemble rapide** : `dev_rapide`, 20 questions de dev tirées une fois pour toutes et
  équilibrées entre les types de questions, sert à itérer malgré les quotas gratuits. Avec si peu
  de questions, seuls de grands écarts sont significatifs.
- **Réponses chiffrées** : notées par règles, sur la seule réponse finale, avec le signe, l'échelle
  (million, milliard) et une tolérance d'arrondi.
- **Réponses rédigées** : notées par un juge LLM (Gemini) d'une **autre famille** que le
  générateur (gpt-oss). Le juge note aussi les réponses chiffrées, ce qui permet de **mesurer son
  accord avec les règles** (kappa de Cohen). Cette mesure ne porte que sur les réponses chiffrées :
  l'accord du juge sur les réponses rédigées n'est pas validé.
- **Page retrouvée** : part des questions où une page citée par FinanceBench a été récupérée.
  Elle n'a de sens que pour les réponses tirées des PDF : une réponse issue d'un fait XBRL n'en
  récupère aucune, ce qui fait baisser cet indicateur pour l'agent sans erreur de sa part.
- **Incertitude** : chaque exactitude est publiée avec son intervalle de confiance à 95 %
  (bootstrap). Deux configurations sont comparées sur les mêmes questions avec un **test de
  McNemar exact**.
- **Prudence** : une question en erreur ou non jugée compte comme non réussie.
- **Reproductibilité** : chaque exécution enregistre le commit, l'empreinte de la configuration,
  des prompts et du code de l'agent, du corpus et du découpage. Une reprise avec une
  configuration modifiée est refusée.

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
uv run poe eval workflow --split dev
uv run poe eval agent --split dev
uv run poe eval-report --split dev             # tableau d'ablation + tests de McNemar

uv run poe api                                 # API sur http://localhost:8000/docs
```

Les offres gratuites limitent le débit : Groq accepte 200 000 tokens par jour et par modèle.
L'évaluation s'arrête proprement quand le quota est atteint, et reprend là où elle s'est arrêtée
en relançant la même commande. Le cache évite de repayer les appels déjà faits.

### Avec Docker

Les images de l'API et de l'interface sont construites et testées par GitHub Actions à chaque PR,
puis publiées sur GHCR. L'image de l'API embarque un **snapshot versionné** des données
(index et base de faits, ≈ 105 Mo), publié comme Release GitHub et vérifié par son empreinte
(`data-snapshot.json`).

```bash
# Avec Docker installé : extraire le snapshot, puis tout lancer
cd backend && uv run python -m ledger_agent.snapshot extract <archive> --dest ../data-snapshot --sha256 <empreinte>
cd .. && docker compose up --build      # interface sur http://localhost:3000
```

L'API sert la configuration `agent` par défaut. Quand le quota du gros modèle est épuisé,
`API_CONFIG=agent_leger` fait piloter l'agent par `gpt-oss-20b`, qui a son propre quota.
Si les appels à Groq échouent avec `ConnectError`, `FORCE_IPV4=true` contourne une route IPv6
défaillante.

## Structure du dépôt

| Dossier | Contenu |
|---|---|
| `backend/src/ledger_agent/` | Paquet Python : ingestion, recherche, outils, LLM, évaluation |
| `backend/tests/` | Tests unitaires et tests de bout en bout (sans réseau) |
| `evaluation/configs/` | Configurations évaluées (`naive`, `hybrid`, `hybrid_rerank`, `workflow`, `agent` et ses ablations) |
| `evaluation/runs/` | Prédictions, scores et résumés de chaque exécution |
| `evaluation/reports/` | Comparaisons statistiques entre configurations |
