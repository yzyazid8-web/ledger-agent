# Interface de ledger-agent

Interface Next.js qui affiche en direct le raisonnement de l'agent, ses sources et l'origine de
chaque chiffre de la réponse.

## Lancer en local

L'API doit tourner (depuis `backend/` : `uv run poe api`).

```bash
npm install
npm run dev        # http://localhost:3000
```

L'adresse de l'API se règle avec `NEXT_PUBLIC_API_URL` (par défaut `http://localhost:8000`).

## Pages

- `/` : poser une question, suivre les étapes de l'agent, consulter les sources citées.
- `/benchmark` : le tableau d'ablation et les comparaisons statistiques.
