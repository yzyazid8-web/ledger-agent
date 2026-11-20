# Interface de ledger-agent

Interface Next.js qui affiche en direct le raisonnement de l'agent, ses sources et l'origine de
chaque chiffre de la réponse.

## Lancer en local

L'API doit tourner (depuis `backend/` : `uv run poe api`).

```bash
npm install
npm run dev        # http://localhost:3000
```

Le navigateur passe par le relais `/api/*` de l'interface, qui joint l'API à l'adresse donnée par
`API_URL` (par défaut `http://localhost:8000`). Cette adresse est lue à l'exécution : la même image
Docker fonctionne en local et sur Azure.

## Pages

- `/` : poser une question, suivre les étapes de l'agent, consulter les sources citées.
- `/benchmark` : le tableau d'ablation et les comparaisons statistiques.
