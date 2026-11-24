# Déploiement sur Azure Container Apps

```
Internet ──► ledger-agent-web (entrée publique, HTTPS)
                    │  relais /api/*  →  http://ledger-agent-api  (réseau interne)
                    ▼
             ledger-agent-api (entrée interne, invisible depuis Internet) ──► Groq
```

- Les images sont construites et publiées sur GHCR par le workflow `Images Docker`.
- L'infrastructure est décrite dans `infra/main.bicep`.
- Les deux applications se mettent en veille sans visiteur (`minReplicas = 0`) : on reste dans la
  part gratuite de Container Apps, au prix d'un premier chargement de 30 à 60 s.

## 1. Prérequis

1. Installer l'Azure CLI, puis ouvrir un **nouveau** terminal :
   ```powershell
   winget install -e --id Microsoft.AzureCLI
   ```
2. Rendre publiques les deux images sur GHCR (elles ne contiennent aucun secret : seulement le
   code et des données publiques) : profil GitHub → **Packages** → `ledger-agent-api` →
   **Package settings** → **Change visibility** → **Public**. Même chose pour `ledger-agent-web`.

## 2. Premier déploiement

```powershell
az login
az provider register --namespace Microsoft.App --wait
az group create --name rg-ledger-agent --location francecentral

# La clé Groq est lue dans .env, sans apparaître à l'écran ni dans l'historique
$groq = ((Get-Content .env | Select-String '^GROQ_API_KEY=').Line -split '=', 2)[1].Trim('"')

az deployment group create `
  --resource-group rg-ledger-agent `
  --template-file infra/main.bicep `
  --parameters apiImage=ghcr.io/yzyazid8-web/ledger-agent-api:latest `
               webImage=ghcr.io/yzyazid8-web/ledger-agent-web:latest `
               groqApiKey=$groq `
  --query properties.outputs.webUrl.value -o tsv
```

La dernière commande affiche l'adresse publique du site.

**Si l'abonnement a déjà un environnement Container Apps dans la région** (erreur
`MaxNumberOfRegionalEnvironmentsInSubExceeded`, certains abonnements n'en autorisant qu'un par
région), on le réutilise en ajoutant son identifiant :

```powershell
$envId = az resource list --resource-type Microsoft.App/managedEnvironments `
  --query "[?location=='francecentral'].id | [0]" -o tsv

az deployment group create `
  --resource-group rg-ledger-agent `
  --template-file infra/main.bicep `
  --parameters existingEnvironmentId=$envId `
               apiImage=ghcr.io/yzyazid8-web/ledger-agent-api:latest `
               webImage=ghcr.io/yzyazid8-web/ledger-agent-web:latest `
               groqApiKey=$groq `
  --query properties.outputs.webUrl.value -o tsv
```

Les applications restent dans `rg-ledger-agent`, mais dépendent alors de cet environnement :
supprimer le groupe qui le contient arrêterait aussi ledger-agent.

## 3. Déploiement continu (après le premier déploiement)

Le workflow `Déploiement Azure` met à jour les applications après chaque publication d'images
sur `main`. Il se connecte à Azure sans mot de passe stocké (OIDC).

```powershell
$sub = az account show --query id -o tsv
$tenant = az account show --query tenantId -o tsv
$app = az ad app create --display-name ledger-agent-github --query appId -o tsv
az ad sp create --id $app
az role assignment create --assignee $app --role Contributor `
  --scope "/subscriptions/$sub/resourceGroups/rg-ledger-agent"
az ad app federated-credential create --id $app --parameters `
  '{\"name\":\"github-production\",\"issuer\":\"https://token.actions.githubusercontent.com\",\"subject\":\"repo:yzyazid8-web/ledger-agent:environment:production\",\"audiences\":[\"api://AzureADTokenExchange\"]}'
"AZURE_CLIENT_ID=$app"; "AZURE_TENANT_ID=$tenant"; "AZURE_SUBSCRIPTION_ID=$sub"
```

**Attention au « subject »** : GitHub peut présenter un sujet qui contient les identifiants
numériques du compte et du dépôt, par exemple
`repo:<compte>@<id>/ledger-agent@<id>:environment:production`. Le sujet déclaré dans Azure doit
être **exactement** celui affiché à la ligne `subject claim` du journal de l'étape `azure/login`.
En cas d'erreur `AADSTS70021` (aucun sujet ne correspond), on remplace la confiance avec ce sujet.
En cas d'erreur `AADSTS700016`, c'est la valeur du secret `AZURE_CLIENT_ID` qui est fausse : elle
doit valoir l'`appId` donné par `az ad app list --display-name ledger-agent-github`.

Puis, dans GitHub → **Settings** du dépôt :
1. **Environments** → **New environment** → `production`.
2. **Secrets and variables** → **Actions** → onglet **Secrets** : ajouter `AZURE_CLIENT_ID`,
   `AZURE_TENANT_ID` et `AZURE_SUBSCRIPTION_ID` avec les valeurs affichées.
3. Onglet **Variables** : ajouter `AZURE_RESOURCE_GROUP` = `rg-ledger-agent`.

## 4. Suivi en production avec Langfuse (facultatif)

1. Sur Langfuse Cloud : créer un projet, puis une paire de clés d'API (clé publique et clé
   secrète). Noter l'adresse du serveur affichée (`https://cloud.langfuse.com` pour l'Europe).
2. Ajouter la clé secrète comme secret de l'API, puis les variables d'environnement :

   ```powershell
   az containerapp secret set -n ledger-agent-api -g rg-ledger-agent `
     --secrets langfuse-secret-key=<clé secrète>
   az containerapp update -n ledger-agent-api -g rg-ledger-agent `
     --set-env-vars LANGFUSE_PUBLIC_KEY=<clé publique> LANGFUSE_SECRET_KEY=secretref:langfuse-secret-key LANGFUSE_HOST=https://cloud.langfuse.com
   ```

3. Poser une question sur le site : elle apparaît dans Langfuse sous quelques secondes.

Pour un nouveau déploiement complet, les paramètres `langfusePublicKey`, `langfuseSecretKey` et
`langfuseHost` du fichier Bicep font la même chose. Sans clés, le suivi reste désactivé.

## Coûts

- Container Apps (plan Consommation) : une part gratuite mensuelle (180 000 vCPU-secondes,
  360 000 Gio-secondes, 2 millions de requêtes) couvre largement une démonstration avec mise en
  veille.
- Aucun Log Analytics ni registre Azure : les images restent sur GHCR, gratuit.
- Pour tout supprimer : `az group delete --name rg-ledger-agent`.
