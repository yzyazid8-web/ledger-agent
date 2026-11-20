// Infrastructure Azure : un environnement Container Apps et deux applications.
// L'interface est publique ; l'API n'est joignable que depuis l'interface (entrée interne).
// Déploiement : voir docs/deploiement.md

@description('Préfixe des noms de ressources')
param name string = 'ledger-agent'

@description('Région Azure')
param location string = resourceGroup().location

@description('Image de l\'API, ex. ghcr.io/<compte>/ledger-agent-api:latest')
param apiImage string

@description('Image de l\'interface, ex. ghcr.io/<compte>/ledger-agent-web:latest')
param webImage string

@description('Clé d\'API Groq, stockée comme secret de l\'application')
@secure()
param groqApiKey string

@description('Configuration d\'agent servie (nom d\'un fichier de evaluation/configs)')
param apiConfig string = 'agent'

@description('Identifiant d\'un environnement Container Apps existant à réutiliser ; vide pour en créer un')
param existingEnvironmentId string = ''

// Créé seulement si aucun environnement existant n'est fourni (certains abonnements n'en
// autorisent qu'un par région). Sans Log Analytics, facturé à part
resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = if (empty(existingEnvironmentId)) {
  name: '${name}-env'
  location: location
  properties: {}
}

var environmentId = empty(existingEnvironmentId) ? environment.id : existingEnvironmentId

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${name}-api'
  location: location
  properties: {
    managedEnvironmentId: environmentId
    configuration: {
      // Entrée interne : l'API n'est pas exposée sur Internet, ce qui protège le quota Groq
      ingress: {
        external: false
        targetPort: 8000
        transport: 'http'
      }
      secrets: [
        {
          name: 'groq-api-key'
          value: groqApiKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            {
              name: 'GROQ_API_KEY'
              secretRef: 'groq-api-key'
            }
            {
              name: 'API_CONFIG'
              value: apiConfig
            }
          ]
          probes: [
            {
              // Chargement du modèle et des index : jusqu'à ~2 minutes tolérées au démarrage
              type: 'Startup'
              httpGet: {
                path: '/health'
                port: 8000
              }
              initialDelaySeconds: 10
              periodSeconds: 10
              failureThreshold: 10
            }
          ]
        }
      ]
      // Une seule instance : Qdrant local et DuckDB ne se partagent pas entre processus.
      // Zéro instance au repos : aucun coût quand personne n'utilise l'application
      scale: {
        minReplicas: 0
        maxReplicas: 1
      }
    }
  }
}

resource web 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${name}-web'
  location: location
  properties: {
    managedEnvironmentId: environmentId
    configuration: {
      ingress: {
        external: true
        targetPort: 3000
        transport: 'http'
      }
    }
    template: {
      containers: [
        {
          name: 'web'
          image: webImage
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              // Nom de l'application API : résolu sur le réseau interne de l'environnement
              name: 'API_URL'
              value: 'http://${api.name}'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 2
      }
    }
  }
}

output webUrl string = 'https://${web.properties.configuration.ingress.fqdn}'
