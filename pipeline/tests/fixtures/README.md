# Fixtures de test

Ce dossier reçoit les fichiers binaires utilisés par les tests d'intégration
qui demandent un VRM réel ou une vidéo réelle.

Les fichiers de ce dossier sont **gitignored** (sauf ce README et `.gitkeep`).

## Fichiers attendus (à créer manuellement)

- `sample_avatar.vrm` — un avatar VRM 1.0 minimaliste pour les tests
  d'inspection. Source recommandée : VRoid Studio (export VRM 1.0).
- `sample_avatar_vrm0.vrm` — un avatar VRM 0.x pour tester la rétrocompatibilité.

Aucun test du suite par défaut ne dépend de ces fichiers ; ils sont activés
quand présents par les tests marqués `@pytest.mark.integration`.
