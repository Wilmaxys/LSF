# Avatars VRM

Place ici les fichiers `.vrm` des avatars que tu veux pouvoir charger dans
le viewer.

Le viewer charge par défaut `avatars/default.vrm` ; pour utiliser un avatar
différent, passer son nom en query string : `?avatar=avatars/alicia.vrm`.

## Où trouver des VRM gratuits

- **VRoid Hub** — https://hub.vroid.com/ (bibliothèque officielle)
- **Booth** — https://booth.pm/ (store japonais, beaucoup d'avatars VRM)
- **VRoid Studio** — https://vroid.com/en/studio (créer son propre avatar)

## Contraintes

L'avatar doit être un VRM **0.x** ou **1.0** conforme. Voir
[docs/AVATARS.md](../../../docs/AVATARS.md) pour vérifier qu'un VRM est
compatible avant de l'utiliser dans le pipeline.

Les fichiers de ce dossier sont gitignored (sauf ce README).
