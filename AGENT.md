# Notes pour l'agent

`ds4.c` est un moteur d'inférence spécifique à DeepSeek V4 Flash. Ce n'est pas un exécuteur
GGUF générique. L'objectif est une base de code C petite, lisible et hautement performante, avec
de l'Objective-C uniquement là où Metal l'exige et des kernels Metal sous `metal/`.

## Objectifs

- Conserver comme chemin de production l'inférence par graphe Metal du modèle complet.
- Toujours s'assurer que le streaming SSD, CUDA, l'inférence distribuée et l'inférence Metal par défaut ne sont pas affectés par les corrections apportées à d'autres parties du code.
- Conserver un chargement du modèle adossé à mmap pour le cas Metal par défaut ; ne pas copier avidement l'intégralité du GGUF. Garder explicite le chargement du modèle pour le streaming SSD des experts routés : tampons alloués, lectures rapides depuis le disque, toujours essayer de masquer le chargement des experts routés manquants en les chargeant pendant l'inférence de l'expert partagé et des experts routés déjà présents en RAM. Toujours essayer de masquer le chargement des couches pour le prefill en mode streaming SSD en utilisant le temps d'inférence de la couche courante pendant que la suivante est chargée.
- Garder le backend CPU purement CPU et l'utiliser uniquement comme code de référence/débogage.
- Préserver l'exactitude avant la vitesse. Ne pas conserver un chemin plus rapide présentant une dérive inexpliquée de l'attention, du KV cache ou des logits.
- Rendre pratiques les longues sessions d'agent locales grâce à la réutilisation du KV en direct et aux points de contrôle KV sur disque.

## Règles de qualité

- Garder l'implémentation petite, nette et facile à comprendre. Essayer d'écrire un code élégant, dans un état de grâce. Ne pas se contenter de la première idée venue, chercher la conception la plus minimale et la mieux conçue. Ne pas introduire de bricolage : du code très fragile qui ne fait que rustiner des cas particuliers, du code mort, du code inutile et du code bien plus compliqué qu'il ne devrait l'être.
- Commenter le code d'inférence important là où les mécanismes du modèle, la durée de vie du cache, la politique mémoire ou l'orchestration de l'API ne sont pas évidents à la lecture du code local.
- Préférer les commentaires à côté de l'implémentation plutôt que des documents de conception séparés.
- Garder les commentaires instructifs et compacts : expliquer pourquoi une forme, un ordonnancement, une frontière de cache ou un choix mémoire existe.
- Garder les API publiques étroites. Le code CLI/serveur ne doit pas connaître les détails internes des tenseurs.
- Ne pas ajouter de variantes sémantiques permanentes derrière des flags. Les commutateurs de diagnostic sont acceptables lorsqu'ils valident l'unique chemin de release.
- Ne pas introduire de C++.

## Sécurité

- Éviter les grandes exécutions d'inférence CPU sur macOS ; le chemin CPU a déjà révélé des défaillances de la VM du noyau avec de très grands mappings.
- Ne pas exécuter simultanément plusieurs processus de modèle gigantesques. Le verrou d'instance est intentionnel.

## Organisation

- `ds4.c` : chargement du modèle, tokenizer, code de référence CPU, ordonnancement du graphe Metal,
  sessions, sérialisation de la charge utile du cache disque.
- `ds4_cli.c` : ligne de commande, REPL linenoise, gestion de la transcription interactive.
- `ds4_server.c` : API HTTP compatible OpenAI/Anthropic, file d'attente des workers, streaming,
  mappage des appels d'outils, politique de KV cache sur disque.
- `ds4_metal.m` : runtime Metal en Objective-C et wrappers de kernels.
- `metal/*.metal` : kernels de calcul.
- `tests/` : tests unitaires et tests d'intégration en direct.
- `misc/` : notes ignorées, expérimentations et anciens documents de planification.

Cette liste n'est pas exhaustive, consultez les fichiers pour plus d'informations.

## Tests

Utilisez `make` pour valider la build. Utilisez `make test` pour les tests unitaires/de régression lorsqu'un
modèle et Metal sont disponibles. N'utilisez les tests de serveur en direct que lorsque vous testez
intentionnellement la surface de l'API.

À chaque changement majeur susceptible d'affecter l'un des éléments suivants, veillez à :

1. Tester le chemin Metal normal et vérifier que la vitesse reste au niveau où elle était.
2. Tester le chemin de streaming SSD.
3. Tester l'inférence distribuée si elle peut être affectée, mais demander à l'utilisateur avant de le faire.
4. Vérifier si CUDA pourrait être cassé par le changement, et demander à l'utilisateur de vous donner accès à la machine CUDA pour tester réellement que tout fonctionne encore.
