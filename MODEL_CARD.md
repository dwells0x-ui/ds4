# Synopsis de la fiche modèle DeepSeek v4

Ce document extrait les informations les plus importantes de la fiche modèle officielle
DeepSeek-V4-Flash sur Hugging Face, en mettant l'accent sur les faits qui comptent
pour l'inférence locale, le développement de DS4 et l'interprétation des benchmarks.

Source : https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash

## Famille de modèles

DeepSeek-V4 est une famille de modèles en préversion comprenant deux modèles de langage
Mixture-of-Experts :

| Modèle | Paramètres totaux | Paramètres actifs | Longueur de contexte |
|---|---:|---:|---:|
| DeepSeek-V4-Flash | 284B | 13B | 1M tokens |
| DeepSeek-V4-Pro | 1.6T | 49B | 1M tokens |

Flash est le modèle le plus petit et le plus efficace. La fiche modèle indique que Flash-Max peut
approcher les performances de raisonnement de Pro lorsqu'on lui accorde un budget de réflexion plus important, tout en
restant en retrait de Pro sur les connaissances pures et les tâches agentiques les plus complexes.

## Architecture

DeepSeek-V4 utilise une attention compressée à long contexte. La fiche modèle nomme la
conception hybride Compressed Sparse Attention (CSA) plus Heavily Compressed
Attention (HCA). En termes DS4, chaque couche conserve un cache KV brut à fenêtre glissante
pour les 128 derniers tokens. C'est le contexte local à haute résolution.

Après cette fenêtre brute, le modèle utilise des lignes KV compressées dépendantes de la couche :

| Indices de couche (base 0) | Ratio DS4 | État supplémentaire | Signification |
|---|---:|---|---|
| 0, 1 | aucun | aucun | Fenêtre glissante brute de 128 tokens uniquement |
| couches paires à partir de 2 | 4 | KV compressé + KV de l'indexeur | Une ligne compressée pour 4 tokens, avec un indexeur sélectionnant les lignes compressées visibles |
| couches impaires à partir de 3 | 128 | KV compressé | Une ligne compressée pour 128 tokens |

Ainsi, après les deux premières couches, le modèle alterne entre une attention compressée de ratio 4 et de ratio 128.
Un token dans une couche compressée porte son attention à la fois sur la fenêtre brute
des 128 derniers tokens et sur l'historique compressé plus ancien. La compression ici
est une compression sur l'axe temporel : plusieurs positions de tokens sont regroupées en une seule ligne KV.
Les lignes d'attention utilisent toujours les dimensions d'attention/valeur du modèle, de sorte que les lignes brutes et
compressées peuvent être consommées par le même calcul d'attention mixte.

Les couches de ratio 4 sont les couches d'attention compressée sélective. Elles maintiennent un
second flux compressé pour l'indexeur, et lorsque l'historique compressé est
plus grand que le top-k configuré, DS4 note les lignes compressées et en sélectionne jusqu'à
512 pour l'attention. Les couches de ratio 128 constituent le chemin fortement compressé :
elles n'ont pas de flux d'indexeur et utilisent directement les lignes compressées de ratio 128 disponibles.

DS4 valide ces détails à partir des métadonnées GGUF. Les constantes d'implémentation
fixes pertinentes sont :

- Couches : 43
- Attention à fenêtre glissante brute : 128 tokens
- Têtes de l'indexeur : 64
- Dimension des têtes de l'indexeur : 128
- Top-k de l'indexeur : 512

C'est la raison pratique pour laquelle le modèle peut exposer un contexte de 1M tokens sans un
cache KV complet standard pour chaque token dans chaque couche. La fiche modèle rapporte
qu'à 1M tokens, DeepSeek-V4-Pro nécessite beaucoup moins de calcul d'inférence par token unique
et de cache KV que DeepSeek-V3.2.

La famille utilise également :

- Manifold-Constrained Hyper-Connections (mHC), destinées à améliorer la stabilité de la
  propagation du signal à travers les couches.
- L'optimiseur Muon, utilisé pour une convergence plus rapide et une stabilité d'entraînement accrue.
- Un pipeline de post-entraînement avec cultivation d'experts de domaine suivie d'une
  consolidation unifiée via distillation on-policy.

## Précision et poids

Les entrées de téléchargement officielles comprennent :

| Modèle | Précision |
|---|---|
| DeepSeek-V4-Flash-Base | FP8 Mixed |
| DeepSeek-V4-Flash | FP4 + FP8 Mixed |
| DeepSeek-V4-Pro-Base | FP8 Mixed |
| DeepSeek-V4-Pro | FP4 + FP8 Mixed |

Pour les modèles instruct, la fiche modèle décrit FP4 + FP8 Mixed comme utilisant FP4
pour les paramètres des experts MoE et FP8 pour la plupart des autres paramètres.

## Modes de raisonnement

Les modèles instruct prennent en charge trois modes d'effort de raisonnement :

| Mode | Comportement attendu | Forme de sortie |
|---|---|---|
| Non-think | Réponses rapides et intuitives | Résumé `</think>` |
| High | Raisonnement délibéré pour les tâches plus difficiles | Résumé `<think>... </think>` |
| Max | Budget de raisonnement le plus large | Prompt système spécial plus réflexion et résumé |

La fiche modèle recommande d'utiliser une fenêtre de contexte d'au moins 384K tokens pour Think
Max.

## Benchmarks Flash importants

### DeepSeek-V4-Flash selon les modes de raisonnement

| Benchmark | Non-Think | High | Max |
|---|---:|---:|---:|
| GPQA Diamond Pass@1 | 71.2 | 87.4 | 88.1 |
| MMLU-Pro EM | 83.0 | 86.4 | 86.2 |
| SimpleQA-Verified Pass@1 | 23.1 | 28.9 | 34.1 |
| Chinese-SimpleQA Pass@1 | 71.5 | 73.2 | 78.9 |
| HLE Pass@1 | 8.1 | 29.4 | 34.8 |
| LiveCodeBench Pass@1 | 55.2 | 88.4 | 91.6 |
| HMMT 2026 Feb Pass@1 | 40.8 | 91.9 | 94.8 |
| IMOAnswerBench Pass@1 | 41.9 | 85.1 | 88.4 |
| SWE Verified Resolved | 73.7 | 78.6 | 79.0 |
| Terminal Bench 2.0 Acc | 49.1 | 56.6 | 56.9 |
| MCPAtlas Pass@1 | 64.0 | 67.4 | 69.0 |
| Toolathlon Pass@1 | 40.7 | 43.5 | 47.8 |

### DeepSeek-V4-Flash-Base

Le tableau du modèle de base rapporte ces scores Flash-Base :

| Benchmark | Shots | Score |
|---|---:|---:|
| SuperGPQA EM | 5-shot | 46.5 |
| MMLU EM | 5-shot | 88.7 |
| MMLU-Pro EM | 5-shot | 68.3 |
| Simple-QA verified EM | 25-shot | 30.1 |
| HumanEval Pass@1 | 0-shot | 69.5 |
| GSM8K EM | 8-shot | 90.8 |
| LongBench-V2 EM | 1-shot | 44.7 |

La fiche modèle rapporte SuperGPQA dans le tableau du modèle de base, et non dans le tableau
de comparaison des modes de raisonnement des modèles instruct.

## Template de chat et encodage

La version ne s'appuie pas sur un template de chat Jinja comme source de vérité. Le
moteur de rendu de prompt officiel est le code Python dans
`encoding/encoding_dsv4.py`, avec des exemples et des tests dans le même répertoire
`encoding` :

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/encoding/encoding_dsv4.py
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/encoding/test_encoding_dsv4.py

Les tokens spéciaux importants sont :

| Objet | Token |
|---|---|
| Début de séquence | `<｜begin▁of▁sentence｜>` |
| Fin du tour de l'assistant | `<｜end▁of▁sentence｜>` |
| Préfixe du tour de l'utilisateur | `<｜User｜>` |
| Préfixe du tour de l'assistant | `<｜Assistant｜>` |
| Préfixe du dernier rappel | `<｜latest_reminder｜>` |
| Début de la réflexion | `<think>` |
| Fin de la réflexion / marqueur de non-réflexion | `</think>` |
| Marqueur de balisage d'outil DSML | `｜DSML｜` |

Le moteur de rendu accepte les rôles `system`, `user`, `assistant`, `tool`,
`latest_reminder` et `developer`. Le rôle `developer` est décrit dans
les commentaires Python comme un rôle interne d'agent de recherche, et non comme un rôle de chat public normal.

Le mode de chat normal commence par le token BOS, puis le texte système s'il est présent, puis
l'alternance des marqueurs utilisateur et assistant. En mode de chat sans réflexion, une nouvelle
génération de l'assistant est ouverte avec :

```text
<｜Assistant｜></think>
```

Ce `</think>` immédiat indique au modèle de sauter le raisonnement caché et de produire
la réponse visible. En mode réflexion, une nouvelle génération de l'assistant est ouverte avec :

```text
<｜Assistant｜><think>
```

Les tours de réflexion terminés de l'assistant sont rendus sous forme de contenu de raisonnement à l'intérieur de
`<think>...</think>`, suivi de la réponse visible et du token EOS.

Par défaut, le moteur de rendu Python supprime le contenu de raisonnement antérieur de l'assistant avant
le dernier message de l'utilisateur. Si des outils sont présents sur un quelconque message, il désactive cette
suppression du raisonnement et conserve l'intégralité du contexte de raisonnement/d'outils. `reasoning_effort=max`
ajoute également un préfixe d'instruction spécial de haut effort avant le premier message rendu
en mode réflexion.

Les définitions d'outils sont transmises sous forme de schéma de fonction compatible OpenAI, mais
le modèle est instruit d'émettre du DSML. Un appel d'outil est rendu sous forme de bloc DSML
`tool_calls` contenant une ou plusieurs entrées `invoke`, chacune avec des paramètres
nommés. Les paramètres portent un indicateur `string="true"` pour les chaînes brutes et
`string="false"` pour les valeurs JSON telles que les nombres, booléens, tableaux ou objets.

DeepSeek-V4 ne rend pas de messages autonomes de rôle `tool`. Le préprocesseur Python
convertit les résultats d'outils en blocs de contenu utilisateur et rend chaque
résultat sous la forme :

```text
<tool_result>...</tool_result>
```

Les corps des résultats d'outils sont rendus sous forme de texte brut. Les caractères littéraux `<`, `>` et `&` provenant du
contenu de fichiers ou de la sortie du shell sont préservés ; seule la sentinelle de fermeture exacte
`</tool_result>` est échappée afin que le wrapper ne puisse pas être terminé par des données.

Lorsqu'il y a plusieurs résultats d'outils, le moteur de rendu les trie pour correspondre à
l'ordre des appels d'outils de l'assistant qui précèdent.

Le même script définit également des tokens de tâche spéciaux pour des tâches internes rapides telles que
la génération de titres, la génération de requêtes de recherche, la sélection d'actions, la classification d'autorité,
la classification de domaine et les décisions de lecture d'URL. Celles-ci sont
distinctes du rendu normal du chat/des outils.

## Notes d'exécution locale

La fiche modèle liste des exemples vLLM et SGLang pour un service compatible OpenAI.
Pour un déploiement local, elle recommande :

- `temperature = 1.0`
- `top_p = 1.0`
- Au moins 384K de contexte pour Think Max

Ce sont des recommandations de déploiement issues de la fiche modèle, pas nécessairement les
mêmes réglages utilisés pour un benchmarking déterministe. DS4 conserve `top_p=1.0` mais
ajoute une valeur par défaut locale `min_p=0.05` afin d'éviter d'échantillonner des tokens dont la probabilité est
bien inférieure à celle du meilleur token.

## Licence

Le dépôt et les poids du modèle sont sous licence MIT License.

## Citation

La fiche modèle cite :

```bibtex
@misc{deepseekai2026deepseekv4,
      title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
      author={DeepSeek-AI},
      year={2026},
}
```
