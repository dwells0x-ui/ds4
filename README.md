<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

**DwarfStar** est un petit moteur d'inférence natif optimisé en priorité pour
**DeepSeek V4 Flash**. Il prend également en charge **GLM 5.2** et, sur les
machines dotées d'une très grande quantité de mémoire, **DeepSeek V4 PRO**. Il
est autonome et volontairement restreint : ce n'est pas un exécuteur GGUF
généraliste. Le chargement des modèles, le rendu des prompts, les appels
d'outils, l'état du cache KV, le serveur HTTP et l'agent de codage sont conçus
et testés ensemble.
Le dépôt inclut également des outils et des données pour GGUF, imatrix, la qualité et la vitesse.

Backends pris en charge :

* **Metal**, la cible principale, sur les Mac disposant de 96 Go ou plus. Les
  machines plus modestes peuvent utiliser le streaming SSD.
* **NVIDIA CUDA**, y compris les systèmes multi-GPU et le DGX Spark.
* **ROCm** sur les systèmes Strix Halo tels que le Framework Desktop.

Ce projet n'existerait pas sans **llama.cpp et GGML** ; assurez-vous de lire
la section des remerciements, un grand merci à Georgi Gerganov et à tous les
autres contributeurs.

La prise en charge des modèles est intentionnellement opportuniste. Le projet
suit les meilleurs poids ouverts pour les tailles de machines locales utiles,
en particulier les ordinateurs portables de 128 Go et les stations de travail de
512 Go. Un modèle peut être retiré lorsqu'un meilleur remplaçant arrive.

# Alors, que puis-je faire avec ce logiciel ?

* Vous pouvez exécuter des modèles très performants sur du matériel grand public, un MacBook, un DGX Spark ou un Strix Halo par exemple. Même si vous n'avez pas assez de RAM, grâce au streaming SSD, vous pouvez l'exécuter à une vitesse décente.
* En utilisant la prise en charge multi-GPU CUDA et le micro-batching du décodage et de la génération de ds4-server, vous pouvez transformer un serveur équipé de cartes CUDA un peu anciennes (architecture Ada Lovelace), qui ne sont plus prises en charge pour les nouveaux modèles par vLLM, en un serveur LLM multi-utilisateur pour votre entreprise. Nous avons testé cette configuration avec 8 cartes NVIDIA L40S et plusieurs sessions avec de très bons résultats. 120 t/s de génération agrégée, 2000 t/s en prefill.
* En utilisant deux MacBook M5 Max / M3 Ultra en RDMA, vous pouvez exécuter DeepSeek Flash 4 bits ou GLM 5.2 avec du parallélisme de tenseurs.
* Vous pouvez également utiliser le parallélisme de pipeline pour assembler plusieurs systèmes afin de cumuler leur RAM et d'exécuter des modèles plus grands.

## Motivations

* Des modèles à poids ouverts performants tiennent désormais sur des machines personnelles haut de gamme.
* DeepSeek V4 Flash et PRO, GLM 5.2 tolèrent une quantification agressive des experts routés.
* Les caches KV compressés et les SSD locaux rapides rendent les contextes longs praticables.
* L'idée d'un système d'inférence spécialisé pour quelques modèles.

# Divulgation complète concernant l'IA

* Ce logiciel est développé avec une **forte assistance de GPT 5.5, 5.6, Claude Fable** et avec des humains qui dirigent les idées, les tests et le débogage. Nous le disons ouvertement car cela a façonné la manière dont le projet a été construit. Si vous n'êtes pas à l'aise avec du code développé par IA, ce logiciel n'est pas pour vous. Le remerciement ci-dessous est tout aussi important : ceci n'existerait pas sans `llama.cpp` et GGML, largement écrits à la main.

## Remerciements à llama.cpp et GGML

`ds4.c` ne se lie pas à GGML, mais il **existe grâce à la voie ouverte par le
projet llama.cpp et aux kernels, formats de quantification, écosystème GGUF et
au savoir d'ingénierie durement acquis qui y ont été développés**.
Nous sommes reconnaissants et redevables envers [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
et ses contributeurs. Leur implémentation, leurs kernels, leurs tests et leurs choix de
conception ont été une référence essentielle lors de la construction de ce chemin d'inférence
spécifique à DeepSeek V4. Certaines parties au niveau du code source sont conservées ou
adaptées ici sous licence MIT : les dispositions et tables de quantification GGUF, la logique
CPU de quantification/produit scalaire, et certains kernels. Pour cette raison, et parce que
nous sommes sincèrement reconnaissants, nous conservons la notice de copyright des auteurs de
GGML dans notre fichier `LICENSE`.

## Statut

Le logiciel évolue actuellement très rapidement. Considérez-le comme de qualité bêta.
Avant chaque version, une grande passe de QA est exécutée, mais des instabilités
sont tout à fait possibles.

# Comment utiliser ce projet ?

Moi (Salvatore), je crois que la manière dont les projets devraient être livrés et utilisés a changé à cause de l'IA. Les principales différences aujourd'hui sont :

1. Avec l'IA, les utilisateurs peuvent modifier le logiciel de manière significative avec peu d'efforts, de coûts, et même sans connaissance approfondie du domaine de la tâche qu'ils veulent accomplir. Par exemple, un utilisateur de DwarfStar avec une configuration matérielle spécifique peut demander à un agent de codage d'améliorer la vitesse d'inférence de ce logiciel pour cette configuration matérielle précise, en demandant au modèle d'atteindre la vitesse maximale de prefill et de génération sans nuire à la correction, et en demandant aussi de réaliser une passe de QA approfondie.
2. De même, à cause du « 1 », les logiciels peuvent être livrés d'une manière différente qu'auparavant. Ils doivent davantage être un modèle fonctionnel pour les principaux cas d'usage, sans essayer de couvrir toutes les configurations possibles. Si DwarfStar présente quelques bonnes implémentations d'exécution en parallélisme de tenseurs, le code servira de rail pour implémenter la même fonctionnalité dans des conditions spécifiques, pour un nouveau modèle, et ainsi de suite.

Donc, bien que ce projet cherche à être utilisable pour les modèles mis en avant et les configurations matérielles les plus courantes, je vous demande, si vous avez accès à des agents de codage, d'envisager d'utiliser ces agents comme interface pour découvrir le projet, faire des modifications, créer des configurations personnalisées. Ainsi, vous pouvez probablement faire plus que ce que nous livrons, et certaines choses qui ne sont pas documentées ou implémentées, et dont vous avez besoin, sont potentiellement très faciles à réaliser.

## Documentation supplémentaire

Si vous cherchez des choses très spécifiques, nous avons d'autres
sous-fichiers README. Sinon, pour une utilisation normale, continuez à lire les
sections suivantes.

- [CONTRIBUTING.md](CONTRIBUTING.md) : guide de test de régression de correction
  et de vitesse pour les contributeurs. **Lisez ceci avant d'envoyer une pull request**.
- [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md) : la matrice complète des tests
  de version, y compris les machines distantes Metal, CUDA et ROCm.
- [gguf-tools/README.md](gguf-tools/README.md) : génération GGUF hors ligne,
  collecte d'imatrix, outillage de quantification et contrôles de qualité.
- [gguf-tools/imatrix/README.md](gguf-tools/imatrix/README.md) : comment
  l'imatrix des MoE routées est collectée et utilisée.
- [gguf-tools/imatrix/dataset/README.md](gguf-tools/imatrix/dataset/README.md) :
  comment le corpus de prompts de calibration est généré.
- [gguf-tools/quality-testing/README.md](gguf-tools/quality-testing/README.md) :
  comment les GGUF locaux sont notés par rapport aux continuations officielles de DeepSeek V4 Flash/PRO.
- [dir-steering/README.md](dir-steering/README.md) : données de pilotage directionnel,
  génération de vecteurs et utilisation.
- [speed-bench/README.md](speed-bench/README.md) : commandes de benchmark, graphiques,
  et génération de CSV.
- [tests/test-vectors/README.md](tests/test-vectors/README.md) : vecteurs de
  continuation officiels utilisés pour les vérifications de régression.

## Poids des modèles

Cette implémentation ne fonctionne qu'avec les GGUF DeepSeek V4 et GLM 5.2 listés
ci-dessous. Ce n'est pas un chargeur GGUF généraliste, et des fichiers GGUF
arbitraires n'auront pas la disposition des tenseurs, le mélange de quantification, les
métadonnées ou l'état MTP optionnel attendus par le moteur. Les quantifications 2 bits
fournies ici sont vérifiées comme étant réellement de haute qualité : elles se
comportent bien, fonctionnent sous des agents de codage, appellent les outils de manière fiable.

Les quants 2 bits utilisent une quantification très asymétrique : seuls les
experts MoE routés sont quantifiés, up/gate en `IQ2_XXS`, down en `Q2_K`. Ils
constituent la majorité de tout l'espace du modèle : les autres composants
(experts partagés, projections, routage) sont laissés intacts pour garantir la qualité.

Téléchargez un modèle principal. **Préférez les versions imatrix.**

```sh
./download_model.sh ds4f-q2      # 96/128 GB RAM machines
./download_model.sh ds4f-q2-q4   # q2 with the last 6 expert layers at q4
./download_model.sh ds4f-q4      # >= 256 GB RAM machines
./download_model.sh ds4f-mxfp4   # native MXFP4 experts, about 156 GB
./download_model.sh pro-q2-imatrix  # 512 GB RAM machines, PRO q2 imatrix quant
```

Le GGUF MXFP4 préserve les poids d'experts routés MXFP4 publiés par DeepSeek
plutôt que de les requantifier. Il fonctionne sur Metal et CUDA ; les
périphériques CUDA Blackwell utilisent des instructions matricielles FP4 natives
et des activations FP4 pour le travail des experts par lots.
Le décodage et les autres périphériques CUDA utilisent des activations Q8.

Pour l'exécution distribuée complète de PRO Q4, téléchargez une moitié sur chaque machine :

```sh
./download_model.sh pro-q4-layers00-30      # first half of PRO Q4 split
./download_model.sh pro-q4-layers31-output  # second half of PRO Q4 split
```

Le script télécharge depuis `https://huggingface.co/antirez/deepseek-v4-gguf`,
stocke les fichiers sous `./gguf/`, reprend les téléchargements partiels avec `curl -C -`, et
met à jour `./ds4flash.gguf` pour qu'il pointe vers le modèle principal sélectionné.
Les cibles `pro-q4-layers00-30`, `pro-q4-layers31-output` et `pro-q4-split`
téléchargent les morceaux distribués de PRO Q4 et ne mettent pas à jour `./ds4flash.gguf`.
L'authentification est optionnelle pour les téléchargements publics, mais `--token TOKEN`,
`HF_TOKEN`, ou le cache local du jeton Hugging Face sont utilisés lorsqu'ils sont présents.

Si vous voulez régénérer des fichiers GGUF ou collecter une nouvelle imatrix, voir
[gguf-tools/README.md](gguf-tools/README.md). Ces outils sont destinés au travail
hors ligne de construction de modèles et peuvent prendre beaucoup de temps sur les
poids complets de DeepSeek V4 Flash. La génération des GGUF Flash est prise en charge par
les outils locaux. La production des GGUF PRO dépend actuellement encore du flux de travail
externe basé sur `llama.cpp` ; un outillage natif pourra être ajouté plus tard.

La prise en charge de GLM 5.2 est limitée aux fichiers GGUF testés par cette branche :

```sh
./download_model.sh glm-unsloth-q4  # Unsloth UD-Q4_K_XL, 11 shards
./download_model.sh glm-antirez-iq2xxs  # antirez routed IQ2_XXS single-file GGUF
./download_model.sh glm-antirez-q2  # antirez routed Q2_K single-file GGUF
./download_model.sh glm-antirez-q4  # antirez routed Q4_K single-file GGUF
```

La disposition GLM prise en charge conserve les tenseurs denses/de contrôle du modèle
dans les chemins Q8/F32 existants et prend en charge les tenseurs gate/up des experts
routés en `Q2_K`, `Q4_K` ou `Q5_K` ; les tenseurs down des experts routés sont pris en
charge en `Q2_K`, `Q4_K`, `Q5_K` ou `Q6_K`. Les autres dispositions de quantification des
GGUF GLM doivent être considérées comme non prises en charge tant qu'elles ne sont pas
ajoutées délibérément et notées par rapport à la référence officielle de 100 cas.

Ces formats ne prennent pas tous en charge les mêmes modes d'exécution. Les fichiers Q4
fonctionnent pour l'inférence Metal et CUDA normale. Le parallélisme de tenseurs entre
deux Mac nécessite actuellement une disposition routée IQ2_XXS ou Q2_K consciente de la
propriété ; un GLM routé Q4 doit être rejeté avant l'évaluation.

Le bloc MTP de GLM fait partie du GGUF principal ; il n'utilise pas le fichier MTP
Flash séparé. Le décodage ordinaire reste le mode par défaut. `--glm-mtp` active la
spéculation gloutonne expérimentale. `--glm-mtp-timing` l'active également et affiche
les compteurs d'acceptation et de temps :

```sh
./ds4 -m gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --glm-mtp-timing --temp 0
```

L'inférence GLM utilise le backend graphe Metal, CUDA ou ROCm. Le pilotage
directionnel, `--power` en dessous de 100, un `--prefill-chunk` explicite et le
fichier externe `--mtp` ne sont pas encore pris en charge pour GLM.

Ensuite, compilez :

```sh
make                  # macOS Metal
make cuda-spark       # Linux CUDA, DGX Spark / GB10
make cuda-generic     # Linux CUDA, other local CUDA GPUs
make strix-halo       # Linux ROCm, AMD Strix Halo
make cpu              # CPU-only diagnostics build
```

`./ds4flash.gguf` est le chemin de modèle par défaut utilisé par les deux binaires. Passez `-m` pour
sélectionner un autre GGUF pris en charge depuis `./gguf/`. Exécutez `./ds4 --help` et
`./ds4-server --help` pour la liste complète des options.

## Décodage spéculatif DSpark

DSpark est un modèle de brouillon auxiliaire publié par DeepSeek pour DeepSeek V4 Flash.
Il lit les états cachés du modèle principal et propose jusqu'à cinq futurs
tokens. DwarfStar vérifie ces propositions avec le modèle Flash principal et ne valide
que le préfixe accepté. Le modèle principal reste l'autorité ; un suffixe rejeté ou
à faible confiance retombe sur le décodage cible ordinaire.

Le gain possible est une génération plus rapide : lorsque plusieurs tokens proposés sont
acceptés, une seule passe de vérification cible fait avancer le flux de plusieurs tokens.
Cela n'accélère pas le prefill, et le travail de brouillon et de vérification n'est pas
gratuit. Les continuations prévisibles, en particulier le code, tendent à en bénéficier le plus ;
les prompts à faible rendement peuvent n'être pas plus rapides, voire plus lents. DSpark est donc encore
expérimental et explicitement optionnel.

Les propositions acceptées conservent l'état produit par le vérificateur cible par lots
au lieu de repasser les mêmes tokens dans le décodage token par token. Les deux chemins
exécutent le même graphe d'inférence, mais les opérations en virgule flottante sont groupées dans
un ordre différent. Une longue exécution gloutonne de DSpark peut donc diverger d'une exécution
sans DSpark après un bloc accepté par ailleurs valide. Ce n'est pas un mode à précision réduite
ni à modèle approximé ; utilisez le décodage ordinaire, `--quality` ou
`--dspark-strict` lorsque la reproductibilité octet par octet avec le décodage token par token est
requise.

Le point de contrôle DSpark pour Flash 0731 est empaqueté ici comme un GGUF de support
distinct d'environ 5,6 Gio. Ce n'est pas un modèle autonome. Téléchargez-le une fois :

```sh
./download_model.sh ds4f-dspark
```

Le fichier de support peut être utilisé avec les modèles Flash 0731 `ds4f-q2`, `ds4f-q2-q4` et
`ds4f-q4` listés ci-dessus. Il est spécifique au point de contrôle
et ne doit pas être associé à un modèle Flash plus ancien. Pour l'instant, **DeepSeek V4 PRO**
n'est pas pris en charge. Sur Metal, le modèle principal peut être résident ou utiliser
`--ssd-streaming` ; le modèle de support ajoute quand même ses propres poids et son état
d'exécution aux besoins en mémoire. DSpark remplace l'ancien modèle de support MTP à une étape
pour cette exécution plutôt que de s'y empiler.

Exécutez-le avec un décodage glouton :

```sh
./ds4 -m ds4flash.gguf \
  --mtp gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --temp 0
```

`--mtp` fournit le GGUF de support, tandis que `--dspark` sélectionne l'exécution DSpark.
Le seuil de confiance par défaut est de `0.6` sur Metal et `0.7` sur CUDA et ROCm ;
il élague les suffixes peu susceptibles de rembourser leur coût de vérification.
`--dspark-confidence 0` force des blocs fixes de cinq tokens et est destiné au
diagnostic. Le décodage échantillonné n'utilise pas les propositions DSpark. `--quality` et
`--dspark-strict` conservent également le décodage cible uniquement, ce qui est utile pour
les vérifications de reproductibilité.

## Vitesse

Les résultats q2 actuels utilisent `ds4-bench` avec l'entrée standard *Promessi sposi*,
des pas de contexte de 2048 tokens, et 128 tokens de génération gloutonne à chaque
frontière. Chaque valeur de prefill correspond au prochain bloc de 2048 tokens. Les balayages
complets sont dans [m5_max.csv](speed-bench/m5_max.csv) et
[gb10.csv](speed-bench/gb10.csv).

| Machine | Backend | Contexte | Prefill | Génération |
| --- | --- | ---: | ---: | ---: |
| MacBook Pro M5 Max, 128 GB | Metal | 2048 | 790.18 t/s | 39.35 t/s |
| MacBook Pro M5 Max, 128 GB | Metal | 16384 | 572.53 t/s | 36.14 t/s |
| MacBook Pro M5 Max, 128 GB | Metal | 32768 | 557.04 t/s | 34.36 t/s |
| MacBook Pro M5 Max, 128 GB | Metal | 65536 | 398.50 t/s | 27.64 t/s |
| DGX Spark GB10, 128 GB | CUDA | 2048 | 825.76 t/s | 18.05 t/s |
| DGX Spark GB10, 128 GB | CUDA | 16384 | 872.44 t/s | 15.10 t/s |
| DGX Spark GB10, 128 GB | CUDA | 32768 | 855.94 t/s | 14.43 t/s |
| DGX Spark GB10, 128 GB | CUDA | 65536 | 822.98 t/s | 13.84 t/s |

Les mesures plus anciennes pour les machines et variantes de modèle non ré-exécutées dans cette passe sont
conservées à titre de référence. Elles utilisaient l'ancienne procédure de prompt en CLI et ne sont pas
directement comparables au tableau ci-dessus.

| Machine | Quant | Prompt | Prefill | Génération |
| --- | ---: | ---: | ---: | ---: |
| MacBook Pro M3 Max, 128 GB | q2 | short | 58.52 t/s | 26.68 t/s |
| MacBook Pro M3 Max, 128 GB | q2 | 11709 tokens | 250.11 t/s | 21.47 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | short | 84.43 t/s | 36.86 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | 11709 tokens | 468.03 t/s | 27.39 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | short | 78.95 t/s | 35.50 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | 12018 tokens | 448.82 t/s | 26.62 t/s |
| Mac Studio M3 Ultra, 512 GB | PRO q2 | 32768 tokens | 138.82 t/s | 9.56 t/s |

![M5 Max t/s](speed-bench/m5_max_ts.svg)
![PRO model M3 Ultra t/s](speed-bench/pro_model_m3_ultra_ts.svg)

## Exécuter des modèles plus grands que la RAM

Le chemin Metal normal essaie de rendre le modèle résident en mémoire
adressable par le GPU. C'est le chemin le plus rapide et il devrait rester votre choix par défaut
lorsque le modèle tient. DwarfStar dispose également d'un mode de capacité par **streaming SSD**
sur Metal et pour GLM 5.2 sur ROCm. Dans ce mode, les poids non routés du modèle restent résidents,
tandis que les experts MoE routés sont conservés dans un cache en mémoire et chargés depuis le fichier
GGUF en cas de défaut de cache.

Le streaming n'est pas aussi rapide que de faire tenir tout le modèle en RAM. Il a quand même besoin de mémoire
pour les poids non routés, le cache KV, le scratch du graphe, les activations et le cache
d'experts routés. Il est utile parce que les experts routés dominent la taille du modèle et que les
SSD Mac modernes sont assez rapides pour rendre les défauts de cache tolérables. Les longs prefills peuvent
rester rapides ; la génération est plus sensible aux défauts de cache car chaque nouveau token
repasse par les experts.

Commencez avec le budget de cache automatique :

```sh
./ds4 -m ./ds4flash.gguf --ssd-streaming
```

Si le démarrage signale que le cache d'experts est trop grand, ou si vous voulez réserver
plus de mémoire pour le contexte, définissez explicitement le cache d'experts routés :

```sh
./ds4 -m ./ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 32GB
```

La valeur `32GB` est un budget mémoire d'experts routés, pas un cache d'octets générique.
DwarfStar réserve d'abord une marge pour les deux couches routées complètes utilisées par le
prefill en streaming avec chevauchement, puis convertit les octets restants en nombre d'experts
dynamiques mis en cache qui tiennent pour le GGUF actuel. Les budgets `NGB` explicites peuvent
aussi être plafonnés après la comptabilité contexte/KV afin que l'ensemble de travail du backend reste hors
de la zone de pression lente. Un simple nombre tel que
`--ssd-streaming-cache-experts 4000` est différent : il signifie exactement 4000 emplacements
d'experts dynamiques, sans comptabilité supplémentaire. Les poids non routés, le cache KV, le scratch
du graphe et les activations nécessitent de la mémoire supplémentaire. Le budget de cache automatique prend
80 % de l'ensemble de travail recommandé par le backend, soustrait les poids non routés, puis
applique la même marge de prefill routé avant de dimensionner le cache dynamique. Laissez
le préchargement des experts chauds activé pour un usage normal ; utilisez `--ssd-streaming-cold` et
`--ssd-streaming-preload-experts N` uniquement pour les mesures.

### Exemples pratiques de streaming SSD

Sur les MacBook de 64 Go, commencez avec le GGUF Flash 2 bits et un cache d'experts modéré :

```sh
./download_model.sh ds4f-q2

./ds4 \
  -m ./ds4flash.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 32GB \
  --ctx 32768 \
  --nothink
```

Sur les MacBook de 128 Go, le streaming PRO q2 est expérimental mais utilisable pour l'inspection
et un travail occasionnel lorsque vous acceptez une génération lente. Commencez avec `--nothink` :

```sh
./download_model.sh pro-q2-imatrix

./ds4 \
  -m gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf \
  --ssd-streaming \
  --ctx 32768 \
  --nothink
```

Sur un M5 Max avec 128 Go de RAM, un court benchmark de décodage en streaming PRO q2 a trouvé
le budget automatique meilleur : il a sélectionné environ `59GB` de cache d'experts routés.
Les caches manuels de `64GB` à `75GB` étaient proches sur cette machine. Préférez le budget
automatique ; si vous réglez le cache manuellement sur cette classe de machine, commencez autour de
`48GB` à `64GB`, puis augmentez seulement tant que la machine reste réactive et que
le journal de démarrage affiche le cache dynamique demandé. Une fois la machine stable,
réactivez le mode réflexion avec une limite de génération conservatrice :

```sh
./ds4 \
  -m gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf \
  --ssd-streaming \
  --ctx 32768 \
  --think \
  --tokens 1500
```

GLM 5.2 utilise la même option. Son chemin de streaming conserve résident le plus grand
préfixe de couches complètes qui tient, puis utilise le budget restant pour un cache d'experts
dynamique. Commencez avec le budget automatique :

```sh
./ds4 \
  -m gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --ssd-streaming \
  --ctx 32768
```

La ligne de démarrage importante est le rapport de cache. Commencez de manière conservatrice, puis
augmentez le cache si la machine a de la marge.

Sur un Strix Halo de 128 Go, utilisez le modèle routé Q2_K et un contexte de 4096 tokens comme
point de départ. Le budget de cache automatique laisse de la place pour le graphe GLM et l'état
KV :

```sh
./download_model.sh glm-antirez-q2
make strix-halo
./ds4 --rocm -m gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf \
  --ssd-streaming --ctx 4096
```

## Inférence distribuée avec parallélisme de pipeline

Le parallélisme de pipeline permet à DwarfStar d'**exécuter un modèle trop grand pour une seule machine** en
répartissant les couches de transformeur sur plusieurs machines. L'exemple principal est le
quant Flash 4 bits complet réparti sur deux MacBook de 128 Go : chaque processus mappe seulement sa
propre tranche de couches, les activations sont envoyées via TCP, et le coordinateur conserve le
comportement CLI/API normal.

Le parallélisme de pipeline peut aussi **accélérer le prefill** en
utilisant plusieurs GPU en même temps pour traiter différents micro-lots à
différentes couches, comme dans une chaîne de montage. Seul le prefill peut être accéléré de cette
façon. La génération est purement autorégressive : chaque token doit terminer tout le
parcours avant que le token suivant puisse commencer. Le travail du modèle est le même que pour un
seul processus, plus la latence de coordination, donc la génération distribuée est plus lente.

Pour construire un premier modèle mental, voici les concepts de haut niveau :

1. Vous placez le GGUF sur chaque machine, mais chacune n'en charge qu'un sous-ensemble. `--layers` contrôle quels tenseurs sont mappés, donc un worker avec `--layers 20:output` ne charge pas les couches antérieures.
2. Les plages de couches sont inclusives : `10:20` signifie les couches 10, 11, ..., 20. `N:output` signifie de la couche `N` jusqu'à la dernière couche plus la tête de sortie.
3. Vous attribuez à l'une des machines le rôle de `coordinator`, aux autres les rôles de `workers`. Les workers se connecteront au coordinateur et signaleront leur présence ainsi que les couches qu'ils sont capables de traiter.
4. Chaque worker conserve sa tranche du cache KV.
5. La communication est de worker à worker, il n'est pas nécessaire d'utiliser le coordinateur comme relais, donc si votre coordinateur est `A` et que vous faites une requête, les activations circuleront en `A -> B -> C -> retour vers A`.

### Comment ça fonctionne et comment le configurer

Le chemin de prefill est pipeliné (c'est pourquoi il peut aller plus vite que sur une seule machine).
Pour les grands prompts, le coordinateur peut exécuter sa
tranche sur le bloc N+1 pendant que le worker exécute sa tranche sur le bloc N. Les
lignes distribuées ci-dessous ont été mesurées avec deux MacBook M5 Max de 128 Go connectés
par Thunderbolt 5, en utilisant le GGUF Flash Q4 et le bloc de prefill distribué par défaut de
4096 tokens. La colonne processus unique est une exécution de référence avec
le GGUF Q2 sur une seule machine, elle est donc un peu plus rapide puisque
les MoE routées sont plus petites.

| Prompt | Référence processus unique | Deux MacBook | Accélération |
| ---: | ---: | ---: | ---: |
| 9421 tokens | 421.70 t/s | 582.22 t/s | 1.38x |
| 28684 tokens | 405.30 t/s | 674.16 t/s | 1.66x |
| 63819 tokens | 353.62 t/s | 654.79 t/s | 1.85x |

La génération est différente. **Elle est strictement autorégressive** : le token N+1 ne peut pas commencer
tant que le token N n'a pas produit de logits et que l'échantillonnage n'a pas sélectionné le token suivant. Cela
signifie que la génération distribuée ne peut pas utiliser le long pipeline de prefill. Elle paie au
moins un saut d'activation inter-machine par token généré, donc la génération est
plus lente qu'un seul processus local. Sur la même configuration à deux Mac en Thunderbolt, une
exécution de contrôle à 12k de contexte avec le quant Flash de 91 Go est passée de 30.59 t/s
en processus unique à 24.67 t/s en distribué, soit une perte de 19,4 %. L'inférence distribuée est
donc principalement destinée à faire tenir des modèles plus grands et à accélérer les longs prefills, pas
à rendre le décodage plus rapide.

### DeepSeek V4 PRO Q4 complet sur deux Mac Studio

Le GGUF PRO Q4 en taille complète peut être exécuté sur deux Mac Studio M3 Ultra de 512 Go
en donnant au coordinateur les couches `0:30` et au worker les
couches `31:output`. Utilisez les fichiers GGUF fractionnés pour que chaque côté ne mappe que les tenseurs dont il a
besoin :

```sh
# Coordinator machine.
./download_model.sh pro-q4-layers00-30

# Worker machine.
./download_model.sh pro-q4-layers31-output
```

Les deux fichiers sont :

```text
gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf
gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf
```

C'est un cas d'usage de capacité : chaque processus ne mappe que sa propre moitié du modèle,
tandis que le worker possède la tête de sortie et renvoie les logits.

Le chemin Metal PRO Q4 actuel utilise des tables d'experts exactes résidentes dans la file pour les
grands experts routés. Cela évite les larges liaisons de tenseurs routés de plusieurs Gio
qui faisaient que les premières tentatives distribuées PRO Q4 tournaient soit très lentement, soit atteignaient les
limites de comptabilité mémoire de Metal. Lors d'un court test de fumée glouton sur le lien direct
`192.168.0.182` / `192.168.0.183`, le modèle a généré un texte cohérent et
mesuré 11.47 t/s de génération après le démarrage. La télémétrie par token était équilibrée :
les couches locales étaient autour de 39-43 ms, les couches distantes autour de 44-49 ms, pour des temps de
token totaux autour de 84-92 ms. Attendez-vous à un démarrage lent pendant que chaque côté mappe et
rend sa moitié du modèle résidente. Les performances de prefill et de décodage PRO Q4 en contexte long
nécessitent encore un benchmarking séparé.

Les mesures ci-dessus utilisent un câble Thunderbolt 5. L'implémentation est du simple
TCP et fonctionne aussi sur des liens plus lents, y compris le WiFi, mais un réseau Ethernet rapide ou
Thunderbolt est fortement recommandé. Les liens lents nuisent surtout à la latence de
génération et aux courts prefills ; les grands prefills peuvent quand même en bénéficier lorsque
la répartition des couches est équilibrée. Dans le chemin de performance normal, le dernier worker
possède la tête de sortie et renvoie les logits directement.

Configuration minimale à deux hôtes :

```sh
# Machine A: coordinator, owns tokenization, sampling, the prompt, and layers 0..30.
./ds4 \
  -m gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf \
  --role coordinator \
  --layers 0:30 \
  --listen 169.254.43.68 1234

# Machine B: worker, connects to A and owns layers 31..output.
./ds4 \
  -m gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf \
  --role worker \
  --layers 31:output \
  --coordinator 169.254.43.68 1234
```

Normalement, le worker final devrait aussi posséder la tête de sortie, par exemple
`--layers 20:output`. Cela évite de renvoyer un lot complet d'états cachés finaux
après le prefill et laisse le worker final produire les logits directement. Sur des liens très
lents ou facturés, `--layers 20:42` est aussi pris en charge : le coordinateur
chargera la tête de sortie et calculera les logits localement, échangeant un surcroît de travail du coordinateur
contre des réponses par token plus petites.

### Comparaison des liens réseau

Le tableau ci-dessous montre les mêmes deux hôtes M5 Max, le même quant Flash de 91 Go,
coordinateur `--layers 0:19`, worker `--layers 20:output`, un prompt de 8192 tokens
depuis `speed-bench/promessi_sposi.txt`, et 128 tokens générés. Les valeurs WiFi et
Internet varient avec les conditions locales, mais l'allure est la partie importante : une latence
élevée nuit directement à la génération, tandis qu'une bande passante plus faible tire aussi vers le
bas la vitesse des longs prefills.

| Lien | Adresses | Ping moyen | Prefill | Génération |
| --- | --- | ---: | ---: | ---: |
| Thunderbolt 5 | `169.254.43.68` -> `169.254.12.245` | 0.45 ms | 582.99 t/s | 25.09 t/s |
| WiFi | `192.168.1.57` -> `192.168.1.95` | 77.20 ms | 250.70 t/s | 10.70 t/s |
| Internet / VPN | `10.77.0.4` -> `10.77.0.3` | 152.10 ms | 114.88 t/s | 3.63 t/s |

Le cas Internet/VPN n'est pas censé être une bonne expérience interactive. Il reste
utile pour les tests collectifs : plusieurs personnes peuvent temporairement combiner
leurs machines pour exécuter un modèle plus grand qui ne tiendrait sur aucun hôte individuel, en acceptant
un décodage lent en échange de la simple possibilité d'inspecter le modèle.

Utilisez le coordinateur exactement comme le `./ds4` normal : chat interactif, `/read`,
et la génération ordinaire passent par la même API de session de haut niveau. Les mêmes
options distribuées sont aussi câblées dans `ds4-agent`, `ds4-eval` et
`ds4-bench`. Pour les benchmarks, les workers devraient déjà tourner ; `ds4-bench`
attend qu'un parcours complet soit disponible.

Réglages et diagnostics utiles :

```sh
./ds4-bench \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 32768 \
  --ctx-max 65536 \
  --step-incr 32768 \
  --gen-tokens 0 \
  --role coordinator \
  --layers 0:19 \
  --listen 169.254.43.68 1234 \
  --debug
```

`--debug` sur le coordinateur affiche la formation du parcours et la télémétrie par saut :
plage de couches, étendue de tokens, temps d'évaluation local, temps d'attente en aval, temps d'envoi
socket, et nombres d'octets en entrée/sortie. C'est l'outil de profilage actuel pour
décider si une répartition est équilibrée. `--dist-prefill-window N` contrôle combien
de blocs de prefill peuvent être en vol de bout en bout ; la valeur par défaut est conservatrice
et bornée. `--dist-prefill-chunk N` existe pour les expériences, mais le bloc par défaut de
4096 tokens est le réglage canonique et devrait être utilisé sauf si vous
validez explicitement une taille de bloc différente.

Par défaut, DwarfStar envoie les activations d'états cachés en flottants 32 bits. Pour réduire
le trafic, passez `--dist-activation-bits 16` ou `--dist-activation-bits 8` sur le
coordinateur. Cela ne change que le format de transport entre machines, pas les
poids du modèle ni le cache KV. Le transport 16 bits divise par deux le trafic d'activations et est la
première option à essayer sur Ethernet ou WiFi. Le transport 8 bits est plus agressif et
devrait être traité comme un mode approximatif/expérimental sauf si vous avez validé
la sortie pour votre cas d'usage. Cependant, expérimentalement, la réduction de la taille des
activations n'a pas apporté d'amélioration significative, donc cette option pourra être supprimée
à l'avenir.

**Si un worker se déconnecte, le coordinateur retire ce worker du parcours actif**.
La requête déjà en vol peut échouer, et les appels ultérieurs signalent un
parcours incomplet jusqu'à ce qu'un worker compatible se reconnecte et envoie une nouvelle
inscription. Pour les sessions en direct, le coordinateur conserve l'historique des tokens et peut
reconstruire l'état KV du worker en rejouant le préfixe lorsque le parcours est de nouveau
disponible. Les workers valident aussi un hachage roulant de préfixe de tokens sur 64 bits sur chaque élément de
travail, donc un worker redémarré à la position 0 ne peut pas accepter silencieusement du travail pour la
position N ; il signale la divergence et le coordinateur rejoue la transcription actuelle. Ctrl+C dans la
CLI et l'agent est coopératif : DwarfStar attend que le token distribué ou le bloc de prefill
en cours se vide avant de rendre le contrôle,
ce qui évite les scissions de KV causées par le coordinateur. Les sessions agent/serveur sauvegardées utilisent le
même format de fichier KV que les sessions sur une seule machine : lors de la sauvegarde, le coordinateur
récupère les tenseurs de couches détenus par les workers et sérialise une charge utile normale ; lors du
chargement, il répartit cette charge utile sur le parcours actuellement inscrit.

### Aperçu du protocole distribué

Au niveau du protocole, il existe deux types de connexions. Les workers gardent une
connexion de contrôle TCP ouverte vers le coordinateur et envoient un `HELLO` avec leur
ID de modèle, la famille de modèle, le profil de quantification, la tranche de couches, la capacité de contexte et le
port de données. Le coordinateur utilise ces inscriptions pour construire un parcours qui couvre toutes les
couches. Le travail se déplace ensuite sur des connexions de données TCP à faible latence : le coordinateur
calcule la première tranche, envoie une trame `WORK` avec l'ID de session, les positions de tokens,
les hachages roulants de préfixe de tokens avant et après l'étendue, les informations de parcours et la
charge utile d'états cachés, et chaque worker calcule sa tranche. Les workers du milieu peuvent
transmettre directement au worker suivant. Le worker final renvoie les logits au
coordinateur, ou des ACK pour les blocs de prefill non finaux afin que le pipeline de prefill puisse
rester plein. Les trames `RESULT` renvoient l'ID de requête et le hachage post-étendue. Une erreur de
statut de worker est gérée différemment d'une panne de socket : une divergence de KV/hachage peut
être récupérée en rejouant l'historique des tokens sur le même parcours, tandis qu'une panne de
transport abandonne le parcours et attend un worker de remplacement. Pour la KV persistante, le
coordinateur ouvre des connexions de données de worker et envoie des messages de sauvegarde/chargement d'instantané
pour chaque plage de couches détenue par un worker ; la charge utile sur disque reste un seul
fichier de cache agent/serveur. Le protocole n'a aucun
chiffrement ni authentification, et n'est pas encore stable pour les versions ; le coordinateur et les
workers devraient être compilés à partir du même commit et utilisés sur des machines de confiance et
des réseaux de confiance.

## Parallélisme de tenseurs sur RDMA

Le parallélisme de tenseurs exécute un seul décodage sur deux Mac connectés par un
câble Thunderbolt 5, répartissant le lourd travail par couche entre les deux
GPU et échangeant des sommes partielles de 16-24 Ko aux portes de synchronisation à l'intérieur du
graphe (RDMA sur Thunderbolt lorsque disponible, un socket TCP dédié
sinon). Contrairement au mode distribué pipeliné ci-dessus, les deux
machines travaillent sur le *même token en même temps*, donc cela réduit
la latence par token au lieu de seulement faire tenir un plus grand modèle.

Chaque machine garde une moitié contiguë des experts routés résidente. Les poids denses,
d'attention, d'experts partagés, d'embedding et de sortie restent répliqués.
Cela permet à un modèle dont les experts routés ne tiennent pas sur une seule machine de s'exécuter entièrement
résident sur la paire ; les kernels routés ne touchent jamais la moitié d'experts du pair.

### Exécuter GLM 5.2 sur deux MacBook de 128 Go

Configuration unique par démarrage, sur les **deux** machines :

```sh
# Let the GPU wire ~117 GB (default cap is ~75% of RAM; the resident
# expert shard needs ~97.5 GiB plus KV/scratch).
sudo sysctl iogpu.wired_limit_mb=120000

# RDMA over Thunderbolt needs an IPv4 address directly on the cabled
# member interface (the bridge IP does not count). Use the interface
# that is 'active' in ifconfig, e.g. en1 on one side and en6 on the
# other. Skip this if you are fine with the TCP fallback.
sudo ifconfig en1 inet 10.99.0.2/30 alias     # machine A
sudo ifconfig en6 inet 10.99.0.1/30 alias     # machine B
```

Vérifiez le périphérique verbs avant de charger le modèle :

```sh
rdma_ctl status
ibv_devinfo -v
```

Le périphérique doit être actif et exposer le GID mappé IPv4 pour l'adresse ci-dessus,
par exemple `::ffff:10.99.0.2`. Un ping IP fonctionnel ne prouve pas que RDMA est
actif.

Les deux machines ont besoin du même arbre, du même commit et du même chemin GGUF. Le parallélisme de tenseurs est
toujours une répartition 50/50 avec un seul worker, donc ne passez pas `--layers`. Démarrez le worker
en premier ; il réessaie pendant que le coordinateur charge. Le worker doit composer l'adresse
sur l'interface membre Thunderbolt, pas l'adresse du pont :

```sh
MODEL=gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf

# Machine B: worker.
./ds4 -m "$MODEL" --tensor-parallel --role worker \
  --coordinator 10.99.0.2 9911 --transport rdma

# Machine A: coordinator.
./ds4 -m "$MODEL" --tensor-parallel --role coordinator \
  --listen 10.99.0.2 9911 --transport rdma -c 8192 \
  -p "Tell me something about the sea."
```

Le périphérique verbs actif et le GID mappé IPv4 sont sélectionnés automatiquement. Si cela
est ambigu, ajoutez `--rdma-device rdma_en6 --rdma-gid-index 1` sur le worker et
les options `rdma_en1` correspondantes sur le coordinateur. Utilisez `--transport tcp` des deux
côtés pour forcer TCP. Les rôles de parallélisme de tenseurs sont actuellement exposés par la CLI `ds4`,
pas par `ds4-server` ni `ds4-agent`.

Le démarrage prend environ 9 secondes par machine : chaque rang pré-charge son
fragment de ~100 Gio depuis le SSD et l'épingle via un ensemble de résidence Metal.
DeepSeek V4 Flash fonctionne de la même façon avec son propre GGUF sur les deux machines.
Les vecteurs gate de DeepSeek font 16 Ko et voyagent en un seul message RDMA. Les
vecteurs de 24 Ko et 6144 de largeur de GLM sont découpés en deux messages RDMA ordonnés.

Mesuré sur deux MacBook M5 Max de 128 Go (GLM 5.2, IQ2_XXS, 188 Gio) :

| | deux Mac, parallélisme de tenseurs | un Mac, streaming SSD |
|---|---|---|
| décodage | ~16.8 t/s (15.4 à 4k de contexte) | ~4.8 t/s |
| prefill (4096 tokens) | ~94 t/s | ~3-5 t/s |
| résidence | entièrement résident en mémoire | streame les experts depuis le SSD |

Notes : le coordinateur reflète chaque synchronisation de prompt et chaque évaluation vers le worker, donc
les deux caches KV restent parfaitement synchronisés ; le traitement du prompt répartit à la fois les
GEMM d'experts routés (par propriété d'expert) et les têtes d'attention (une
moitié contiguë par machine) avec un seul échange de sommes partielles en masse par
couche et par étape (`--tensor-parallel-token-prefill` sélectionne un prefill token par token plus
lent qui correspond exactement à l'arithmétique d'une seule machine).
Le graphe scindé est déterministe, mais son ordre de réduction en virgule flottante modifié
n'est en général pas identique octet par octet à l'exécution sur une seule machine.

## Parallélisme de tenseurs entre GPU CUDA

Sur un seul serveur CUDA, `--cuda-tensor-parallel` répartit le travail de tenseurs et
d'experts routés de DeepSeek V4 Flash sur un nombre pair de GPU. C'est distinct
du mode Mac-à-Mac ci-dessus : il n'utilise pas `--role`, RDMA, ni le pipeline
de couches distribué. Le placement des GPU et les budgets mémoire sont sélectionnés avec
les options normales `--gpu-devices` et `--gpu-vram`.

L'ordre des périphériques est significatif. Avec `N` périphériques, les `N/2` premiers niveaux
logiques sont des foyers contigus du pipeline de couches et les `N/2` seconds niveaux sont leurs
partenaires en parallélisme de tenseurs. Spécifiez d'abord tous les foyers puis tous les partenaires, avec
la paire P2P la plus proche à des positions correspondantes. Par exemple, l'hôte L40S testé
utilise les paires physiques `(0,1)`, `(2,3)`, `(4,5)` et `(6,7)`, exprimées par
`0,2,4,6,1,3,5,7`. Chaque paire stocke une répartition 50/50 des experts routés, et
la tête de vocabulaire est répartie par lignes sur les niveaux de sortie participants.
Ces grands tenseurs ne sont pas dupliqués. Les poids d'attention dense, de routeur et
d'experts partagés sont répliqués au sein de chaque paire.

Pour un débit maximal sur huit cartes L40S de 48 Go, utilisez le modèle imatrix Q4.
Sa disposition routée `Q4_K` dispose des kernels multi-session groupés natifs ; le modèle Q2
est le choix à plus faible mémoire (y compris les exécutions testées à quatre cartes), mais ses
formes routées groupées non prises en charge utilisent le repli exact et ont un débit de service
agrégé plus faible. Téléchargez et compilez la cible L40S avec :

```sh
./download_model.sh ds4f-q4
make cuda CUDA_ARCH=sm_89
```

Voici la configuration agent-interactif utilisée sur le serveur à huit L40S :

```sh
MODEL=gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf

./ds4-agent --cuda --cuda-tensor-parallel \
  --gpu-vram auto \
  --gpu-devices 0,2,4,6,1,3,5,7 \
  --model "$MODEL" \
  --ctx 100000
```

Pour le service, gardez plusieurs sessions KV résidentes afin que les lignes de décodage puissent être groupées
entre les requêtes. L'hôte testé est configuré pour un maximum de 16 sessions résidentes :

```sh
./ds4-server --cuda --cuda-tensor-parallel \
  --gpu-vram auto \
  --gpu-devices 0,2,4,6,1,3,5,7 \
  --model "$MODEL" \
  --ctx 100000 \
  --batched-session 16 \
  --host 0.0.0.0
```

Les lanceurs locaux équivalents sont `./run-nvidia-tp-agent.sh` et
`./run-nvidia-tp-server.sh`. Le lanceur de serveur active aussi le cache KV sur disque
et utilise par défaut le GGUF MXFP4 0731 natif. Définissez `DS4_MODEL` pour utiliser le fichier Q4
ci-dessus à la place. Réduisez le nombre de sessions ou la taille du contexte si les caches KV
résidents demandés ne tiennent pas après le chargement du modèle. La TP CUDA, la propriété d'experts
à moitié résidente, la répartition de sortie, le prefill pipeliné et le décodage groupé compatible sont
sélectionnés par `--cuda-tensor-parallel` ; aucun réglage d'environnement `DS4_CUDA_*` n'est requis.
Sans `--prefill-chunk` explicite, ce mode utilise des blocs de 2048 tokens afin que la disposition
testée à 16 sessions et 100k de contexte conserve assez de VRAM pour les caches KV résidents.
Un `--prefill-chunk` explicite reste une surcharge pour d'autres topologies.

Tout nombre pair de cartes pouvant contenir le modèle sélectionné et le scratch du graphe est une
topologie valide. Sur cette classe de carte de 48 Go, les extrémités mesurées utiles sont
Q2 sur quatre cartes (deux étapes de pipeline) et Q4 sur huit cartes (quatre étapes).
Pour un sous-ensemble de quatre cartes appariées en PIX tel que les GPU physiques `0,1,4,5`, la liste
ordonnée est `0,4,1,5`. Deux cartes n'ont pas assez de mémoire pour ces modèles Flash.

Ce mode nécessite actuellement DeepSeek V4 Flash et un placement multi-GPU
pair. GLM 5.2 utilise à la place un placement de couches normal sur les périphériques CUDA
sélectionnés. Le DGX Spark est une cible à GPU unique et ne doit pas être démarré avec
`--cuda-tensor-parallel`.

## Réduire la chaleur, la consommation d'énergie et le bruit du ventilateur

Les longues exécutions d'inférence locale peuvent tenir le GPU occupé pendant des périodes prolongées. Si vous
vous souciez plus de la chaleur, du bruit du ventilateur, de l'autonomie de la batterie sur les MacBook, ou de la
réduction du stress thermique sur le matériel que du débit maximal, utilisez `--power N`.

`--power 100` est la valeur par défaut et signifie pleine vitesse. Des valeurs plus basses demandent à DwarfStar de viser
ce pourcentage d'utilisation du GPU : `--power 70` vise environ 70 %, `--power 50`
vise environ la moitié de l'utilisation, et ainsi de suite. DwarfStar fait cela en mesurant le temps de travail du GPU
et en insérant de petites pauses entre les unités de travail : pendant le prefill, il fait une pause entre
les couches, et pendant la génération, il fait une pause entre les tokens décodés. Cela réduit
la charge soutenue sans changer la sortie du modèle.

L'option est disponible sur la CLI, le serveur, l'agent, l'évaluation et les outils de benchmark
pour les modèles DeepSeek. GLM 5.2 n'accepte actuellement que `--power 100`. Par exemple :

```sh
./ds4 --power 50
./ds4-agent --power 70
./ds4-server --power 40 --ctx 100000
```

## Agent natif

DwarfStar propose un agent de codage natif qui fonctionne d'une manière différente
de la plupart des autres systèmes : l'inférence est contrôlée depuis l'agent
lui-même, sans frontières de socket/API, de sorte que la session est représentée
par le cache KV sur disque lui-même. De plus, les outils et le prompt système
sont tous conçus verticalement pour DeepSeek v4 Flash et PRO. Cela offre
quelques avantages :

* Expérience à faible latence, bornée principalement par les limites de vitesse de prefill. L'affichage du texte généré, l'appel d'outils, le début d'une nouvelle session sont toujours instantanés.
* Barre de progression en direct pendant le temps de prefill.
* Aucune conversion d'appel d'outils DSML, les outils sont gérés nativement dans le format du LLM.
* Les divergences de cache KV sont impossibles par construction, l'état actuel est toujours la vérité.
* Tout est réglé pour ce modèle.
* Possibilité de basculer entre les sessions sauvegardées avec `/list` et `/switch` ; les sessions KV complètes reprennent sans étape de prefill.

Les sessions de l'agent sont stockées dans `~/.ds4/kvcache`. Utilisez `/save` pour persister la
session actuelle, `/list` pour afficher les sessions sauvegardées triées par date de mise à jour récente,
et `/switch <sha>` pour en reprendre une. L'ID de session est stable à travers les
sauvegardes futures et est dérivé du premier prompt utilisateur et de l'heure de création.
`/del <sha>` supprime une session sauvegardée. `/strip <sha>` conserve le texte rendu de la
conversation et le titre mais retire la lourde charge utile KV ; basculer vers une
session dépouillée reconstruit le cache KV en préremplissant le texte sauvegardé.

Utilisez `--chdir /path/to/ds4` lors du lancement de `ds4-agent` depuis un autre répertoire,
afin que les fichiers d'exécution relatifs tels que `metal/*.metal` se résolvent depuis l'arbre du projet.

Cependant, bien que le système fonctionne déjà, il y a beaucoup de travail à faire
pour le rendre prêt pour le grand public. Lorsque l'agent atteindra finalement
la forme voulue, nous scinderons *probablement* le serveur et le client en créant un protocole
avec état basé sur les sessions qui pourra recréer tout cela d'une manière client-serveur.

## Analyse comparative

`ds4-bench` mesure le débit instantané de prefill et de génération aux frontières de
contexte au lieu de rapporter une seule moyenne sur toute l'exécution. Il charge le modèle une fois,
parcourt une séquence de tokens fixe jusqu'à des frontières telles que 2048, 4096, 6144, et utilise
un prefill incrémental afin que chaque ligne ne mesure que l'intervalle de tokens nouvellement ajouté.
Après chaque frontière, il sauvegarde l'état KV en direct en mémoire, génère une sonde gloutonne
fixe non-EOS, restaure l'instantané mémoire et poursuit le prefill.

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Le fichier d'exemple est un texte nettoyé du domaine public issu du Project Gutenberg de
*I Promessi Sposi* d'Alessandro Manzoni (ebook #45334), avec l'en-tête et le pied de page
Gutenberg supprimés : <https://www.gutenberg.org/ebooks/45334>.

Utilisez `--step-incr N` pour un espacement linéaire différent, ou `--step-mul F` pour des balayages
exponentiels. La sortie est du CSV avec une ligne par frontière : tokens/s du dernier intervalle
de prefill, tokens/s de génération à cette frontière, et
`kvcache_bytes`.

Les sessions préremplissent les longs prompts en blocs de 4096 tokens par défaut. Utilisez
`--prefill-chunk 2048`, par exemple, pour correspondre au chemin strict du point de contrôle des
vecteurs officiels. Changer le bloc change le chemin de point de contrôle KV/logit, donc
comparez-le comme une configuration d'exécution explicite.
Le prefill Metal par blocs réutilise le même graphe orienté par couches capable de gérer des plages pour chaque
bloc, préservant les frontières absolues du compresseur/indexeur tout en évitant l'ancien
chemin de répartition par bloc et par couche.

## Évaluation des capacités

`ds4-eval` est un petit benchmark d'intégration sur modèle réel. Ce n'est pas un exécuteur de
classement et il ne doit pas être rapporté comme un score officiel de benchmark GPQA, SuperGPQA, AIME ou
de sécurité : les questions sont un sous-ensemble intégré de 92 éléments choisi
pour rendre les tests de régression locaux utiles et inspectables visuellement. Le programme
charge le vrai GGUF, rend les prompts de chat DeepSeek, diffuse les tokens échantillonnés dans un TUI en écran divisé, note
la réponse finale, et affiche un rapport par question avec les tokens du prompt,
les tokens générés, l'état réussite/échec, la réponse du modèle et la réponse correcte.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
```

L'exécution par défaut utilise `--tokens 16000`, le mode réflexion activé, et une limite budgétaire
souple/dure `</think>` afin que le modèle ait de la place pour produire une réponse visible.
`ds4-eval` dimensionne le contexte en interne à partir du plus grand prompt sélectionné plus
le budget de génération, et refuse les exécutions qui nécessiteraient plus de 1M de tokens de contexte.
Appuyez sur `p` pour mettre en pause, `q` pour quitter et afficher le rapport, Haut/Bas pour
inspecter ou sélectionner une autre question, et Entrée pour exécuter ensuite la question sélectionnée.
`--plain` désactive le TUI.

Utilisez `--regrade-trace /path/to/trace.txt` pour rejouer l'extracteur de réponse actuel
et le noteur par rapport à un fichier `--trace` antérieur sans charger le modèle
ni régénérer de tokens. C'est utile pour auditer les changements de l'évaluateur : cela
montre quels cas ont changé, l'ancienne réponse retenue, la nouvelle réponse retenue, et un
récapitulatif réussite/échec.

Pour les changements d'inférence qui peuvent affecter la dérive de génération, gardez cette barrière
déterministe de comptage de tokens q1..q4 dans le plan de test :

```sh
./ds4-eval \
  -m ds4flash.gguf \
  --plain \
  --questions 4 \
  --tokens 2048 \
  --temp 0 \
  --seed 1
```

Les comptages de tokens générés doivent rester alignés avec la référence :

| Question | État attendu | Tokens générés attendus | Donné/correct attendu |
|---:|---|---:|---|
| 1 | `PASSED` | 2048 | `B` / `B` |
| 2 | `PASSED` | 438 | `C` / `C` |
| 3 | `PASSED` | 666 | `70` / `70` |
| 4 | `FAILED` | 2048 | `A` / `C` |

Les 75 premières questions intégrées sont entrelacées en 25 GPQA Diamond, 25 SuperGPQA
audités, et 25 problèmes AIME 2025. Les 17 dernières sont un sous-ensemble COMPSEC audité
de questions de localisation de vulnérabilités C/C++ réduites à une seule fonction.
On demande au modèle la meilleure ligne source unique, ou le plus petit ensemble exact de lignes
seulement lorsque le bug ne peut pas être localisé sur une seule ligne ; le noteur accepte de petites
plages auditées uniquement lorsque les lignes adjacentes sont des emplacements équivalents pour le même
bug. L'ordre est
intentionnellement progressif : les premières questions sont des tests de fumée utiles, tandis que les
questions ultérieures sont assez difficiles pour qu'un modèle de raisonnement solide en manque encore
certaines. La tranche SuperGPQA est sélectionnée plutôt qu'aveugle : les lignes en amont avec
des clés erronées, des figures manquantes ou des prompts sous-spécifiés sont remplacées par des lignes plus
propres.

L'ensemble devrait être traité comme une suite de régression de capacités difficile plutôt que
comme un test unitaire réussite/échec.

- **GPQA Diamond** apporte des questions scientifiques de niveau doctoral avec des
  réponses à choix multiples. La fiche modèle de DeepSeek rapporte de bons résultats
  sur GPQA Diamond complet en mode réflexion, mais les éléments individuels nécessitent quand même
  un raisonnement soigné en physique, chimie ou biologie et sont faciles à perdre avec une
  petite régression de prompt/rendu ou d'échantillonnage.
- **SuperGPQA** apporte de larges connaissances spécialisées et des questions de transfert
  de domaine. Le chiffre SuperGPQA de la fiche modèle est bien plus bas que celui de GPQA Diamond,
  donc ces éléments sont censés être inégaux : certains semblent banals, d'autres nécessitent une
  connaissance professionnelle de niche ou une interprétation exacte d'une question d'examen de style
  traduit.
- **AIME 2025** apporte des maths de concours à réponse exacte. Ce sont souvent les éléments les plus
  impitoyables de l'ensemble : pas d'a priori à choix multiples, pas de note partielle, et
  un seul dérapage arithmétique ou algébrique change la note.
- **COMPSEC** apporte des éléments de raisonnement de sécurité C/C++ à fonction unique
  réduits à partir de comptes rendus publics de CVE. Ce ne sont pas des prompts d'exploitation : la tâche est
  d'identifier la meilleure ligne source où le défaut de code défensif est introduit,
  ou de renvoyer `0` pour une fonction sûre.

En pratique, cela signifie que `ds4-eval` ne devrait pas être censé produire une exécution parfaite
92/92. Il est destiné à répondre à une question d'ingénierie plus utile : après un
changement de kernel, de quantification, de rendu de prompt, de cache KV ou de streaming d'outils, est-ce que
DeepSeek V4 Flash résout encore un mélange représentatif de problèmes difficiles de science, de large
connaissance, de maths exactes et de code de sécurité tout en utilisant le même chemin d'inférence
que celui exécuté par les utilisateurs ?

## CLI

Prompt unique :

```sh
./ds4 -p "Explain Redis streams in one paragraph."
```

Sans `-p`, le prompt interactif démarre :

```sh
./ds4
ds4>
```

La CLI interactive est un vrai chat multi-tour. Elle conserve la transcription de chat
rendue et le point de contrôle KV du graphe en direct, donc chaque tour prolonge la conversation
précédente. Les commandes utiles sont `/help`, `/think`, `/think-max`, `/nothink`,
`/ctx N`, `/read FILE` et `/quit`. Ctrl+C interrompt la génération en cours
et retourne à `ds4>`.

La CLI utilise le mode réflexion par défaut. Utilisez `/nothink` ou `--nothink` pour des réponses
directes. `--mtp MTP.gguf --mtp-draft 2` active le chemin spéculatif MTP optionnel ;
il n'est utile que pour le décodage glouton, utilise actuellement une porte de confiance
(`--mtp-margin`) pour éviter les acceptations partielles lentes, et devrait être traité comme un
chemin expérimental à légère accélération.

## Serveur

Démarrez un serveur local compatible OpenAI/Anthropic :

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Utilisez `--chdir /path/to/ds4` lors du lancement de `ds4-server` depuis un autre répertoire,
afin que les fichiers d'exécution relatifs tels que `metal/*.metal` se résolvent depuis l'arbre du projet.

Par défaut, le serveur garde un point de contrôle backend/KV mutable en mémoire, de sorte que les
clients sans état qui renvoient une version plus longue du même prompt peuvent réutiliser le
préfixe partagé au lieu de préremplir depuis le token zéro.

`--batched-session N` préalloue `N` sessions KV résidentes indépendantes. Les étapes de
décodage prêtes sont évaluées ensemble, tandis que les longs prefills alternent par blocs bornés
afin qu'une requête ne bloque pas tous les décodeurs. Les requêtes au-delà de `N` attendent
un emplacement résident. Si la mise en cache KV sur disque est activée, un emplacement inactif est persisté
avant réutilisation et peut être restauré lorsque cette conversation revient ; une requête
active n'est jamais évincée. Choisissez `N` et `--ctx` afin que toutes les allocations KV résidentes
tiennent en mémoire GPU. Sans cette option, l'inférence conserve le comportement original
à session unique.

Pendant que la génération est active, le prefill cède la place tous les 128 tokens par défaut.
`--mixed-prefill-quantum N` change cet intervalle pour les tests ; des valeurs plus grandes
réduisent les transferts d'ordonnancement mais peuvent faire attendre plus longtemps les décodeurs actifs.

Le batching de décodage est exact : lorsqu'un kernel batché natif n'est pas disponible,
DwarfStar exécute les lignes concernées dans un ordre fixe et renvoie les mêmes logits
complets que des évaluations de session séparées. Le comportement actuel du backend est :

| Backend et modèle | Exécution de session |
| --- | --- |
| Metal, DeepSeek Flash résident | Batching natif des experts partagés et QKV à partir de deux lignes lorsque pris en charge ; repli ordonné sinon. |
| Metal, GLM 5.2 | Repli exact ordonné. |
| CUDA, DeepSeek Flash sur une disposition multi-GPU TP/EP prise en charge | Décodage natif et prefill/décodage mixte, avec des replis exacts pour les formes de kernel non prises en charge. |
| CUDA GPU unique, y compris DGX Spark | Repli exact ordonné. |

`N` sessions résidentes allouent `N` états KV, donc une taille de contexte qui tient une fois
peut ne pas tenir huit fois. Le batching natif peut améliorer le débit agrégé ; un
repli ordonné fournit la concurrence et l'équité, mais pas la même accélération.
Le décodage spéculatif MTP est désactivé pendant que le batching natif de session est actif.

Points de terminaison pris en charge :

- `GET /v1/models`
- `GET /v1/models/deepseek-v4-flash`
- `GET /v1/models/deepseek-v4-pro`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/messages`

Les points de terminaison des modèles Flash et PRO sont des alias de compatibilité. Ils rapportent tous les deux
le modèle actuellement chargé depuis le GGUF passé avec `-m` ; le nom du point de terminaison ne
sélectionne pas un modèle différent.

`/v1/chat/completions` accepte les habituels `messages` de style OpenAI,
`max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
`seed`, `stream`, `stream_options.include_usage`, `tools` et `tool_choice`.
Les schémas d'outils sont rendus dans le format d'outils DSML de DeepSeek, et les appels d'outils DSML
générés sont remappés vers des appels d'outils OpenAI.

`/v1/responses` accepte les `input`, `instructions`, `tools`, `tool_choice`,
`max_output_tokens`, `temperature`, `top_p`, `stream` et `reasoning` de style OpenAI Responses.
C'est le point de terminaison préféré pour Codex CLI. Le serveur garde les
continuations Responses liées à l'état en direct lorsque c'est possible, et peut retomber sur
le même rendu DSML et la même réutilisation de préfixe KV que les chat completions.

`/v1/messages` est le point de terminaison compatible Anthropic utilisé par les clients de style Claude Code.
Il accepte `system`, `messages`, `tools`, `tool_choice`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `stream`, `stop_sequences` et les contrôles de
réflexion. Les utilisations d'outils sont renvoyées sous forme de blocs `tool_use` Anthropic.

La génération API échantillonnée par défaut utilise `temperature=1`, `top_p=1` et
`min_p=0.05`, donc le filtre par défaut est la probabilité relative plutôt que la
masse du noyau. En mode réflexion, DwarfStar applique ces valeurs d'échantillonnage fixes
à tout paramètre que la requête omet, correspondant au comportement d'API à réflexion fixe de DeepSeek,
mais les paramètres d'échantillonnage définis explicitement dans la requête l'emportent toujours : une
requête `temperature=0` est gloutonne à travers toute la phase de raisonnement, donc les
harnais de benchmark obtiennent une sortie déterministe en mode réflexion.

Les points de terminaison chat, Responses et Anthropic prennent en charge le streaming SSE. En mode
réflexion, le raisonnement est diffusé dans la forme native de l'API au lieu d'être mélangé au
texte final. Le streaming chat OpenAI
diffuse aussi les appels d'outils dès que l'invocation DSML est reconnue : l'en-tête de l'outil
est envoyé en premier, puis les octets de paramètres sont transmis sous forme de deltas
`tool_calls[].function.arguments` pendant que la génération continue. Le
point de terminaison Anthropic diffuse la réflexion et le texte en direct, puis émet des blocs
`tool_use` structurés lorsque le bloc d'outils généré est complet.
Le point de terminaison Responses diffuse le cycle de vie des événements Responses attendu par Codex,
y compris `response.output_text.delta`, les événements d'arguments d'appel de fonction, et les
événements terminaux `response.completed` / `response.incomplete` / `response.failed`.

Pour les clients JavaScript de navigateur servis depuis une autre origine, démarrez le serveur avec
`--cors` pour émettre les en-têtes `Access-Control-Allow-*`. Cela ne change que les en-têtes HTTP ;
cela n'expose pas le serveur sur le LAN. Utilisez `--host 0.0.0.0`
explicitement lorsque des machines distantes doivent pouvoir se connecter.

### Gestion des appels d'outils et canonicalisation

DeepSeek V4 émet les appels d'outils sous forme de [texte DSML](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/README.md). Les clients agents ne renvoient pas ce
même texte lors de la requête suivante : ils envoient des objets d'appel d'outil JSON
OpenAI/Anthropic normalisés. **Si le serveur re-rendait ces objets de façon légèrement
différente, le préfixe d'octets rendu ne correspondrait plus au point de contrôle KV en
direct** et le tour suivant devrait être reconstruit.

La première ligne de défense est la relecture exacte. Chaque appel d'outil reçoit un
ID d'outil API impossible à deviner, et le serveur mémorise `tool id -> bloc DSML exact échantillonné` dans
une carte en mémoire bornée soutenue par des arbres radix. Lorsque le client renvoie plus tard cet
ID d'outil, le générateur de prompt utilise les octets DSML exacts que le modèle a échantillonnés,
pas une approximation fraîchement formatée. Cette carte peut aussi être sauvegardée dans les fichiers de
cache KV, donc la relecture exacte survit aux redémarrages du serveur pour les historiques mis en cache.

**La canonicalisation n'est que le chemin de secours**. Si le bloc DSML exact est manquant,
ou si la relecture exacte est désactivée avec `--disable-exact-dsml-tool-replay`, le serveur
rend une forme DSML déterministe à partir de l'objet d'outil JSON. Après un tour d'appel
d'outil, il compare le flux de tokens échantillonné en direct avec le prompt que la requête client
suivante rendra. Si nécessaire, il réécrit le point de contrôle en direct, ou
retombe sur un instantané KV disque plus ancien et ne rejoue que le suffixe. Cela garde
la continuation du modèle alignée avec la transcription d'API sans état.

Pendant la génération, le serveur traite aussi la syntaxe DSML différemment de la charge utile.
Lorsque le modèle émet une structure de protocole stable telle que des balises DSML, des
en-têtes de paramètres, de la ponctuation JSON ou des marqueurs de fermeture, l'échantillonnage est forcé à
`temperature=0` afin que l'appel d'outil reste analysable. Ce mode glouton ne s'**applique pas**
aux charges utiles d'arguments : les corps de paramètres `string=true` et les valeurs de chaîne JSON,
y compris le contenu des fichiers et le texte d'édition, utilisent les paramètres d'échantillonnage normaux
de la requête. Cette séparation est importante : le décodage déterministe est utile pour
la syntaxe, mais peut créer du texte répété lorsqu'il est appliqué à de longs corps de code ou de fichiers.

Exemple OpenAI minimal :

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash",
    "messages":[{"role":"user","content":"List three Redis design principles."}],
    "stream":true
  }'
```

### Utilisation par un client agent

`ds4-server` peut être utilisé par des agents de codage locaux qui parlent le protocole chat
completions compatible OpenAI. Démarrez le serveur d'abord, et fixez la limite de contexte du client à une valeur
pas plus élevée que la valeur `--ctx` avec laquelle vous avez démarré le serveur :

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Vous pouvez utiliser un contexte plus grand et un cache plus grand si vous le souhaitez. Un contexte complet de
1M de tokens va utiliser plus ou moins 26 Go de mémoire (l'indexeur compressé
à lui seul fera environ 22 Go), donc configurez un contexte qui a du sens dans
votre système. Avec 128 Go de RAM, vous exécuteriez les quants 2 bits, qui font
déjà 81 Go, 26 Go vont être probablement trop, donc une fenêtre de contexte
de 100~300k tokens est plus sage. Cependant, des utilisateurs ont rapporté avoir pu exécuter des
quants 2 bits avec une fenêtre de contexte de 250k sur des Mac avec seulement 96 Go de mémoire système : assurez-vous
de tuer les processus qui utilisent trop de mémoire, si vous prévoyez de le faire ;)

La limite de sortie `384000` ci-dessous évite les plafonds de tokens puisque le modèle est capable de
générer des réponses très longues autrement (jusqu'à 384k tokens). Le serveur
s'arrête quand même lorsque la fenêtre de contexte configurée est pleine.

Pour **opencode**, ajoutez une entrée de fournisseur et d'agent à
`~/.config/opencode/opencode.json` :

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ds4": {
      "name": "ds4.c (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dsv4-local"
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "limit": {
            "context": 100000,
            "output": 384000
          }
        }
      }
    }
  },
  "agent": {
    "ds4": {
      "description": "DeepSeek V4 Flash served by local ds4-server",
      "model": "ds4/deepseek-v4-flash",
      "temperature": 0
    }
  }
}
```

Pour **Pi**, ajoutez un fournisseur à `~/.pi/agent/models.json` :

```json
{
  "providers": {
    "ds4": {
      "name": "ds4.c local",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "dsv4-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "deepseek",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [
        {
          "id": "deepseek-v4-flash",
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh"
          },
          "input": ["text"],
          "contextWindow": 100000,
          "maxTokens": 384000,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Optionnellement, faites-en le modèle Pi par défaut dans `~/.pi/agent/settings.json` :

```json
{
  "defaultProvider": "ds4",
  "defaultModel": "deepseek-v4-flash"
}
```

Pour **Codex CLI**, utilisez l'API filaire Responses :

```toml
[model_providers.ds4]
name = "DS4"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

Puis exécutez :

```sh
codex --model deepseek-v4-flash -c model_provider=ds4
```

Pour **Claude Code**, utilisez le point de terminaison compatible Anthropic. Un wrapper comme celui-ci
correspond à la configuration locale `~/bin/claude-ds4` :

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_AUTH_TOKEN="dsv4-local"
export ANTHROPIC_MODEL="deepseek-v4-flash"

export ANTHROPIC_CUSTOM_MODEL_OPTION="deepseek-v4-flash"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="DeepSeek V4 Flash local ds4"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="ds4.c local GGUF"

export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$HOME/.local/bin/claude" "$@"
```

Claude Code peut envoyer un grand prompt initial, souvent autour de 25k tokens, avant de
commencer à faire du travail utile. Gardez `--kv-disk-dir` activé : après le premier prefill
coûteux, le cache KV sur disque permet aux continuations ultérieures ou aux sessions redémarrées de réutiliser
le préfixe sauvegardé au lieu de traiter à nouveau tout le prompt.

## Modes de réflexion

DeepSeek V4 Flash a des modes distincts sans réflexion, avec réflexion, et Think Max.
Le serveur utilise le mode réflexion par défaut. `reasoning_effort=max` demande Think
Max, mais il n'est appliqué que lorsque la taille du contexte est assez grande pour la recommandation de
la fiche modèle ; les contextes plus petits retombent sur la réflexion normale. `reasoning_effort=xhigh`
d'OpenAI correspond encore à la réflexion normale, pas à Think Max.

Pour des réponses directes, utilisez `thinking: {"type":"disabled"}`, `think:false`, ou un
alias de modèle sans réflexion tel que `deepseek-chat`.

## Cache KV sur disque

Les API chat/completion sont sans état : les clients agents renvoient généralement toute la
conversation à chaque requête. `ds4-server` essaie d'abord la vérification bon marché du préfixe exact de tokens,
puis retombe sur la comparaison des octets du prompt rendu avec les octets du
point de contrôle décodé. Le point de contrôle en mémoire en direct couvre la session actuelle ; le
cache KV sur disque fait survivre les préfixes utiles aux changements de session et aux
redémarrages du serveur.

Pour des raisons de RAM, il n'y a actuellement qu'un seul cache KV en direct en mémoire. Lorsqu'une nouvelle
session non liée le remplace, l'ancien point de contrôle ne peut être repris sans
retraitement que s'il a été écrit dans le cache KV sur disque. Autrement dit, le cache
mémoire gère la session active ; le cache disque est le mécanisme de reprise pour des
sessions différentes.

Activez-le avec :

```sh
./ds4-server --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

La clé de cache est le SHA1 du préfixe d'octets rendu, et les fichiers sont nommés
`<sha1>.kv`. La charge utile DS4 stocke quand même les IDs de tokens exacts et l'état du graphe
pour ce préfixe. Cela importe pour les chats continués : le modèle peut avoir généré
un token dont le texte décodé est plus tard renvoyé par un client sous forme de deux tokens de prompt
canoniques. Un succès sur le préfixe d'octets rendu peut quand même réutiliser le point de contrôle et
ne tokeniser que le nouveau suffixe.
Le fichier est intentionnellement écrit avec des E/S `read`/`write` ordinaires, pas
`mmap`, afin que la restauration des entrées de cache n'ajoute pas plus de mappages VM à un processus
qui mappe déjà le modèle.

Les appels d'outils gardent aussi une carte de relecture DSML exacte bornée, indexée par des IDs d'outils
impossibles à deviner, afin que l'historique JSON du client puisse être rendu vers le texte exact échantillonné. La
carte en RAM garde jusqu'à 100000 IDs par défaut ; réglez-la avec `--tool-memory-max-ids`.
Utilisez `--disable-exact-dsml-tool-replay` pour désactiver cela et retomber sur le rendu
canonique JSON-vers-DSML.

Sur le disque, un fichier de cache est :

```text
KVC fixed header, 48 bytes
u32 rendered_text_bytes
rendered_text_bytes of UTF-8-ish token text
DS4 session payload, payload_bytes from the KVC header
optional tool-id map section
```

L'en-tête fixe est en petit-boutiste (little-endian) :

```text
0   u8[3]  magic = "KVC"
3   u8     version = 1
4   u8     routed expert quant bits, currently 2 or 4
5   u8     save reason: 0 unknown, 1 cold, 2 continued, 3 evict, 4 shutdown
6   u8     extension flags, bit 0 = appended tool-id map
7   u8     reserved
8   u32    cached token count
12  u32    hit count
16  u32    context size the snapshot was written for
20  u8[4]  reserved
24  u64    creation Unix time
32  u64    last-used Unix time
40  u64    DS4 session payload byte count
```

Le texte rendu est le texte décodé par le tokeniseur pour le préfixe de tokens mis en cache.
C'est à la fois le préfixe inspectable par un humain et l'identité de recherche : son SHA1 est
le nom de fichier, et un fichier n'est réutilisable que lorsque ces octets sont un préfixe du
prompt rendu entrant. Après le chargement, les tokens exacts du point de contrôle depuis la charge utile
DS4 restent l'autorité, et seul le suffixe de texte entrant après les octets mis en
cache est tokenisé.

La carte optionnelle des IDs d'outils n'est présente que lorsque le bit d'extension 0 de l'en-tête est activé.
Les sections ajoutées utilisent un ordre de bits fixe, afin que les futurs bits d'extension puissent ajouter des champs
sans ambiguïté. La carte stocke les IDs d'appels d'outils API impossibles à deviner vers le
bloc DSML exact que le modèle a échantillonné. Seuls les mappages dont le bloc DSML est présent
dans le texte rendu mis en cache sont stockés. Cela permet aux serveurs redémarrés de rendre
l'historique client ultérieur octet par octet comme la sortie originale du modèle, même si le
client réordonne les arguments JSON.

La section actuelle de la carte des IDs d'outils est :

```text
0   u8[3]  magic = "KTM"
3   u8     version = 1
4   u32    entry count

For each entry:
0   u32    tool id byte length
4   u32    sampled DSML byte length
8   bytes  tool id
... bytes  exact sampled DSML block
```

La section est une mémoire de relecture auxiliaire, pas un état du modèle. Un succès de cache restaure
d'abord la charge utile de session, puis charge la carte si elle est présente. Avant de rendre une
requête, le serveur peut aussi scanner les fichiers de cache pour les IDs d'outils présents dans
l'historique client et ne charger que ces mappages, afin qu'une relecture DSML exacte puisse survivre aux
redémarrages du serveur même lorsque l'instantané KV correspondant n'est pas celui finalement
utilisé pour le succès sur le préfixe rendu.

La charge utile de session DS4 commence par treize champs `u32` petit-boutistes :

```text
0   magic = "DSV4"
1   payload version = 2
2   saved context size
3   prefill chunk size
4   raw KV ring capacity
5   raw sliding-window length
6   compressed KV capacity
7   checkpoint token count
8   layer count
9   raw/head KV dimension
10  indexer head dimension
11  vocabulary size
12  live raw rows serialized below
```

Puis elle stocke :

- `u32[token_count]` les IDs de tokens du point de contrôle.
- `float32[vocab_size]` les logits pour le token suivant après ce point de contrôle.
- `u32[layer_count]` les comptes de lignes d'attention compressées.
- `u32[layer_count]` les comptes de lignes d'indexeur ratio-4.
- Pour chaque couche : les lignes KV vivantes de la fenêtre glissante brute, écrites en ordre de
  position logique plutôt qu'en ordre physique de l'anneau.
- Pour les couches compressées : les lignes KV compressées vivantes et les tenseurs de frontière du
  compresseur.
- Pour les couches compressées ratio-4 : les lignes compressées vivantes de l'indexeur et les tenseurs de
  frontière de l'indexeur.

Les logits sont des valeurs brutes IEEE-754 `float32` issues du tampon `ds4_session` de l'hôte.
Elles sont sauvegardées immédiatement après les tokens du point de contrôle afin qu'un
instantané chargé puisse échantillonner ou continuer à partir de la distribution exacte du token suivant sans
exécuter une étape de décodage supplémentaire. Les logits/l'état du brouillon MTP ne sont pas persistés ; après
le chargement d'un point de contrôle disque, l'état du brouillon est invalidé et reconstruit par la génération
normale.

Les sessions du coordinateur distribué utilisent la même charge utile `DSV4`. Les tenseurs de couches
détenus par les workers sont récupérés lors de la sauvegarde et fusionnés dans le flux de tenseurs ordonné par
couches normal ; lors du chargement, le coordinateur scinde ce flux dans le parcours actuel
et repousse les tenseurs de couches pertinents vers les workers. Le fichier sauvegardé
ne conserve pas la topologie distribuée.

La charge utile de tenseurs est un état KV/session spécifique à DS4, pas un vidage générique de graphe
d'inférence. Elle est censée être portable uniquement entre des builds `ds4.c` compatibles
pour cette disposition de modèle.

Le cache stocke les points de contrôle à quatre moments :

- `cold` : après qu'un long premier prompt atteint un préfixe stable, avant la génération.
- `continued` : lorsque le prefill ou la génération atteint la prochaine frontière absolue alignée.
- `evict` : avant qu'une requête non liée remplace la session en mémoire en direct.
- `shutdown` : lorsque le serveur se ferme proprement.

Les sauvegardes cold rognent intentionnellement un petit suffixe de tokens et s'alignent vers le bas sur une frontière
de bloc de prefill. Cela évite les échecs courants de retokenisation aux frontières BPE lorsqu'une
requête future ajoute du texte au même prompt. Les valeurs par défaut sont conservatrices :
stocker des préfixes d'au moins 512 tokens, sauvegarder en cold des prompts jusqu'à 30000 tokens,
rogner 32 tokens de queue, et s'aligner sur des blocs de 2048 tokens. Les paramètres importants sont :

Les sauvegardes continued utilisent le même alignement et ne sont écrites que lorsque le graphe en direct
atteint naturellement une frontière absolue. Avec les valeurs par défaut, cela signifie à peu près
tous les 10k tokens, indépendamment de l'endroit où le premier point de contrôle cold a atterri, de sorte que les longues
générations laissent derrière elles des points de reprise sans persister les fragiles derniers
tokens.

- `--kv-cache-min-tokens`
- `--kv-cache-cold-max-tokens`
- `--kv-cache-continued-interval-tokens`
- `--kv-cache-boundary-trim-tokens`
- `--kv-cache-boundary-align-tokens`
- `--tool-memory-max-ids`
- `--disable-exact-dsml-tool-replay`

Par défaut, les points de contrôle peuvent être réutilisés entre les variantes d'experts routés 2 bits et 4 bits
si le préfixe rendu correspond. Utilisez `--kv-cache-reject-different-quant`
lorsque vous voulez une réutilisation stricte à quantification identique uniquement.

Le répertoire de cache est jetable. Si le comportement semble suspect, arrêtez le
serveur et supprimez-le. Vous pouvez examiner ce qui est mis en cache avec hexdump car
les fichiers du cache KV incluent le prompt verbatim mis en cache.

## Backends

Le backend graphe par défaut est Metal sur macOS et CUDA dans les builds CUDA :

```sh
./ds4 -p "Hello" --metal
./ds4 -p "Hello" --cuda
```

Sur Linux, un simple `make` affiche les cibles de build disponibles au lieu de sélectionner une
cible CUDA implicitement. Utilisez `make cuda-spark` pour DGX Spark / GB10. Il omet un
`nvcc -arch` explicite car c'est actuellement le chemin le plus rapide sur GB10. Utilisez
`make cuda-generic` pour un build CUDA local normal, ou définissez `CUDA_ARCH` explicitement
lors d'une compilation croisée ou lorsque vous avez besoin d'une cible connue :

```sh
make cuda CUDA_ARCH=sm_120
make cuda CUDA_ARCH=native
```

Les builds CUDA acceptent `--gpu-vram N[,N,...]` et `--gpu-devices N[,N,...]` dans la
CLI, le serveur, l'agent et le benchmark. Les valeurs de VRAM sont des budgets par périphérique en Gio ;
`--gpu-vram auto` utilise la mémoire libre rapportée par CUDA. La liste de périphériques contrôle
l'ordre de placement et doit avoir le même nombre d'entrées qu'une liste de budgets explicite.
Le placement réserve la mémoire de graphe et de KV pour le contexte demandé
et refuse de démarrer si les couches du modèle débordaient vers le CPU.

Sans `--cuda-tensor-parallel`, CUDA utilise un placement de couches normal sur les
périphériques listés. C'est aussi la disposition multi-GPU prise en charge pour GLM 5.2 :

```sh
./ds4 -m gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf \
  --gpu-vram auto --gpu-devices 0,2,4,6,1,3,5,7 \
  --ctx 32768 -p "Hello"
```

Pour la disposition tenseurs/experts-parallèles de DeepSeek Flash, voir
« Parallélisme de tenseurs entre GPU CUDA » ci-dessus.

Il existe aussi un chemin de référence/débogage CPU :

```sh
./ds4 -p "Hello" --cpu
make cpu
./ds4
./ds4 -p "Hello"
```

Ne traitez pas le chemin CPU comme la cible de production. La CLI et `ds4-server`
prennent en charge le backend CPU pour un usage de référence/débogage et partagent le même format de
session KV et d'instantané que Metal et CUDA, mais l'inférence normale devrait utiliser Metal ou
CUDA.

## Pilotage (steering)

Ce projet prend en charge le pilotage avec des directions d'activation à vecteur unique ; voir le
répertoire `dir-steering` pour plus d'informations. Cela suit l'idée centrale de l'article
[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717).
Vous pouvez l'utiliser pour rendre le modèle plus ou moins verbeux, moins susceptible de
répondre aux questions de programmation s'il s'agit d'un chatbot pour votre site web de location de voitures,
et ainsi de suite, bien plus rapidement que le fine-tuning.
C'est aussi utile pour les chercheurs en cybersécurité qui veulent réduire la propension d'un modèle
à fournir des conseils de sécurité offensive ou à double usage.

## Vecteurs de test

`tests/test-vectors` contient des vecteurs de continuation à contexte court et long
capturés depuis l'API officielle de DeepSeek V4 Flash. Les requêtes utilisent
`deepseek-v4-flash`, le décodage glouton, la réflexion désactivée, et la tranche
`top_logprobs` maximale exposée par l'API. Les vecteurs locaux sont générés avec
`./ds4 --dump-logprobs` et comparés par octets de tokens, de sorte que les régressions de
tokeniseur/gabarit ou d'attention apparaissent avant de devenir de longs échecs de génération.
Le runner C fixe un bloc de prefill de 2048 tokens pour cette comparaison stricte de vecteurs d'API.

Les tests locaux principaux sont pilotés par le runner C, avec un petit auto-test de l'extracteur
`ds4-eval` exécuté en premier :

```sh
make test                  # ./ds4-eval --self-test-extractors && ./ds4_test --all
./ds4_test --logprob-vectors
./ds4_test --server
```

Les tests de batching sont adossés au modèle et doivent s'exécuter sur le backend GPU correspondant :

```sh
# Metal, with DS4_TEST_SESSION_COUNT set to 2, 4, 8, and 16.
DS4_TEST_MODEL=/path/to/model.gguf DS4_TEST_SESSION_COUNT=4 \
  make test-metal-session-batch

# CUDA multi-GPU Flash.
DS4_TEST_MODEL=/path/to/model.gguf make test-cuda-session-batch
DS4_TEST_MODEL=/path/to/model.gguf make test-cuda-mixed-batch
```

Pour GLM, exécutez le même test de session Metal avec un GGUF GLM et exécutez
`tests/glm_long_context_smoke.sh /path/to/model.gguf`. Les noteurs de qualité officiels de 100 cas,
les tests TCP/RDMA à deux Mac, la matrice CUDA et les vérifications manuelles de l'agent sont des
barrières de version plutôt que des tests locaux rapides ; suivez
[QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md).

## Notes de débogage

Lorsqu'une génération semble erronée, trois petits outils suffisent généralement pour obtenir une
première réponse :

```sh
./ds4 --dump-tokens -p "..."
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4 --dump-logits /tmp/logits.json --metal --nothink --prompt-file prompt.txt
./ds4-server --trace /tmp/ds4-trace.txt ...
```

- `--dump-tokens` tokenise la chaîne `-p` ou `--prompt-file` exactement telle
  qu'écrite, reconnaît les caractères spéciaux du protocole DS4, puis quitte avant que l'inférence
  ne commence. Par exemple, le marqueur de fermeture d'outil DSML commence par deux tokens : `</`
  et `｜DSML｜`.
- `--dump-logprobs` stocke une continuation gloutonne avec les meilleures alternatives
  locales à chaque étape, ce qui aide à séparer les choix d'échantillonnage des problèmes de
  logit/modèle.
- `ds4-server --trace` écrit les prompts rendus, les décisions de cache, le texte généré,
  et les événements de l'analyseur d'outils pour toute une session d'agent.

## Logo

Le logo DwarfStar a été conçu à la main par Salvatore Sanfilippo, rendu plus
graphique avec l'IA, et retravaillé manuellement par Ben Gnomino, dont la touche humaine l'a rendu
formidable.
