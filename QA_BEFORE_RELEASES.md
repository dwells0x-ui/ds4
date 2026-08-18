# QA avant les publications

Ceci est le point de contrôle de publication pour DwarfStar. Exécutez-le avant
de taguer ou de pousser une version de publication. L'objectif n'est pas de
prouver de façon exhaustive chaque chemin de code ; il s'agit d'exercer les
chemins qui ont historiquement régressé : inférence par graphe Metal, CUDA,
ROCm, streaming SSD, exécution distribuée, cache KV sur disque, API serveur et
la machine à états TUI/outils de l'agent.

N'exécutez pas plusieurs processus de gros modèles en même temps. Consignez le
commit, le matériel, le fichier GGUF, la taille de contexte et tous les
indicateurs non par défaut pour chaque exécution manuelle.

Hôtes de test de publication préférés :

- CUDA / DGX Spark : `toor@192.168.60.184`.
- Tests Metal / distribués sur Mac : `mac-m5max-it` et `mac-m5max-us`.
- ROCm : le système Strix Halo sur antirez@strixhalo (Framework Desktop).

`192.168.60.250` est réservé à un usage sur permission uniquement. Ne vous y
connectez jamais pour le QA, n'arrêtez ni ne démarrez son serveur, ne compilez
pas dessus, et n'y exécutez ni tests ni benchmarks sans demander à Salvatore et
recevoir une permission explicite pour cette passe de QA précise. Une permission
antérieure ne se reporte pas sur un travail ultérieur.

Les hôtes Mac ont des entrées DNS et sont joints via un VPN internet. Ils sont
connectés entre eux par WiFi ainsi que par une liaison point à point
Thunderbolt 5. La route TB5 est le réseau d'inférence distribuée préféré
lorsqu'elle est disponible, mais elle peut être fragile et ne fonctionne parfois
que lorsque `ds4` est exécuté au premier plan. Préférez ces machines pour les
tests de publication, en particulier pour l'inférence distribuée. Un test de
repli local sur cette machine est acceptable en cas de besoin ; il s'agit d'un
M3 Max avec 128 Go de RAM.
Le système Strix Halo est également joignable via le VPN et possède une adresse
WiFi locale sur le même réseau local que les systèmes M5 Max. Les hôtes CUDA se
trouvent sur un réseau local distant différent et sont accessibles via un autre
VPN actif sur ce système.

## 1. Intégrité du dépôt et de la compilation

- Partez d'un arbre propre à l'exception des notes de publication
  intentionnelles :
  `git status --short`.
- Compilez la cible locale normale :
  `make clean && make`.
- Compilez les binaires CPU uniquement comme simple vérification de compilation :
  `make clean && make cpu`.
- Traitez les avertissements du compilateur comme des échecs de compilation.
  Sauvegardez la sortie complète de chaque compilation de publication et de test
  et exigez l'absence de lignes `warning:` ou de lignes NVCC `warning #`.
  Corrigez la source lorsque c'est possible ; n'utilisez une suppression étroite
  spécifique à une cible que lorsqu'un test compile délibérément une unité de
  traduction partielle.
- Répétez le contrôle de compilation sans avertissement sur le matériel de
  publication :
  `make clean && make` sur Metal,
  `make clean && make cuda-spark` sur DGX Spark,
  `make clean && make cuda-generic CUDA_HOME=/usr` sur l'hôte CUDA multi-GPU
  uniquement après avoir reçu la permission pour `192.168.60.250`,
  `make clean && make strix-halo` sur Strix Halo.
- Exécutez les vérifications d'espaces blancs avant de commiter :
  `git diff --check`.
- Confirmez que `./ds4 --help`, `./ds4-server --help` et `./ds4-agent --help`
  s'affichent proprement, avec des couleurs de section lisibles et sans retour à
  la ligne défectueux.

## 2. Tests de régression de base

- Exécutez la suite par défaut :
  `make test`.
- Exécutez explicitement `tests/test_gpu_args_cli.sh` après avoir modifié
  l'analyse des options des exécutables ou le placement multi-GPU. Les valeurs
  invalides et les incohérences entre le nombre de périphériques et de budgets
  doivent atteindre l'analyseur GPU partagé dans les quatre binaires ; une
  réponse `unknown option` d'un binaire qui annonce l'indicateur est un
  bloqueur de publication. Sur CUDA, démarrez aussi `ds4-server` une fois avec
  `--gpu-vram auto` et la liste `--gpu-devices` prévue, puis conservez la ligne
  de disposition résolue.
- Exécutez explicitement les vérifications de vecteurs après tout changement de
  tokenizer, template, KV, kernel, quantification ou rendu de prompt :
  `DS4_TEST_MODEL=/path/to/0731.gguf
  DS4_TEST_VECTOR_FILE=tests/test-vectors/flash-0731/official.vec
  ./ds4_test --logprob-vectors`
  et
  `DS4_TEST_MODEL=/path/to/0731.gguf
  DS4_TEST_LOCAL_GOLDEN_FILE=tests/test-vectors/flash-0731/local-golden.vec
  ./ds4_test --local-golden-vectors`.
- Exécutez les tests serveur lorsque HTTP, SSE, le rendu de prompt, la politique
  de cache ou la relecture d'appels d'outils ont changé :
  `./ds4_test --server`.
- Exécutez `./ds4-eval --self-test-extractors`.

### Passe critique de régression sur les entrées et le serveur

Exécutez ces vérifications après avoir modifié les analyseurs, la génération
serveur, le chargement de modèle, les instantanés distribués, les caches, DSpark
ou les règles de compilation CUDA. Conservez les numéros des éléments dans le
rapport de QA afin que les omissions soient visibles.

1. Envoyez des requêtes OpenAI, Responses et Anthropic malformées avec des
   champs de chaîne ou de tableau détenus répétés sous ASan. Chaque requête doit
   échouer proprement et une requête valide suivante doit toujours fonctionner.
   Exécutez aussi `./ds4_test --server`.
2. Rejouez au moins 4 096 paires assistant/résultat-d'outil à travers les
   validateurs Responses et Anthropic. La validation doit rester en temps
   linéaire et préserver les mêmes historiques acceptés et rejetés qu'une
   relecture courte.
3. Fournissez à un worker distribué un en-tête d'instantané dont les longueurs
   déclarées dépassent les limites configurées et de protocole. Il doit rejeter
   l'en-tête avant toute allocation importante ou lecture de charge utile, sans
   croissance matérielle de la RSS.
4. Faites passer des fixtures safetensors et GGUF malformées à travers le
   chargeur et `gguf-tools/deepseek4-quantize` sous ASan et UBSan. Les fichiers
   tronqués, les dimensions impossibles et les tailles de tenseur en dépassement
   doivent être rejetés.
5. Avec un drafter DSpark correspondant au checkpoint, comparez la sortie à
   température zéro à une exécution sans drafter pour 400 et 800 tokens générés.
   Consignez la première différence de sortie, l'acceptation, les commits
   directs, les replis en relecture et la vitesse de décodage. L'identité
   octet par octet n'est pas requise : DSpark commite l'état du vérificateur par
   lots, dont l'ordre des opérations en virgule flottante diffère du décodage
   token par token. Les erreurs de vérificateur, un texte invalide ou une
   régression matérielle de la qualité de continuation restent des bloqueurs de
   publication. Utilisez `--dspark-strict` pour le contrôle cible uniquement.
6. Exercez un raisonnement non terminé et fermé deux fois dans des requêtes
   OpenAI, Responses et Anthropic en streaming et hors streaming, avec et sans
   outils. Le raisonnement ne doit jamais fuir dans le contenu de la réponse.
7. Sur du vrai matériel Blackwell, compilez les cibles CUDA pour `sm_120` ou
   `sm_120a` et pour DGX Spark `sm_121`. Confirmez que les indicateurs
   d'architecture émis conservent le suffixe de fonctionnalité spécifique à
   l'architecture et exécutez `make cuda-regression`.
8. Compilez avec CUDA 12.8 ou plus récent et exigez que les unités de traduction
   CUDA se compilent sans avertissement, y compris les utilisateurs de
   `FLT_MAX`.
9. Forcez une conversation au-delà du seuil KV en mémoire, restaurez deux fois le
   même checkpoint sur disque et confirmez que le fichier de checkpoint reste
   présent après les deux chargements réussis. Les checkpoints corrompus doivent
   toujours être rejetés.
10. Exécutez `make dspark-verify-depth` avec des fichiers cible et drafter 0731
    correspondants. La capture stricte doit ignorer les couches sans compresseur
    et comparer chaque couche de compresseur capturée.
11. Envoyez deux fois le même long prompt GLM 5.2 à une même session serveur. La
    seconde requête doit rapporter `cache_source: memory-rewind`, réutiliser
    jusqu'à un token avant la frontière du prompt et produire la même sortie
    gloutonne qu'une session fraîche.
12. Exécutez `./ds4_test --think-tool-recovery`, puis répétez à travers les trois
    API HTTP. Un bloc d'outil complet à l'intérieur d'un raisonnement non fermé
    doit être récupéré une fois, la prose précédente doit rester du raisonnement,
    et aucune continuation synthétique ne doit être générée.
13. Exécutez `./ds4_agent_test` sous ASan avec des chaînes de cache d'agent dont
    les longueurs déclarées dépassent le reste du fichier. Le chargement doit
    échouer sans allouer la taille déclarée, et un cache valide doit toujours se
    charger.
14. Exécutez les tests de l'analyseur serveur sous UBSan avec `NaN`, l'infini
    positif et l'infini négatif là où des champs JSON entiers sont attendus. La
    conversion doit être définie et bornée, sans rapport du sanitizer.

## 3. Points de contrôle officiels de qualité de continuation

Ces tests bloquent la publication après des changements de tokenizer, template,
cache KV, attention, routage MoE, quantification, logit ou graphe de modèle.
Ce sont des vérifications de continuation avec teacher forcing par rapport à la
sortie d'un modèle hébergé et à des tranches de top-logprob d'API, donc ne les
remplacez pas par une seule réponse de chat échantillonnée.

- Compilez le scoreur :
  `make -C gguf-tools quality-score`.
- Associez chaque GGUF Flash à la fixture capturée depuis le même checkpoint. Le
  checkpoint de publication actuel est 0731 et utilise
  `tests/test-vectors/flash-0731/` ; l'ancien GGUF non daté utilise la fixture
  préservée `tests/test-vectors/flash-pre-0731/`. Ne rapportez jamais un échec
  inter-checkpoints comme une régression de qualité. Les nouveaux checkpoints
  nécessitent un nouveau répertoire `flash-CHECKPOINT/` avant le QA de
  publication ; ne remplacez pas une ancienne fixture. Les GGUF étiquetés par
  checkpoint tels que `-0731` doivent utiliser la fixture portant la même
  étiquette.
- Exécutez les vecteurs de smoke suivis DeepSeek V4 Flash 0731 :
  `DS4_TEST_MODEL=/path/to/0731.gguf
  DS4_TEST_VECTOR_FILE=tests/test-vectors/flash-0731/official.vec
  ./ds4_test --logprob-vectors`.
  Cela couvre les prompts courts et les cas d'attention à prompt long. Le runner
  utilise cette fixture par défaut, mais les journaux de publication devraient
  conserver le chemin explicite.
- Exécutez la fixture DeepSeek V4 Flash à 100 cas pour chaque GGUF Flash publié :
  `gguf-tools/quality-testing/score_official /path/to/deepseek-v4-flash.gguf gguf-tools/quality-testing/data/flash/manifest.tsv /tmp/flash.tsv 4096`.
  Ce manifeste est également pour le checkpoint 0731. Un checkpoint ultérieur
  nécessite une fixture à 100 cas nommée séparément et ne doit pas être noté par
  rapport à celle-ci.
- Traitez le GGUF Flash MXFP4 natif comme un artefact de publication distinct.
  Exécutez la même fixture à 100 cas sur Metal, CUDA résident et streaming SSD
  CUDA lorsque ces backends sont annoncés ; comparez chaque résultat avec la
  référence Metal. Pour le CUDA multi-GPU résident, passez les indicateurs de
  placement normaux au scoreur, par exemple `--gpu-vram auto --gpu-devices
  0,2,4,6,1,3,5,7 --cuda-tensor-parallel`.
- Exécutez la fixture GLM 5.2 OpenRouter à 100 cas pour chaque GGUF GLM publié :
  `gguf-tools/quality-testing/score_official models/GLM-5.2-UD-Q4_K_XL.gguf gguf-tools/quality-testing/data/glm52-openrouter-100/manifest.tsv /tmp/glm52-q4.tsv 4096`.
  Bande de référence Q4 XL actuelle : correspondance du premier token `95/100`,
  accord API top-1 d'environ `0.942`, et accord API sur l'ordre des paires
  d'environ `0.880`.
- Exécutez la même fixture GLM pour les fichiers de publication GLM à précision
  réduite. La référence Q2 routée est de moindre qualité mais devrait rester
  proche d'une correspondance du premier token `92/100`, d'un accord API top-1
  d'environ `0.890` et d'un accord API sur l'ordre des paires d'environ `0.800`
  à moins que la quantification n'ait changé délibérément.
- Exécutez la fixture DeepSeek V4 PRO à 100 cas pour chaque GGUF PRO publié :
  `gguf-tools/quality-testing/score_official /path/to/deepseek-v4-pro.gguf gguf-tools/quality-testing/data/pro/manifest.tsv /tmp/pro.tsv 4096`.
- Pour le streaming SSD, exécutez le même scoreur de continuation officielle une
  fois avec résidence complète et une fois avec `--ssd-streaming` pour le modèle
  de publication. Le résumé et l'accord API devraient rester dans la même bande
  de qualité.
- Comparez tout candidat à la publication précédente ou à la dernière sortie
  connue comme bonne :
  `python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv`.
  Traitez une forte baisse de correspondance du premier token, une régression
  claire de NLL, ou une régression matérielle de l'API top-1/ordre des paires
  comme un bloqueur, à moins que les notes de publication ne signalent un
  compromis de qualité intentionnel.
- Conservez les lignes brutes `summary` et `api_summary` dans les notes de
  publication ou le journal de QA. N'utilisez pas de manifestes obsolètes de
  `misc/` comme preuve de publication.

## 4. Chemin Flash Metal

Utilisez le GGUF Flash normal que les utilisateurs de 128 Go exécutent.

- CLI ponctuelle :
  `./ds4 -m ds4flash.gguf --ctx 32768 --nothink -p "Explain C pointers in one paragraph."`
- Prompts de réflexion et de réflexion maximale :
  exécutez un court prompt de code avec la réflexion par défaut et un avec la
  réflexion maximale.
- Rappel en contexte long :
  exécutez le test de rappel de nom/nombre long ou d'archive utilisé pour
  détecter la dérive de l'attention et du routage MoE.
- Vérification de cohérence des logprob :
  `./ds4 --nothink --temp 0 --dump-logprobs /tmp/ds4-logprobs.json --logprobs-top-k 20 -p "..."`
  et inspectez que la continuation est saine.
- Vérification de cohérence de la vitesse :
  exécutez `ds4-bench` avec `speed-bench/promessi_sposi.txt` et comparez le
  prefill, la vitesse de génération et les octets KV avec les derniers bons
  chiffres connus pour la même machine.
- Pour les changements MXFP4 natifs, exécutez `make mxfp4-dot-test
  test-mxfp4-metal`, puis un court prompt glouton et la fixture de continuation
  de la section 3 avec le GGUF MXFP4. Le test synthétique de MoE fusionné et le
  point de contrôle de qualité du modèle complet doivent tous deux passer.

### Runtime DSpark / DeepSpec

DSpark est optionnel, mais il mute le vérificateur, la capture cachée de la
cible, le chargement du modèle de support et les chemins du planificateur.
Exécutez ces tests à chaque fois que le support DSpark, la vérification
spéculative, la politique de confiance/planificateur, la capture cachée de la
cible, les petits kernels de vérificateur MoE routés ou le code partagé du
modèle de support `--mtp` changent :

Utilisez le GGUF de support DSpark 0731 uniquement avec une cible Flash 0731. Un
modèle de support d'un autre checkpoint peut avoir des statistiques
d'acceptation plausibles tout en produisant une continuation gloutonne
différente.

Les exécutions DSpark normales commitent directement l'état accepté du
vérificateur cible. Le vérificateur par lots et le décodage token par token
utilisent le même graphe avec un ordre d'opérations en virgule flottante
différent, donc `output_match=0` par rapport à la référence est diagnostique
plutôt qu'un échec. `--dspark-strict` reste le mode cible uniquement identique à
l'octet.

- Fixture d'acceptation gloutonne par défaut :
  `DS4_DSPARK_MODEL=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-acceptance`.
- Garde-fou à 64 tokens :
  `DS4_DSPARK_FIXTURE_TOKENS=64 DS4_DSPARK_MODEL=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-acceptance`.
- Commit partiel direct à bloc fixe :
  `DS4_DSPARK_FIXTURE_CONFIDENCE=0 DS4_DSPARK_FIXTURE_TOKENS=8 DS4_DSPARK_FIXTURE_REQUIRE_PARTIAL=1 DS4_DSPARK_MODEL=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-acceptance`.
- Smoke de l'invariant du vérificateur DSpark :
  `DS4_TEST_MODEL=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-verify-depth`.
- Si les structures partagées du modèle de support ou du vérificateur ont changé,
  exécutez aussi le MTP hérité :
  `make mtp-verify-depth` avec `DS4_TEST_MTP` défini sur un GGUF de support MTP à
  un étage, ou confirmez que la cible ne saute que parce que le fichier optionnel
  est manquant.
- Consignez `c_add`, `accepted_draft`, `direct_full`, `direct_partial`,
  `replay_fallbacks`, `errors=0`, `verify_layer`, `net_saved` et `output_match`
  pour les exécutions à 32 tokens et à 64 tokens. Au moins un commit direct doit
  se produire. Une exécution plus rapide avec une qualité de proposition
  inférieure est une régression à moins qu'il ne s'agisse d'un changement de
  planificateur intentionnel.
- Si les kernels MoE du vérificateur ont changé, exécutez un profil diagnostique
  `c_add` avec `DS4_DSPARK_VERIFY_SELECTED_PROFILE=1` ou le profileur d'étage MoE
  Metal et consignez l'empreinte des experts sélectionnés ou le temps d'étage
  dans le journal DSpark.

### Micro-batching de sessions et TP Metal

Exécutez ces points de contrôle à chaque fois que la planification de sessions,
le décodage par lots, le prefill/décodage mixte, la projection QKV, les experts
partagés ou routés, le parallélisme de tenseurs ou la sélection de repli de
backend changent.

- Sur une seule machine Metal, exécutez l'oracle de logit exact à vocabulaire
  complet avec 2, 4, 8 et 16 sessions :
  `DS4_TEST_MODEL=/path/to/ds4flash.gguf DS4_TEST_SESSION_COUNT=N make test-metal-session-batch`.
  Les exécutions Q8 résidentes compatibles doivent rapporter `native_shared=1
  native_qkv=1` à chaque nombre testé. L'exécution à 16 sessions couvre des
  nombres de lignes au-dessus de l'ancienne limite artificielle de huit lignes.
- Répétez l'oracle à quatre sessions avec
  `DS4_METAL_SESSION_BATCH_SHARED=0` et avec
  `DS4_METAL_SESSION_BATCH_QKV=0`. La première exécution doit utiliser le repli
  complet ; la seconde peut ne mettre en lot que l'expert partagé. Les deux
  doivent rester exactes au bit près.
- L'oracle doit couvrir un ordre de lignes inversé, au moins six étapes de
  décodage et un appel de prefill/décodage mixte. Tout nombre non nul de logits
  divergents est un bloqueur ; un accord uniquement sur l'argmax est
  insuffisant.
- Faites un benchmark de 1, 2, 4, 8 et 16 sessions résidentes simultanées sur le
  même hôte et modèle. Consignez la latence par étape de modèle et le débit de
  décodage agrégé en tokens/seconde, pas seulement la vitesse d'achèvement des
  requêtes. Le chemin Metal actuel met en lot le QKV et une partie de l'expert
  partagé, mais exécute toujours l'attention, les experts routés, le down
  partagé et la tête de sortie par session. Traitez une mise à l'échelle agrégée
  plate comme un travail d'implémentation inachevé, et non comme une preuve que
  Metal ne peut pas bénéficier du batching.
- Sur `mac-m5max-it` et `mac-m5max-us`, exécutez le même oracle en mode TP
  physique sur des transports `tcp` et `rdma` explicites. Définissez
  `DS4_TEST_TP_MODE=leader` sur le leader et `DS4_TEST_TP_MODE=worker
  DS4_TEST_TP_LEADER_HOST=HOST` sur le worker, avec un `DS4_TEST_TP_PORT` unique.
  Exécutez au moins 2 et 4 sessions et conservez les deux journaux. Le TP utilise
  actuellement le repli ordonné par session, donc les indicateurs natifs de
  grille de lignes sur une seule machine doivent rester désactivés dans les
  journaux TP.
- Pour la liaison TB5 MacBook actuelle, US est `10.99.0.2` sur `en1`/`rdma_en1`
  et IT est `10.99.0.1` sur `en6`/`rdma_en6` ; les deux utilisent l'index GID 1.
  Avant de tester, exigez que `rdma_ctl status` rapporte `enabled` et que
  `ibv_devinfo -v` affiche `PORT_ACTIVE` plus le GID `::ffff:10.99.0.x`
  correspondant. Forcez le périphérique et le GID avec `--rdma-device NAME
  --rdma-gid-index 1` si la sélection automatique est ambiguë. Un ping IP TB
  fonctionnel à lui seul n'est pas une preuve de RDMA.
- Tuez le worker TP pendant un lot avec `DS4_TEST_TP_DISCONNECT=1` sur le leader.
  L'opération doit échouer proprement, invalider chaque session affectée et
  rendre le contrôle sans blocage.
- Vérifiez explicitement les combinaisons non prises en charge : les modèles de
  support GLM, DSpark/MTP, le streaming SSD, les modes qualité/référence, le
  steering, le chevauchement d'experts Q4 résidents et les modes routeur CPU
  doivent utiliser le repli exact établi ou rejeter la combinaison avant
  l'évaluation. Ils ne doivent pas activer partiellement le batching natif.

## 5. Chemin PRO Metal

Le support PRO est expérimental, mais les versions de publication ne doivent pas
le casser silencieusement.

- Si une machine capable de PRO est disponible, exécutez un court prompt PRO q2
  et vérifiez le template correct, le comportement de réflexion et les alias de
  point de terminaison.
- Pour les versions distribuées PRO Q4, testez uniquement sur les machines à
  haute mémoire prévues.
- Si PRO ne peut pas être exécuté localement, compilez au moins tous les binaires
  et passez en revue les changements touchant la forme du modèle, la recherche de
  tenseurs, le mappage des experts routés, la logique de template et la
  compatibilité de la charge utile KV.

## 6. GLM 5.2

GLM a un template, une forme de modèle, un bloc MTP, une disposition d'attention,
une largeur de porte en parallélisme de tenseurs et une politique de streaming
différents. Le succès de Flash ou PRO ne se substitue pas à cette matrice.

- Sur une machine Metal de 512 Go, exécutez de courts prompts gloutons avec les
  GGUF de publication Q4 XL et Q2 à précision réduite. Couvrez les templates avec
  et sans réflexion, et vérifiez que le serveur rapporte la famille de modèles
  GLM plutôt qu'un alias DeepSeek en interne.
- Exécutez explicitement les vecteurs de smoke OpenRouter :
  `DS4_TEST_MODEL=/path/to/glm.gguf
  DS4_TEST_VECTOR_FILE=tests/test-vectors/glm-openrouter/official.vec
  ./ds4_test --logprob-vectors`.
  Conservez le rapport comme diagnostic. Les vecteurs hébergés incluent des
  queues top-20 à très faible probabilité dont l'appartenance n'est pas stable
  après la quantification des experts routés GLM, donc une assertion individuelle
  `official top token missing locally` n'est pas à elle seule un bloqueur de
  publication. Les incohérences de token sélectionné doivent rester cohérentes
  avec la bande de premier token à 100 cas du modèle, et le scoreur de la
  section 3 est le point de contrôle de publication pour la qualité GLM agrégée.
- Exécutez les fixtures officielles à 100 cas Q4 XL et Q2 de la section 3 et
  conservez les lignes `summary` et `api_summary`. Comparez indépendamment aux
  bandes de référence documentées Q4 et Q2.
- Exécutez `tests/glm_long_context_smoke.sh` avec le contexte annoncé pour la
  publication sur l'hôte Metal de 512 Go. La continuation générée doit commencer
  par `>` et ne contenir aucun des marqueurs connus de corruption de tokens.
- Exercez le MTP GLM intégré avec `--glm-mtp-timing` sur un prompt déterministe.
  Comparez le texte glouton à une exécution sans MTP, exigez des cycles
  spéculatifs propres et consignez l'acceptation et le timing. Exécutez aussi une
  fois avec le MTP désactivé pour prouver que le décodage ordinaire reste le
  comportement par défaut.
- Exécutez l'oracle de session Metal avec 2 et 4 sessions GLM. Il doit rapporter
  `family=glm native_shared=0 native_qkv=0` et rester exact, y compris le
  prefill/décodage mixte ; les kernels de grille de lignes exclusifs à DeepSeek
  ne doivent pas s'activer.
- Exécutez des prompts GLM Q2 résidents et en streaming SSD avec la même entrée
  gloutonne. Comparez le premier token et la cohérence des top-logprob, et
  consignez le préfixe de couche complète sélectionné et le budget dynamique de
  cache d'experts.
- Exécutez un TP GLM physique à deux machines sur TCP et RDMA avec des prompts
  courts et longs. Consignez la vitesse de prefill/décodage, le transport, la
  résidence des rangs et l'arrêt propre. Répétez une exécution avec
  `--tensor-parallel-token-prefill` comme diagnostic à arithmétique exacte.
  Utilisez un GGUF dont le type d'expert routé possède des kernels TP GLM
  conscients de la propriété. Testez aussi un fichier GLM routé Q4 comme point de
  contrôle négatif : tant que les kernels de propriété Q4 ne sont pas
  implémentés, les deux rangs doivent le rejeter clairement avant l'évaluation
  plutôt que de charger un découpage partiel ou de se bloquer.
- Avec une permission explicite pour la passe de QA actuelle, exécutez un prompt
  GLM Q2 résident, un prompt en contexte long, le MTP GLM intégré et des requêtes
  serveur concurrentes sur l'hôte CUDA à huit GPU. Utilisez le placement de
  couches ordinaire à huit GPU pour GLM ; ne passez pas l'option
  `--cuda-tensor-parallel` spécifique à Flash. Le prefill GLM multi-niveaux doit
  rapporter la progression à travers le chemin token-major de commutation de
  niveau, et le décodage, les mises à jour de cache et l'assemblage de la tête de
  sortie/logit doivent se terminer sans débordement CPU. Le placement automatique
  doit réserver le cache DSA/indexeur compact de chaque couche et l'espace de
  travail du graphe avant de charger les poids ; un échec tardif d'allocation de
  graphe est un bloqueur de publication. Confirmez que la disposition en contexte
  long reste dans le budget de chaque périphérique et utilise des niveaux
  supplémentaires lorsque le cache ne tient plus sur les précédents.
  Le harnais de contexte long peut sélectionner ce backend avec
  `DS4_GLM_BACKEND=cuda` et passer les indicateurs de placement via
  `DS4_GLM_EXTRA_ARGS="--gpu-vram auto --gpu-devices 0,2,4,6,1,3,5,7"`.
- Via `ds4-server`, exercez les requêtes OpenAI chat, Responses et Anthropic
  contre GLM, y compris la réflexion et le SSE. Les alias de point de terminaison
  de compatibilité DeepSeek peuvent se résoudre vers le modèle chargé, mais les
  prompts rendus et le texte généré doivent utiliser le template GLM.

## 7. Streaming SSD

Le streaming SSD est un chemin de capacité, donc testez à la fois la justesse et
l'expérience utilisateur.

- Streaming Flash q2/q2-q4 :
  `./ds4 -m ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 32GB -p "..."`
- Testez la régression du streaming SSD Flash à quantification mixte. Utilisez le
  GGUF mixte q2/q4 avec des couches d'experts routés Q4 renforcées et un prompt
  assez long pour exercer le chemin de prefill à adresse sélectionnée ; il ne
  doit pas échouer avec « model range is not covered by mapped model views » :
  `./ds4 -m gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf --ssd-streaming --ssd-streaming-cache-experts 16GB --ctx 4096 --tokens 1 --nothink --prompt-file /tmp/ds4_600tok_prompt.txt`.
- Mesure de streaming à froid :
  exécutez une fois avec `--ssd-streaming-cold` et vérifiez l'absence d'interblocage,
  d'expert manquant ou de ralentissement impossible.
- Confirmez que le démarrage rapporte le budget de cache et que la génération ne
  cale pas sur des manques d'experts répétés pour un petit prompt interactif.
- Si les mécanismes internes du cache de streaming ont changé, testez le même
  prompt deux fois et comparez la cohérence du premier token/logprob entre les
  exécutions.

## 8. CUDA / DGX Spark

Avant une publication, demandez à l'utilisateur l'accès CUDA s'il n'est pas déjà
configuré. Utilisez l'hôte DGX Spark / GB10 `toor@192.168.60.184`. Ne prétendez
pas que CUDA est prêt pour la publication sans cette passe.

- Récupérez ou poussez le commit de publication exact vers la machine CUDA.
- Compilez :
  `make clean && make cuda-spark`.
- Exigez que la version DGX Spark et la version CUDA à huit GPU se terminent
  toutes deux sans avertissement du compilateur. La version à huit GPU n'est
  effectuée qu'après avoir reçu la permission explicite d'utiliser
  `192.168.60.250` pour cette passe de QA.
- Exécutez :
  `make cuda-regression`.
- Pour les changements MXFP4 natifs, exécutez
  `make test-mxfp4-cuda CUDA_ARCH=native` sur l'hôte CUDA multi-GPU uniquement
  après avoir reçu la permission explicite pour `192.168.60.250`, et
  `make test-mxfp4-cuda CUDA_ARCH=sm_121` sur DGX Spark. Le MMQ dense, le MMQ
  routé, le MMVQ routé, le gate/up fusionné et le down fusionné doivent passer.
  L'exécution Spark doit aussi passer le garde-fou de K-tile Blackwell. Ce test
  synthétique de parité ne remplace pas le scoring de continuation du modèle
  complet.
- Avec cette permission, exécutez le GGUF MXFP4 natif résident sur l'hôte
  multi-GPU, et exécutez-le avec `--ssd-streaming` sur DGX Spark. Utilisez le
  même prompt glouton et la même fixture de continuation sur les deux. Consignez
  la vitesse de prefill et de génération, exigez des logits finis et comparez la
  qualité au résultat MXFP4 Metal. Le MMQ Blackwell quantifie les activations en
  FP4 natif pour le travail par lots ; le MMVQ de décodage conserve des
  activations Q8, donc la qualité doit être vérifiée plutôt qu'inférée de la
  parité au niveau des kernels uniquement.
- Exécutez un court prompt CLI avec le GGUF Flash et consignez le débit de
  génération en t/s.
- Exécutez un prompt plus long qui exerce les experts routés au-delà de quelques
  milliers de tokens.
- Avec une permission explicite pour cette passe de QA, exécutez l'oracle de
  décodage à vocabulaire complet sur l'hôte CUDA à huit GPU :
  `DS4_TEST_MODEL=/path/to/flash.gguf make test-cuda-session-batch`.
  Conservez le timing par lot pour 2, 4 et 8 lignes et exigez
  `nonexact_logits=0`. Exécutez le fichier Q4 publié et le fichier Q2 à précision
  réduite : Q4 exerce les étages routés/partagés groupés, tandis que les formes
  MoE natives Q2 non prises en charge doivent conserver le repli exact ordonné.
- Avec l'attention TP CUDA activée, les exécutions Q4 compatibles doivent
  utiliser par défaut l'attention-core, le QKV, le KV-store et
  l'attention-post groupés et rester exactes à vocabulaire complet par rapport au
  décodage isolé. Sur l'hôte à huit L40S, l'étape de décodage à 16 lignes doit
  rester au-dessus de 110 tokens/s agrégés. Répétez une fois avec
  `DS4_CUDA_TP_ATTN=0` uniquement à titre de couverture de rollback ; ce n'est
  pas la configuration de production.
- Exécutez le prefill/décodage mixte natif à la frontière par défaut et à
  contexte compressé :
  `DS4_TEST_MODEL=/path/to/flash.gguf make test-cuda-mixed-batch` et
  `DS4_TEST_CONTEXT=4096 DS4_TEST_MIXED_INITIAL=2048 DS4_TEST_MIXED_ROUNDS=8
  DS4_TEST_MODEL=/path/to/flash.gguf make test-cuda-mixed-batch`.
  Chaque tour doit rapporter des logits exacts et `mode=native` ; un repli
  sérialisé est un échec pour la topologie TP/EP à huit GPU. Sous attention TP
  CUDA, l'étape mixte native doit utiliser les mêmes étages de décodage groupés
  exacts lorsque leurs vérifications de capacité passent ; consignez la justesse
  et l'accélération séparément. Forcez aussi un quantum de prefill de 800 lignes
  avec `DS4_TEST_ALLOW_FALLBACK=1` ; il doit rapporter le repli de sécurité
  sérialisé.
- Avec une permission explicite pour l'hôte à huit GPU, démarrez `ds4-server`
  avec 8 et 16 sessions par lots et émettez au moins autant de requêtes
  simultanées avec des longueurs de prompt mixtes. Vérifiez l'absence de
  confusion de session, d'interblocage ou de famine et consignez le débit de
  génération agrégé.
- Sur DGX Spark, vérifiez que la même API de lot publique et la même concurrence
  serveur utilisent le repli mono-GPU sans créer d'état TP/EP entre pairs
  uniquement. L'oracle natif à huit GPU n'est pas un test Spark valide car sa
  topologie y est intentionnellement indisponible.
- Si le code CUDA Q4, distribué, les hooks de streaming, le chargement de plages
  de tenseurs ou le cache de modèle ont changé, testez le GGUF et le mode de
  découpage spécifiques qui utilisent ce chemin.
- Vérifiez que toute correction d'avertissement exclusive à CUDA est aussi propre
  sur macOS et ne change pas le comportement de Metal.

## 9. ROCm / Strix Halo

Utilisez le Strix Halo Framework Desktop via le nom d'hôte VPN `strixhalo`
(`antirez@strixhalo`). Cet hôte valide le backend ROCm ; ne l'utilisez pas comme
substitut aux tests de publication CUDA ou Metal.

- Récupérez ou poussez le commit de publication exact vers la machine Strix Halo.
- Compilez :
  `make clean && make strix-halo`.
- Exigez que la version ROCm se termine sans avertissement du compilateur.
- Utilisez le GGUF imatrix Flash q2 pour les tests de smoke de publication :
  `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`.
- N'utilisez pas encore les GGUF Flash mixtes q2-q4 ou Q4 pour le QA de routine
  Strix Halo. Ils sont dangereux sur cette machine pour l'instant car le chemin
  ROCm peut provoquer un OOM système au lieu d'échouer proprement.
- Exécutez un court prompt CLI :
  `./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf --ctx 4096 --nothink -p "Reply with exactly: OK"`.
- Pour le décodage DeepSeek Flash, confirmez que le chemin par défaut utilise des
  activations Q8 préquantifiées. Répétez la même exécution gloutonne avec
  `DS4_ROCM_DSV4_PREQUANT_DECODE=0` uniquement comme contrôle diagnostique. Le
  chemin par défaut doit être matériellement plus rapide et doit toujours passer
  le point de contrôle de qualité de continuation. GLM et `--quality` doivent
  rester sur le chemin d'activation FP32 complet.
- Testez DSpark avec les fichiers cible et de support 0731 correspondants :
  `DS4_BIN=./ds4 DS4_DSPARK_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf DS4_DSPARK_FIXTURE_TOKENS=64 sh tests/dspark_acceptance_fixture.sh`.
  Exigez des propositions, des tokens de brouillon acceptés, au moins un commit
  d'état direct, zéro erreur de vérificateur et zéro repli en relecture.
  Consignez séparément la vitesse de génération ordinaire et DSpark. Lorsque la
  gestion directe de l'état du vérificateur change, comparez aussi avec une
  version de test uniquement de son prédécesseur immédiat en relecture ; la
  version directe doit être plus rapide. On ne s'attend pas actuellement à ce que
  DSpark batte le décodage ROCm ordinaire, donc ne le décrivez pas comme une
  accélération ROCm sans une nouvelle mesure.
- Exécutez un prompt plus long si les kernels ROCm, les hooks de backend, le
  chargement de tenseurs, le cache de modèle, le cache KV ou le code de prefill
  de graphe ont changé.
- Exécutez le modèle de publication GLM Q2 via le streaming SSD ROCm avec au
  moins quatre tokens générés :
  `./ds4 --rocm -m gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf --ssd-streaming --ctx 4096 --nothink --tokens 4 -p "Reply with exactly: OK"`.
  Le démarrage doit sélectionner un budget de cache qui passe le garde-fou
  mémoire sans surcharge, et le prefill indexé compact ainsi que le décodage
  doivent tous deux se terminer.
- Exécutez un prompt GLM plus long avec le contexte Strix annoncé pour la
  publication après des changements de l'attention GLM, des projections
  quantifiées typées, des caches d'experts en streaming ou de la budgétisation
  mémoire. Consignez le contexte, le découpage du cache et si la continuation
  reste exempte de marqueurs de corruption de tokens.
- Exécutez le même modèle GLM avec `--glm-mtp-timing --temp 0`. Au moins un cycle
  de vérification de brouillon doit se terminer sans message `glm mtp step
  failed`.
- Consignez les messages mémoire/cache de démarrage, la vitesse de prefill, la
  vitesse de génération et si le backend rapporte `ROCm backend initialized`.

## 10. Inférence distribuée

Le code distribué a régressé autour de la configuration des routes, des
instantanés KV, des identifiants de requête et du chargement de modèles
découpés. Testez-le à chaque fois que le code distribué, KV, de session ou de
chargement de modèle change.

- Préférez `mac-m5max-it` et `mac-m5max-us` pour les tests distribués Metal.
  Utilisez la liaison point à point TB5 lorsqu'elle fonctionne ; sinon, notez que
  l'exécution a utilisé le routage WiFi/VPN.
- Démarrez d'abord les workers, puis le coordinateur.
- Testez un petit prompt et un prompt plus long.
- Vérifiez que le coordinateur attend une route complète et se termine
  proprement.
- Vérifiez que `Ctrl+C` rend le contrôle après le vidage du token ou du fragment
  distribué en cours.
- Sauvegardez et restaurez un instantané KV distribué si ce code a changé.
- Si le CUDA distribué est pertinent, testez à travers les hôtes CUDA et
  consignez la vitesse de génération, pas seulement « ça marche ».

## 11. Cache KV sur disque

Les bugs de cache KV sur disque ont un fort impact pour les utilisateurs de
serveur.

- Démarrez le serveur avec :
  `./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192`.
- Exécutez la même requête deux fois et vérifiez que la seconde requête touche le
  cache.
- Remplissez le cache suffisamment pour déclencher l'éviction ; vérifiez que
  l'entrée nouvellement écrite n'est pas évincée et que les ancres utiles sont
  conservées.
- Testez le rejet des checkpoints incompatibles lorsque le modèle, la
  quantification, le contexte ou la disposition KV brute/compressée change.
- Testez les sessions d'agent nettoyées : `/strip <id>` puis `/switch <id>`
  doit reconstruire par prefill et rendre un historique sain.

## 12. API serveur

Le serveur doit conserver la compatibilité entre les clients OpenAI, Responses et
Anthropic.

- `GET /v1/models/deepseek-v4-flash` et `GET /v1/models/deepseek-v4-pro`
  doivent tous deux servir le GGUF chargé, quel qu'il soit.
- Testez la complétion de chat OpenAI, OpenAI Responses et les messages
  Anthropic.
- Testez le streaming SSE avec la réflexion activée et désactivée.
- Testez le keepalive pendant un long prefill et confirmez que les clients
  n'expirent pas.
- En mode par lots, fermez les clients pendant que leurs requêtes sont en file
  d'attente, en prefill et en décodage en streaming. Répétez pour OpenAI chat,
  Responses, Anthropic et complétions. Le travail abandonné doit s'arrêter à la
  prochaine frontière sûre du backend, et une requête valide après chaque
  annulation doit se terminer normalement.
- Uniquement après avoir reçu une permission explicite pour cette passe de QA,
  démarrez `ds4-server` sur la cible TP CUDA à huit L40S avec les options TP de
  publication et vérifiez que les 16 sessions à contexte de 100k s'allouent. Le
  démarrage doit rapporter un plafond de prefill de 2048 tokens ; un repli
  silencieux à 4096 est une régression OOM.
- Testez `--trace` et confirmez que les prompts rendus, les décisions de cache,
  le texte généré et les événements de l'analyseur d'outils sont utiles sans
  fuite d'état non lié.

## 13. ds4-agent

L'agent est le composant le plus riche en état. Testez-le manuellement, pas
seulement par compilation.

- Bannière de démarrage, barre d'état, aide, `/power`, `/save`, `/list`,
  `/switch`, `/history`, `/compact`, `/new`, `/del` et `/strip`.
- Ctrl+C pendant la génération, pendant le prefill, pendant une récupération web
  et pendant un long appel d'outil. Après `Stopped by user`, taper un nouveau
  prompt doit fonctionner.
- Mettez des messages en file d'attente pendant que le modèle est occupé. Les
  messages en file d'attente ne doivent pas sauter l'exécution des outils ; après
  les résultats d'outils, le texte utilisateur en file d'attente doit être
  fourni.
- Outils read/search/edit/write :
  créez un projet temporaire et demandez des modifications. Par défaut, vérifiez
  que les remplacements exacts old/new fonctionnent et que le prompt de l'outil
  n'annonce pas `[upto]`. Dans une exécution `--edit-upto` séparée, vérifiez que
  les modifications ancrées échouent de façon sûre sur les correspondances
  ambiguës et ne nécessitent pas de retaper des fichiers entiers.
- Boucle de modification de code réelle :
  supprimez `/tmp/mymandel`, demandez à ds4-agent d'y créer un petit programme C
  de Mandelbrot en ASCII, compilez-le et exécutez-le, puis lors d'un second tour
  utilisateur demandez une petite modification qui devrait naturellement utiliser
  l'outil d'édition, comme changer la rampe de caractères ASCII ou les dimensions
  de sortie. Vérifiez que l'agent modifie le fichier existant au lieu de
  réécrire tout le projet, et que le programme final compile et s'exécute
  toujours.
- Outils bash :
  testez une sortie courte, la troncature d'une grande sortie, une sortie à code
  de retour non nul, des tâches de longue durée, `bash_status` et `bash_stop`.
- Outils web :
  `google_search` et `visit_page` doivent demander une approbation Chrome visible
  avec timeout, ouvrir les pages sans voler le focus lorsque c'est possible,
  extraire du Markdown, fermer les onglets et gérer les murs de
  consentement/confidentialité comme des erreurs d'outil que le modèle peut voir.
- TUI :
  testez l'édition de prompt multiligne, la navigation dans l'historique,
  l'affichage des prompts en file d'attente, le remplissage de la barre d'état à
  la largeur du terminal, la coloration syntaxique dans les blocs Markdown/code et
  le scintillement des terminaux SSH/distants.

## 14. Script de téléchargement et fichiers de modèles

- Testez `download_model.sh` dans un répertoire temporaire afin que les poids
  locaux ne soient pas écrasés.
- Testez une cible Flash et une cible PRO suffisamment pour vérifier l'URL, la
  reprise, le comportement du CLI/curl Hugging Face, le nommage des fichiers et
  la politique de liens symboliques.
- Vérifiez que les anciennes cibles supprimées échouent clairement.
- Vérifiez que les noms de modèles du README correspondent au script et au dépôt
  Hugging Face.

## 15. Performance et puissance

- Exécutez `ds4-bench` sur la machine de publication et comparez aux références
  CSV suivies.
- Testez que `--power 100` n'est pas bridé.
- Testez que `--power 50` réduit visiblement le cycle de service dans la CLI, le
  serveur, l'agent, l'éval et le bench là où c'est praticable.
- Confirmez que la taille du buffer de contexte, les lignes KV brutes, les lignes
  KV compressées et le comportement de mmap correspondent aux attentes pour 32k,
  100k et toute taille de contexte annoncée pour la publication.

## 16. Régression de vitesse

La performance est un point de contrôle de publication. Un résultat correct qui
est étonnamment beaucoup plus lent nécessite quand même une explication avant la
publication.

Utilisez le même commit, le même checksum GGUF, le même prompt, la même frontière
de contexte, le même nombre de tokens générés, le même réglage de puissance et
les mêmes indicateurs de backend que l'exécution de référence. Laissez la machine
devenir inactive, écartez la première exécution de préchauffage, puis consignez
la médiane de trois exécutions. Ne comparez pas différents checkpoints ou
quantifications de modèle. Pour les tests par lots, consignez la vitesse de
décodage agrégée et par session.

- Un ralentissement de plus de 5 % nécessite une réexécution propre et une
  investigation.
- Un ralentissement reproductible de plus de 10 % en prefill, décodage ou
  décodage par lots agrégé est un bloqueur de publication à moins que le
  changement et le compromis ne soient documentés.
- Conservez le CSV `ds4-bench` complet. Une seule moyenne sur prompt court ne
  suffit pas à détecter une régression dépendante du contexte.
- Comparez le temps de démarrage et la mémoire de pointe ainsi que les tokens par
  seconde lorsque le chargement de modèle, les caches, le streaming ou les arènes
  temporaires ont changé.
- Exécutez les tests par lots spécifiques au backend des sections 4 et 8. Un
  décodage mono-session rapide ne se substitue pas au débit multi-session agrégé.

Voici les dernières bonnes observations connues disponibles lorsque ce point de
contrôle a été ajouté. Ce sont des points de référence pour du matériel et des
charges de travail correspondants, pas des affirmations de performance pour
différents modèles ou contextes.

| Système et backend | Modèle et charge de travail | Prefill | Décodage |
| --- | --- | ---: | ---: |
| MacBook Pro M3 Max 128 Go, Metal | Flash q2, prompt de 11 709 tokens | 250.11 t/s | 21.47 t/s |
| MacBook Pro M5 Max 128 Go, Metal | Flash q2, prompt de 11 707 tokens | 463.44 t/s | 25.90 t/s |
| Mac Studio M3 Ultra 512 Go, Metal | Flash q2, prompt de 11 709 tokens | 468.03 t/s | 27.39 t/s |
| Mac Studio M3 Ultra 512 Go, Metal | Flash q4, prompt de 12 018 tokens | 448.82 t/s | 26.62 t/s |
| Deux Mac M5 Max 128 Go, TP Metal sur TB5 RDMA | GLM 5.2 IQ2_XXS, contexte de 4 096 tokens | environ 94 t/s | 15.4 t/s |
| DGX Spark GB10, CUDA | Flash q2, prompt de 7 047 tokens | 343.81 t/s | 13.75 t/s |
| DGX Spark GB10, CUDA | Flash q2 DSpark, fixture C de 64 tokens | - | 24.48 t/s direct ; 13.93 t/s prédécesseur en relecture |
| Strix Halo gfx1151, ROCm | Flash IQ2 résident, court smoke de la section 9 | - | 17.27 t/s ; rollback FP32 9.70 t/s |
| Strix Halo gfx1151, ROCm | Flash IQ2 résident, contexte de 4 096 tokens | - | 14.82 t/s ; rollback FP32 8.76 t/s |
| Strix Halo gfx1151, ROCm | Flash IQ2 DSpark, fixture C de 64 tokens | - | 11.40 t/s direct ; 9.77 t/s prédécesseur en relecture ; 16.70 t/s ordinaire |
| 8x L40S, TP CUDA | Flash q4, benchmark de prefill de 2 048 tokens | 1,524.84 t/s | 46.93 t/s |
| 8x L40S, TP CUDA | Flash q4, oracle de décodage à 16 lignes | - | 126.0 t/s agrégés |

Les valeurs 8x L40S sont conservées de la dernière exécution enregistrée sur
`192.168.60.250`. Ce sont uniquement des références historiques : ne vous
connectez jamais à cet hôte et n'interrompez jamais son serveur de production
sans permission explicite pour la passe de QA actuelle. Si la permission est
accordée, le plancher dur existant reste de 110 t/s agrégés pour l'oracle de
décodage à 16 lignes.

## 17. Validation de publication

Ne validez pas tant que :

- Le Flash Metal macOS n'a pas réussi.
- Les points de contrôle GLM 5.2 Metal, qualité officielle, MTP, repli de
  batching et TP ou CUDA applicables n'ont pas réussi.
- Les points de contrôle officiels de qualité de continuation n'ont pas réussi
  pour chaque famille de modèles publiée.
- CUDA n'a pas été testé sur la machine CUDA ou les notes de publication ne
  déclarent pas explicitement que CUDA n'a pas été validé.
- ROCm n'a pas été testé sur Strix Halo ou les notes de publication ne déclarent
  pas explicitement que ROCm n'a pas été validé.
- Les versions Metal, CUDA, ROCm, CPU uniquement et de test ne se sont pas
  terminées sans avertissement du compilateur sur chaque cible de publication
  validée.
- Le cache KV sur disque n'a pas été exercé.
- Le streaming de l'API serveur n'a pas été exercé.
- L'interruption de l'agent et les boucles d'outils n'ont pas été exercées
  manuellement.
- Le point de contrôle de régression de vitesse n'a pas réussi sur chaque backend
  validé, avec toute référence ignorée ou tout ralentissement intentionnel
  documenté.
- Les points de contrôle d'exactitude des sessions Metal 2/4/8/16 et de repli
  forcé n'ont pas réussi.
- Le batching TP Metal physique et le décodage/batching mixte natif CUDA n'ont
  pas réussi lorsque ces backends font partie de la publication.
- Tout élément ignoré est consigné avec la raison.
