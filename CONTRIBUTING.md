# Contribuer

Les modifications de DwarfStar4 doivent être testées par rapport au mode de défaillance
qu'elles peuvent réalistement affecter. Le projet dispose de deux pistes de régression : correction et vitesse. Merci
d'inclure les commandes que vous avez exécutées, la machine/le backend, le quant du modèle, ainsi que toute
défaillance notable dans la PR ou les notes de commit.

N'envoyez pas de PR affectant un ou plusieurs backends d'inférence sans vérifier si le
code résultant est toujours correct et rapide. La seule régression de vitesse acceptable
est celle qui survient lorsqu'un bug de correction important est corrigé et que cela nécessite une certaine pénalité de vitesse.

## Tests de régression de correction

Compilez d'abord le backend par défaut :

```sh
make clean
make
```

Le lanceur de tests C est `ds4_test`. L'exécuter sans argument équivaut à
`--all` :

```sh
make test
```

Vérifications ciblées utiles :

```sh
./ds4_test --server
./ds4_test --logprob-vectors
./ds4_test --long-context
./ds4_test --tool-call-quality
./ds4_test --metal-kernels
```

Ce qu'ils couvrent :

- `--server` : analyse des requêtes, rendu du chat, streaming, analyse des appels d'outils,
  contrôles de réflexion, gestion du cache disque KV, et autre logique côté serveur.
  C'est la meilleure vérification rapide pour les modifications de l'API et du rendu des prompts.
- `--logprob-vectors` : compare les octets de tokens locaux et les tranches de top-logprob aux
  vecteurs de continuation officiels de DeepSeek V4 Flash. Cela détecte les régressions du tokenizer,
  du template, de l'attention et des logits.
- `--long-context` : exécute une régression de rappel de faits sur une histoire à long contexte depuis
  `tests/long_context_story_prompt.txt`. Le modèle doit retrouver des associations personne-nombre
  explicitées dans un long prompt en prose et renvoyer des lignes `Name=number`
  que le test analyse.
- `--tool-call-quality` : exerce le comportement réel du modèle pour l'émission d'appels d'outils
  DSML dans les chemins rapide et exact.
- `--metal-kernels` : vérifications numériques isolées des kernels Metal.

Le lanceur utilise par défaut `ds4flash.gguf`. Remplacez les chemins au besoin :

```sh
DS4_TEST_MODEL=/path/to/model.gguf ./ds4_test --logprob-vectors
DS4_TEST_VECTOR_FILE=/path/to/official.vec ./ds4_test --logprob-vectors
DS4_TEST_LONG_PROMPT=/path/to/prompt.txt ./ds4_test --long-context
```

Pour les modifications spécifiques à CUDA, testez sur une machine CUDA :

```sh
make
make cuda-regression
```

Pour la portabilité CPU, vérifiez au moins que la cible CPU se compile toujours :

```sh
make cpu
```

Le backend CPU est un chemin de référence/débogage, pas la cible de performance
de production. Rappelez-vous qu'exécuter le chemin CPU sur Metal peut faire planter le système
à cause d'un bug du kernel dans macOS.

## Vérifications de qualité pour les modifications de quantification

Pour le travail sur GGUF ou la quantification, utilisez le scoreur de continuation officielle dans
`gguf-tools/quality-testing`. Le test compare la probabilité qu'un GGUF local
attribue aux continuations officielles de DeepSeek V4 Flash, token par token.

Compilez le scoreur :

```sh
make -C gguf-tools quality-score
```

Puis notez les anciens et nouveaux GGUF par rapport au même manifeste et comparez :

```sh
gguf-tools/quality-testing/score_official OLD.gguf \
  gguf-tools/quality-testing/data/manifest.tsv /tmp/old.tsv 4096

gguf-tools/quality-testing/score_official NEW.gguf \
  gguf-tools/quality-testing/data/manifest.tsv /tmp/new.tsv 4096

python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```

Un `avg_nll` plus bas est meilleur. Voir
`gguf-tools/quality-testing/README.md` pour collecter ou rafraîchir les continuations
officielles.

## Tests de régression de vitesse

Utilisez `ds4-bench` pour les régressions de débit. Il rapporte la vitesse instantanée de prefill et
de génération aux frontières de contexte, pas une seule moyenne sur l'ensemble de l'exécution. Le prefill est
incrémental : chaque ligne mesure uniquement le suffixe nouvellement traité depuis la
frontière précédente.

Balayage linéaire par défaut :

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv /tmp/ds4-speed.csv
```

Utilisez la même machine, le même backend, le même fichier de modèle, le même balayage de contexte, le même état d'alimentation/thermique,
et la même charge en arrière-plan lorsque vous comparez deux commits. Pour le travail sur les backends, exécutez au moins
un CSV avant/après et comparez à la fois `prefill_tps` et `gen_tps`. La génération est
gloutonne et ignore l'EOS afin que chaque frontière obtienne le même nombre de tokens générés.

Pour générer un graphique à partir d'un CSV :

```sh
python3 speed-bench/plot_speed.py /tmp/ds4-speed.csv --title "Machine t/s"
```

## Signaler des bugs de sessions

Pour déboguer une génération défaillante, conservez la trace :

```sh
./ds4-server --trace /tmp/ds4-trace.txt ...
```
