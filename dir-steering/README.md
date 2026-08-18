# Pilotage directionnel

Le pilotage directionnel (directional steering) est une modification d'activation à l'exécution pour DS4. Un fichier de pilotage est une
matrice `f32` plate avec une direction normalisée de 4096 de large par couche. Pendant
l'inférence, ds4 peut appliquer la modification après les sorties d'attention, les sorties FFN, ou les deux :

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

Une échelle positive supprime la direction représentée. Une échelle négative l'amplifie.
Sans fichier de pilotage ou avec des échelles nulles, ds4 suit le chemin d'inférence normal.

## Options d'exécution

```text
--dir-steering-file FILE   load a 43 x 4096 f32 direction file
--dir-steering-ffn F       apply steering after FFN outputs; default is 1 when a file is provided
--dir-steering-attn F      apply steering after attention outputs; default is 0
```

La sortie FFN est généralement la meilleure première cible car elle est suffisamment tardive dans
chaque couche pour représenter des signaux de comportement, de style et de sujet. Le pilotage d'attention
est disponible pour les expériences, mais il peut être plus fragile.

## Exemple de verbosité

L'exemple fourni construit une direction de style à partir de 100 paires de prompts. Chaque paire
demande la même information de deux manières différentes :

- `examples/succinct.txt` : prompts cibles concis.
- `examples/verbose.txt` : prompts de contraste détaillés.

Comme la direction extraite est `succinct - verbose`, les échelles FFN négatives
rendent les réponses plus courtes, tandis que les échelles FFN positives tendent à rendre les réponses plus longues et
plus explicatives.

Construisez le vecteur :

```sh
python3 dir-steering/tools/build_direction.py \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --good-file dir-steering/examples/succinct.txt \
  --bad-file dir-steering/examples/verbose.txt \
  --out dir-steering/out/verbosity.json \
  --component ffn_out \
  --ctx 512
```

Cela écrit :

```text
dir-steering/out/verbosity.json
dir-steering/out/verbosity.f32
```

Essayez une exécution concise :

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 160 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  -p "Explain why databases use indexes."
```

Essayez une exécution verbeuse :

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 220 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn 2 \
  -p "Explain why databases use indexes."
```

Le même vecteur peut être utilisé dans l'une ou l'autre direction. Le signe est la partie importante :

- une échelle négative amplifie la direction cible concise ;
- une échelle positive supprime cette direction et donne généralement au modèle plus de latitude
  pour développer.

## Évaluation des échelles

Utilisez l'assistant de balayage (sweep) pour tester plusieurs intensités sur un ensemble de prompts fixe :

```sh
python3 dir-steering/tools/run_sweep.py \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --direction dir-steering/out/verbosity.f32 \
  --prompts dir-steering/examples/eval_prompts.txt \
  --scales "-1,-0.5,0,0.5,1,2" \
  --tokens 180 \
  --nothink
```

Commencez avec des échelles FFN entre `-1` et `2`. Si le modèle devient répétitif,
ignore le prompt, ou commence à perdre du contenu factuel, l'échelle est trop forte.
Pour cet exemple, `-1` est un bon premier réglage concis et `2` est un bon premier
réglage verbeux. Les fortes échelles négatives comme `-2` ou `-3` peuvent sur-amplifier
la direction concise et s'effondrer en répétition sur certains prompts.

## Effet observé

Avec le vecteur de 100 paires construit à partir des commandes ci-dessus, des vérifications gloutonnes (greedy) locales
ont montré le comportement attendu :

- Prompt : `Explain why databases use indexes.`
- `--dir-steering-ffn -1` : 67 mots, un paragraphe compact.
- `--dir-steering-ffn 0` : 136 mots, explication structurée.
- `--dir-steering-ffn 1` : 140 mots, explication structurée avec plus de détails.

Sur un prompt auquel le modèle non piloté répondait déjà brièvement, le pilotage positif
a rendu l'expansion plus visible :

- Prompt : `What does DNS do?`
- `--dir-steering-ffn 0` : 44 mots.
- `--dir-steering-ffn 2` : 171 mots, avec des sections et un détail étape par étape.

## Construire d'autres directions

L'extracteur compare deux ensembles de prompts :

- `good-file` : prompts cibles pour la direction que vous voulez représenter.
- `bad-file` : prompts de contraste qui doivent être séparés de la cible.

Il capture les activations DS4 depuis le même graphe GPU local utilisé pour l'inférence,
moyenne la cible moins le contraste, normalise un vecteur par couche, et écrit à la fois
le JSON de métadonnées et le fichier `.f32` d'exécution.

Suppression de concept :

1. Placez les prompts riches en concept dans `good-file`.
2. Placez les prompts neutres dans `bad-file`.
3. Exécutez avec une échelle FFN positive.

Amplification de concept :

1. Placez les prompts du concept souhaité dans `good-file`.
2. Placez les prompts neutres dans `bad-file`.
3. Exécutez avec une échelle FFN négative.

Contrôle de style :

1. Placez les prompts pour le style cible dans `good-file`.
2. Placez les prompts de style contrastant dans `bad-file`.
3. Utilisez une échelle négative pour amplifier le style cible, une échelle positive pour le réduire.

La méthode n'est pas un fine-tune. C'est une modification à l'exécution de faible rang (low-rank), donc elle fonctionne mieux
pour des directions grossières de comportement, de sujet ou de style qui sont présentes de manière cohérente dans
les captures d'activation.
