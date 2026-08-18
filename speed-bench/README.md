## Analyse comparative

Nous rassemblons ici les vitesses de préremplissage (prefill) et de génération obtenues avec différents matériels.

Lancez `ds4-bench` ainsi :

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Soumettez une PR incluant vos chiffres si votre matériel n'a pas encore été testé.
Nommez le fichier csv du benchmark quelque chose comme `m3_max.csv` ou similaire, afin
qu'il soit clair quel matériel a été utilisé pour le benchmark.

Pour générer un graphique SVG à partir d'un fichier CSV :

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

Le script n'utilise que la bibliothèque standard de Python. Par défaut, il écrit un fichier
à côté du CSV en utilisant le suffixe `_ts.svg`, comme `speed-bench/m3_max_ts.svg`.

### Comparaison A/B du schedule de décodage Metal

Compilez la comparaison de décodage Metal équilibrée, sur le même moteur, avec :

```
make metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench \
  -m ds4flash.gguf \
  --include-selection
```

Le harnais préremplit deux sessions et alterne à la fois l'ordre des variantes et
l'affectation variante-à-session. Il s'interrompt à moins que chaque ligne de logits
sur le vocabulaire complet ne soit identique bit à bit et, avec `--include-selection`, que les deux variantes ne sélectionnent le
même token non-EOS. Utilisez `--candidate-env NAME` pour mesurer un contrôle de rollback,
ou `--help` pour comparer des schedules de découpage explicites.

Pour comparer la fusion pack/transpose par défaut du compresseur ratio-4 pré-M5 avec le
chemin de décodage historique, y compris la sélection de token, utilisez :

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION \
  --include-selection \
  --tokens 1024
```

### Comparaison A/B de la variante de préremplissage Metal

Compilez la comparaison de préremplissage équilibrée. Pour comparer le cull tail-SIMDgroup
par défaut de la paire MXFP4 résidente pré-M5 au kernel de paire original, faites du
chemin de rollback le candidat :

```
make metal-prefill-variant-bench
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL
```

Pour isoler le cull tail-SIMDgroup routed-down par défaut de la paire par défaut
conservée, utilisez son rollback spécifique au down comme candidat :

```
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_TAIL_SIMDGROUP_CULL
```

Le harnais utilise un seul moteur Metal et des sessions neuves pour chaque exécution. Il chauffe
les deux variantes avec au moins 32 tokens, alterne l'ordre contrôle/candidat en
blocs ABBA et BAAB, empoisonne les buffers de logits de l'hôte avant la copie, et s'interrompt
à moins que chaque ligne finale de logits sur le vocabulaire complet ne soit identique bit à bit. Les valeurs par défaut sont un
préfixe de 8192 tokens, un contexte de 8193 tokens dimensionné automatiquement, et deux répétitions ;
utilisez `--help` pour les remplacer.
