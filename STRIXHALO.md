# DS4 sur Strix Halo

Voici la configuration minimale pour l'inférence DS4 ROCm sur une
machine Strix Halo dotée de 128 Go de RAM et d'un Radeon 8060S (`gfx1151`).

## 1. Installer ROCm

Sous Ubuntu 26.04 LTS, installez le compilateur/runtime ROCm et les bibliothèques utilisés par le backend Strix Halo :

```sh
sudo apt-get update
sudo apt-get install -y \
  hipcc rocminfo rocm-smi \
  libamdhip64-dev \
  libhipblas-dev libhipblaslt-dev \
  librocblas-dev \
  librocwmma-dev \
  libhipcub-dev
```

Le backend utilise rocWMMA. Sur cette installation Ubuntu 26.04, `librocwmma-dev`
installe les en-têtes rocWMMA de premier niveau mais omet `rocwmma/internal/`.
Aucun paquet Ubuntu ne fournit actuellement ces en-têtes internes. Installez une
arborescence d'en-têtes rocWMMA complète et correspondante :

```sh
git clone --depth 1 --branch rocm-7.1.0 https://github.com/ROCm/rocWMMA.git /tmp/rocWMMA-rocm-7.1.0
sudo mkdir -p /usr/local/include
sudo cp -a /tmp/rocWMMA-rocm-7.1.0/library/include/rocwmma /usr/local/include/
```

Si ROCm est installé sous `/usr` alors que l'outillage s'attend à `/opt/rocm`, ajoutez ces
liens de compatibilité :

```sh
sudo mkdir -p /opt/rocm/bin
sudo ln -sf /usr/bin/hipcc /opt/rocm/bin/hipcc
sudo ln -sfn /usr/lib/x86_64-linux-gnu /opt/rocm/lib
sudo ln -sfn /usr/include /opt/rocm/include
```

## 2. Activer l'accès ROCm

L'utilisateur qui exécute DS4 doit pouvoir ouvrir `/dev/kfd` et le nœud de rendu DRM :

```sh
sudo usermod -aG render,video "$USER"
```

Déconnectez-vous puis reconnectez-vous, ou redémarrez. Vérifiez :

```sh
rocminfo | grep -A80 'Name:                    gfx1151'
```

Si DS4 indique `no ROCm-capable device is detected`, vérifiez que `rocminfo` peut ouvrir
`/dev/kfd` et que `groups` inclut `render`.

## 3. Augmenter la mémoire visible par le GPU

Un système Strix Halo de 128 Go peut initialement n'exposer qu'environ 62 Go de mémoire
visible par le GPU. DS4 a besoin de l'ouverture GTT plus large pour le modèle de 80,76 Gio
plus les tampons d'exécution.

Utilisez ces paramètres noyau :

```text
amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856
```

Sous Ubuntu avec GRUB :

```sh
sudo cp /etc/default/grub /etc/default/grub.bak
sudoedit /etc/default/grub
```

Définissez :

```text
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856"
```

Puis :

```sh
sudo update-grub
sudo reboot
```

Après le redémarrage, vérifiez :

```sh
cat /proc/cmdline
sudo dmesg | grep -Ei 'GTT|gttsize|TTM|VRAM'
rocminfo | grep -A80 'Name:                    gfx1151'
```

Signes attendus :

```text
amdgpu:  126976M of GTT memory ready
rocminfo gfx1151 pool: 130023424 KB
```

## 4. Compiler DS4

Utilisez la cible Strix Halo habituelle. Elle produit les noms de binaires standard :

```sh
make strix-halo -j"$(nproc)"
```

`make rocm` est un alias de `make strix-halo`.

## 5. Utiliser le bon GGUF

Utilisez le GGUF imatrix IQ2XXS/Q2K/Q8 standard :

```text
DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

Évitez pour l'instant les GGUF mixtes IQ2/IQ4 ou IQ2/Q4 sur cette machine. Ils exercent
une pression mémoire bien plus forte sur le chemin ROCm et peuvent déclencher un OOM système
plutôt qu'un échec DS4 propre.

## 6. Exécuter DS4

Lancez-le normalement :

```sh
./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

La build ROCm utilise automatiquement le backend Strix Halo.
