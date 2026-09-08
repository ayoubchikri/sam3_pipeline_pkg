# Pipeline de segmentation d’organoïdes SAM3-AMG

Initialisation AMG sur la dernière frame, puis propagation vidéo inverse.

## Contenu

| Fichier | Rôle |
| --- | --- |
| [`SAM2_segmentation/run_segmentation.py`](SAM2_segmentation/run_segmentation.py) | Pipeline principal : AMG SAM3 et propagation vidéo |
| [`SAM2_segmentation/focus_filter.py`](SAM2_segmentation/focus_filter.py) | Filtre de netteté optionnel |
| [`SAM2_segmentation/benchmarks/`](SAM2_segmentation/benchmarks/) | Dépendances internes : détection du puits et AMG |
| [`SAM2_segmentation/run.sh`](SAM2_segmentation/run.sh) | Lancement SLURM à adapter |
| [`requirements.txt`](requirements.txt) | Dépendances Python |
| `model_weights/hf/` | Cache des poids SAM3 à ajouter |

Ce paquet contient uniquement le pipeline et ses dépendances internes :
ni le benchmark comparatif complet, ni `study_pixel.py`, ni données, ni poids.
Les images TIFF originales sont indispensables pour segmenter et calculer
la netteté. Les masques seuls ne permettent pas de relancer ces opérations.

## À adapter avant utilisation

- `DATASET_ROOT` : chemin absolu vers vos images, à exporter avant le lancement.
- `HF_HOME` : chemin absolu vers votre cache des poids Hugging Face.
- `run.sh` : chemins, environnement conda et paramètres SLURM (voir ci-dessous).
- Sorties : vérifier l'espace disponible et les droits d'écriture dans le
  dossier du paquet. Compter environ 470 Mo de masques par puits sur le
  dataset de référence, auxquels s'ajoutent les grilles TIFF.
- Filtre de netteté : fournir les deux chemins explicitement (voir plus bas).

## Installation

```bash
# Depuis la racine de sam3_pipeline_pkg :
conda create -n sam3 python=3.12 && conda activate sam3
# Installer d'abord PyTorch adapté au GPU et à CUDA de votre serveur.
# PyTorch n'est pas installé par requirements.txt. Puis :
pip install -r requirements.txt
```

## Poids du modèle (non inclus)

Modèle `facebook/sam3` à accès restreint sur Hugging Face. Le téléchargement nécessite un compte ayant accepté la licence :

```bash
export HF_HOME="$PWD/model_weights/hf"
huggingface-cli login
huggingface-cli download facebook/sam3
```

Vous pouvez aussi copier un cache Hugging Face existant dans `model_weights/hf/`.

## Configurer les chemins

```bash
# Depuis la racine de sam3_pipeline_pkg, remplacer le chemin du dataset :
export HF_HOME="$PWD/model_weights/hf"
export DATASET_ROOT="/CHEMIN/VERS/processed_files_ALL_frames"
# Arborescence attendue : patient/puits/cavité.tif (films TIFF multipages).
# Sans DATASET_ROOT, le script utilise un chemin fictif et s'arrête.
```

## Lancer la segmentation

Un processus = un worker. Pour quatre workers, utiliser quatre GPU via votre propre `sbatch`.

```bash
cd SAM2_segmentation
python -u run_segmentation.py --queue --worker-id 0
#   ... --worker-id 1 / 2 / 3 sur d'autres GPU.
# --worker-id sert à identifier les logs ; il ne sélectionne pas le GPU.
# Affecter un GPU distinct à chaque worker via SLURM ou CUDA_VISIBLE_DEVICES.
# Par défaut, tous les patients sont traités. Ajouter --only CGR ou --only PGR
# pour restreindre le traitement.
```

## Adapter `run.sh` pour SLURM

Le fichier fourni conserve la configuration et les chemins du DCE d'origine.
Modifier `SAM2_segmentation/run.sh` :

- `#SBATCH --partition` : choisir une partition GPU de votre cluster.
- `#SBATCH --exclude` : supprimer ou adapter la liste de nœuds du DCE.
- Ajouter la réservation d'un GPU par tâche selon votre cluster
  (par exemple `#SBATCH --gres=gpu:1` si cette syntaxe y est utilisée).
- Adapter `--time`, `--cpus-per-task` et `--array=0-3` aux ressources autorisées.
- Remplacer `--output` et `--error` par vos chemins de logs ; créer leur dossier
  avant sbatch. Utiliser `%A_%a` pour conserver les logs de chaque soumission.
- Remplacer l'activation conda actuelle par une initialisation explicite :

  ```bash
  source "/CHEMIN/VERS/CONDA/etc/profile.d/conda.sh"
  conda activate sam3
  ```

  Adapter le chemin conda et le nom de l'environnement installé.
- Remplacer `export HF_HOME` par le chemin absolu de votre cache.
- Ajouter `export DATASET_ROOT="/CHEMIN/VERS/processed_files_ALL_frames"`.
- Remplacer le `cd` par le chemin absolu vers votre dossier SAM2_segmentation.
- Ajouter `--only PGR` à la commande Python si vous souhaitez traiter uniquement
  les PGR : le nom du job sam3PGR ne filtre pas les patients.

Les commentaires « fp16 + focus gate » de `run.sh` sont obsolètes : ce script
n'active pas explicitement fp16 et ne lance pas le filtre de netteté.

## Sorties

```text
<racine de sam3_pipeline_pkg>/<nom_dataset>_masks_SAM3/
```

L'arborescence des patients est conservée ; chaque dossier puits devient :

- `<puits>.npy` : dictionnaire `{cavité: masques booléens [n_frames, H, W]}`.
- `<puits>.tif` : vidéo-grille de contrôle (masque rouge par cavité).

Les masques sont à la résolution d'origine, les cellules de la grille à 140 px.
Ce ne sont pas des images d'organoïdes en niveaux de gris sur fond noir.
Conserver les TIFF sources pour pouvoir produire ces images ultérieurement.

Attention : la sortie est située dans le paquet, PAS à côté du dataset.
Pour changer sa destination, modifier l'affectation `root_out` dans `main()`
de `run_segmentation.py`. Les options `--in` et `--out` mentionnées dans l'en-tête
du script ne sont pas implémentées ; l'entrée se configure via `DATASET_ROOT`.

## Reprise et contrôle des résultats

Relancer les workers permet de reprendre les puits sans paire .npy + .tif.
Cette reprise vérifie uniquement la présence des fichiers, pas leur intégrité.

- Une cavité en erreur est enregistrée comme `None`. Si les deux fichiers du
  puits existent, ce puits sera sauté, même avec des cavités en échec.
- Une interruption pendant l'écriture peut laisser un fichier incomplet.
  Contrôler les logs et les sorties des puits interrompus avant de reprendre.
- Un verrou devient récupérable après 3 heures, sans vérification du worker.
  Si un puits peut prendre plus longtemps, adapter `STALE_SECONDS` dans
  `run_segmentation.py` pour éviter deux traitements simultanés du même puits.
  Ne retirer manuellement un verrou qu'après avoir vérifié l'arrêt du worker.
- `--overwrite` n'a pas d'effet avec `--queue`. Pour reprendre un puits en échec,
  arrêter les workers concernés, déplacer ses sorties hors du dossier de
  résultats, puis relancer la file d'attente.

## Filtre de netteté (optionnel, CPU)

```bash
# Depuis SAM2_segmentation, adapter les chemins :
python focus_filter.py \
  --dataset-root "$DATASET_ROOT" \
  --masks-root "/CHEMIN/VERS/sam3_pipeline_pkg/processed_files_ALL_frames_masks_SAM3" \
  --thresh 280 --k 15
```

`focus_filter.py` ne lit pas la variable `DATASET_ROOT` de l'environnement et ses
chemins par défaut pointent vers le DCE : fournir impérativement les DEUX
options ci-dessus. Adapter le nom du dossier de masques à celui du dataset.
Le seuil 280 et K=15 viennent du dataset de référence ; les adapter si besoin.
Le filtre produit des fichiers de classification et un CSV, sans supprimer
les masques. Il exige les images sources pour calculer la netteté ; des images
absentes ne doivent pas être interprétées comme des films nets.
Après changement des images, des masques ou de `--tau`, utiliser `--recompute`
pour recalculer les mesures mises en cache. Changer seulement `--thresh` ou `--k`
ne nécessite pas de recalcul.
