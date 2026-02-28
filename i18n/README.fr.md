[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


> 🌐 **État du multilingue :** `i18n/` est présent et réservé aux fichiers README spécifiques à chaque langue. Les documents localisés liés sont planifiés/en cours.

<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="Bannière LazyingArt" />
</p>

# Lazeal OptiX

![Project Stage](https://img.shields.io/badge/stage-research%20prototype-orange)
![Primary Workflow](https://img.shields.io/badge/workflow-notebook--centric-blue)
![Python](https://img.shields.io/badge/python-3.7-blue)
![Conda](https://img.shields.io/badge/environment-conda-44A833)
![Jupyter](https://img.shields.io/badge/interface-jupyter-F37626)
![OpenCV](https://img.shields.io/badge/cv-opencv%204.x-5C3EE8)
![License](https://img.shields.io/badge/license-TBD-lightgrey)

<table width="100%">
  <tr>
    <td align="center" valign="middle" width="50%">
      <img src="./prototype_individual.jpg" alt="Prototype pour usage individuel" style="width: 90%" />
    </td>
    <td align="center" valign="middle" width="50%">
      <img src="./prototype_institute.png" alt="Prototype pour usage institutionnel" style="width: 90%" />
    </td>
  </tr>
</table>

*Prototype pour usage individuel (gauche) et institutionnel (droite)*

## Vue d’ensemble

Lazeal OptiX est un projet innovant de technologie de santé. Le cœur du projet est le développement d’un appareil offrant des diagnostics avancés aux utilisateurs depuis le confort de leur domicile. En utilisant des techniques avancées de microscopie et d’analyse biochimique, l’appareil vise à faciliter la détection précoce de divers problèmes de santé, contribuant ainsi à de meilleurs résultats en matière de soins.

Le projet Lazeal OptiX est né d’un engagement à réduire la souffrance et à rendre les diagnostics de santé plus accessibles à toutes et tous. En donnant aux individus les outils nécessaires pour prendre le contrôle de leur santé, nous cherchons à contribuer à une société en meilleure santé.

Le dépôt est actuellement orienté recherche/prototype et centré sur les notebooks. La plupart des détails d’implémentation et des expérimentations sont suivis dans les notebooks Jupyter sous `notebooks/`.

### En bref

| Domaine | État actuel |
|---|---|
| Maturité du projet | Prototype de recherche |
| Modèle d’exécution principal | Workflows de notebooks Jupyter |
| Principaux domaines d’expérimentation | Reconstruction, localisation de source lumineuse, appariement multi-images |
| Packaging/CI à la racine | Non déclaré actuellement |
| Documentation multilingue | Structure de répertoire `i18n/` existante |

## Fonctionnalités

1. **Microscopie avancée :** Exploitation de techniques de microscopie avancées pour une analyse détaillée.
2. **Analyse biochimique :** Une analyse biochimique approfondie permet de détecter divers indicateurs de santé.
3. **Facile d’utilisation :** Conçu pour un usage domestique, avec une interface simple et accessible.
4. **Compact et abordable :** Lazeal OptiX est compact et proposé à un coût accessible, apportant des diagnostics avancés aux utilisateurs du quotidien.
5. **Workflows de reconstruction lensless :** Pipelines d’imagerie computationnelle et de reconstruction basés sur des notebooks.
6. **Expériences de localisation de source lumineuse :** Notebooks d’optimisation pour l’estimation de la position de la source lumineuse.
7. **Utilitaires d’appariement multi-images :** Workflows notebook et C++ OpenCV pour l’appariement/l’alignement de caractéristiques.

## Structure du dépôt

```text
lazealoptix/
├── README.md
├── prototype_individual.jpg
├── prototype_institute.png
├── figs/
│   ├── banner.svg|png
│   ├── logo.svg|png
│   └── logo-w-text.svg|png
├── camera/
│   └── README.md
├── light_source/
│   └── README.md
├── reconstruction/
│   └── README.md
├── three_axis_cnc/
│   └── README.md
├── notebooks/
│   ├── light_source_location/
│   ├── multiple_match/
│   └── reconstruction/
└── i18n/
```

### Notes sur les modules

- `camera/` : scripts/ressources liés à l’utilisation de la caméra pour la capture d’échantillons en haute résolution.
- `light_source/` : scripts/ressources pour le contrôle et l’optimisation de la source lumineuse.
- `reconstruction/` : scripts/ressources pour la reconstruction computationnelle.
- `three_axis_cnc/` : scripts/ressources pour le positionnement/contrôle CNC à trois axes.
- `notebooks/` : espace de travail technique principal pour les expérimentations et méthodes.

## Notebooks

Le répertoire `notebooks` contient des notebooks Jupyter documentant différents aspects du projet Lazeal OptiX. Ces notebooks incluent du code, des visualisations et des explications détaillées des méthodologies du projet. Ils offrent un moyen interactif d’explorer et de comprendre le projet.

### `light_source_location`

Le répertoire `light_source_location` contient des notebooks liés à l’estimation des positions de sources lumineuses. Ces notebooks présentent les algorithmes et méthodes utilisés pour estimer avec précision la position de la source lumineuse, un aspect crucial du projet Lazeal OptiX.

### `multiple_match`

Le répertoire `multiple_match` contient des notebooks et scripts liés à l’appariement de plusieurs images ou motifs. Cette partie du projet met en jeu des algorithmes complexes pour apparier et aligner précisément les images, ce qui est nécessaire à la reconstruction d’images haute résolution à partir du système d’imagerie lensless.

### `reconstruction`

Le répertoire `reconstruction` contient des notebooks consacrés à la reconstruction d’images capturées par l’appareil Lazeal OptiX. Ces notebooks documentent les techniques computationnelles avancées utilisées pour reconstruire des images haute résolution à partir du système d’imagerie lensless.

## Prérequis

- OS : Linux/macOS recommandé pour les workflows actuels notebooks et OpenCV.
- Python : Le fichier d’environnement fourni cible **Python 3.7**.
- Conda : Requis pour reproduire l’environnement `lensless` documenté.
- Jupyter Notebook/Lab.
- Toolchain C++ optionnelle pour `multiple_match.cpp` :
  - `g++` avec prise en charge C++17.
  - OpenCV 4.x avec modules contrib (`opencv2/xfeatures2d.hpp` / SIFT).

## Installation

### 1) Cloner

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) Créer l’environnement notebook (recommandé)

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) Démarrer Jupyter

```bash
jupyter notebook
```

## Utilisation

Ce dépôt s’utilise principalement en ouvrant les notebooks et en exécutant les cellules dans l’ordre.

### Parcours reconstruction

- Ouvrez `notebooks/reconstruction/dataset_prep.ipynb` pour la préparation du jeu de données.
- Ouvrez `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` pour les expérimentations de reconstruction/entraînement.

### Parcours localisation de source lumineuse

- Ouvrez les notebooks sous `notebooks/light_source_location/`.

### Parcours multiple match

- Ouvrez les notebooks sous `notebooks/multiple_match/`.
- Utilitaire C++ optionnel : `notebooks/multiple_match/multiple_match.cpp`.

## Configuration

### Environnement Conda

La spécification d’environnement principale se trouve ici :

- `notebooks/reconstruction/lensless.yaml`

Signaux de dépendances notables de ce fichier :

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- Dépendances de workflow de vision par ordinateur liées à `opencv` dans les notebooks

### Données et chemins

- **Hypothèse :** les notebooks attendent des jeux de données/fichiers locaux qui ne sont pas déclarés de manière centralisée à la racine du dépôt.
- **Hypothèse :** l’utilitaire C++ d’appariement attend un répertoire `all/` (relatif à son chemin d’exécution) contenant des images lisibles en niveaux de gris.

Si votre configuration locale diffère, mettez à jour les cellules de chemins dans les notebooks et le répertoire d’entrée C++ en conséquence.

## Exemples

### Exécuter l’utilitaire d’appariement (exemple)

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

Comportement attendu :

- Lit les images depuis `all/`
- Calcule des appariements chaînés basés sur SIFT entre les images
- Écrit une image de sortie nommée comme `result_<timestamp>.png`

### Lancer un notebook spécifique

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## Notes de développement

- Le dépôt n’a actuellement pas de packaging à la racine (`pyproject.toml`, `requirements.txt` ou `setup.py`) ni de harnais CI/tests à la racine.
- Le travail est d’abord orienté expérimentation : les notebooks sont la source de vérité pour la plupart des algorithmes.
- `camera/`, `light_source/`, `reconstruction/` et `three_axis_cnc/` fournissent actuellement des descriptions de modules à haut niveau et peuvent être enrichis avec des runbooks au fil du temps.
- `i18n/` existe et est réservé aux variantes multilingues du README.

## Dépannage

- **Problèmes de résolution Conda :** mettez à jour Conda puis réessayez la création de l’environnement.
- **Incohérence de noyau dans les notebooks :** assurez-vous que le noyau actif correspond à `lensless` lorsque nécessaire.
- **Erreurs de compilation OpenCV/SIFT :** installez les modules OpenCV contrib et vérifiez la disponibilité de `opencv2/xfeatures2d.hpp`.
- **Erreurs de fichier introuvable dans les notebooks :** vérifiez les chemins des jeux de données et les répertoires relatifs attendus par les cellules.
- **Le matcher C++ ne lit aucune image :** vérifiez que `notebooks/multiple_match/all/` existe et contient des fichiers image valides.

## Feuille de route

- Étendre les runbooks au niveau module dans `camera/`, `light_source/`, `reconstruction/` et `three_axis_cnc/`.
- Documenter les contrats de jeux de données et fournir des pointeurs vers des exemples de données reproductibles.
- Ajouter des scripts reproductibles pour les principaux pipelines notebook.
- Ajouter des vérifications de test/validation pour les sorties de reconstruction et d’appariement.
- Finaliser les fichiers README multilingues sous `i18n/`.

## Participer

Nous accueillons volontiers la collaboration et les contributions. Si vous souhaitez vous impliquer dans le projet Lazeal OptiX, n’hésitez pas à soumettre une issue ou une pull request, ou à nous contacter directement.

## Contribution

1. Forkez le dépôt.
2. Créez une branche de fonctionnalité.
3. Gardez des changements ciblés et documentés (en particulier pour les notebooks).
4. Ouvrez une pull request décrivant la motivation, la méthode et la validation.

Si vous prévoyez des changements majeurs de matériel/protocole, il est recommandé d’ouvrir d’abord une issue pour alignement.

## Support

Aucune métadonnée dédiée de financement/sponsoring n’est actuellement déclarée dans ce dépôt.

Si cela change, les détails de sponsoring et de dons doivent être ajoutés ici sans supprimer la documentation technique existante.

## Licence

Aucun fichier de licence n’est actuellement présent à la racine du dépôt.

**Hypothèse/Action nécessaire :** ajouter un fichier `LICENSE` et mettre à jour cette section avec l’identifiant SPDX exact.

## Contact

Pour toute demande complémentaire ou intérêt de collaboration, veuillez nous contacter à `contact@lazealoptix.com`.
