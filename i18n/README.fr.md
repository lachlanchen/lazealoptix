[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Lazeal OptiX

![Project Stage](https://img.shields.io/badge/stage-research%20prototype-orange)
![Primary Workflow](https://img.shields.io/badge/workflow-notebook--centric-blue)
![Python](https://img.shields.io/badge/python-3.7-blue)
![Conda](https://img.shields.io/badge/environment-conda-44A833)
![Jupyter](https://img.shields.io/badge/interface-jupyter-F37626)
![OpenCV](https://img.shields.io/badge/cv-opencv%204.x-5C3EE8)
![License](https://img.shields.io/badge/license-TBD-lightgrey)
![Localization](https://img.shields.io/badge/localization-11%20languages-8A4FFF)
![Platform](https://img.shields.io/badge/platform-linux%2FmacOS-2D9CDB)

> 🌐 **Statut multilingue :** `i18n/` est présent et réservé aux fichiers README par langue. Les documents localisés associés sont planifiés ou en cours.

## ✨ Vue d'ensemble

| Focus | Emplacement |
|---|---|
| Workflow principal | `notebooks/` |
| Spécification de l'environnement | `notebooks/reconstruction/lensless.yaml` |
| Notes de composants | `camera/`, `light_source/`, `reconstruction/`, `three_axis_cnc/` |
| Documentation d'entrée | `i18n/README.*.md` |

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

*Prototype pour usage individuel (à gauche) et usage institutionnel (à droite)*

## Aperçu

Lazeal OptiX est un projet de recherche/prototype pour des flux d'imagerie lensless dans des contextes de diagnostic liés à la santé. Le dépôt est actuellement centré sur les notebooks et de nature expérimentale, et vise à rendre des approches de diagnostic avancé plus accessibles dans des environnements contraints.

Principales idées :

- reconstruction d'images sans lentille,
- localisation de la source lumineuse,
- appariement et alignement d'images multiples.

Le dépôt est principalement maintenu via des notebooks Jupyter sous `notebooks/`, avec un contexte spécifique par module stocké dans des répertoires dédiés.

### Instantané de l'état du dépôt

| Domaine | Statut actuel |
|---|---|
| Maturité du projet | Prototype de recherche |
| Modèle d'exécution principal | Workflows avec notebooks Jupyter |
| Domaines d'expérimentation principaux | Reconstruction, localisation de la source lumineuse, appariement d'images multiples |
| Packaging/CI à la racine | Non déclaré actuellement |
| Documentation multilingue | Répertoire `i18n/` présent |

## Fonctionnalités

1. **Concepts avancés de microscopie** : optique avancée et motifs de capture d'image pour une analyse détaillée.
2. **Contexte biochimique / diagnostic** : workflows expérimentaux ciblés vers la détection d'indicateurs de santé.
3. **Orientation conviviale** : conçu pour une utilisation simple et un déploiement pratique.
4. **Expérience laptop-first** : les notebooks fournissent le chemin d'exécution principal.
5. **Utilitaires de reconstruction lensless** : pipelines computationnels pour une reconstruction haute résolution.
6. **Outils de localisation de source lumineuse** : expériences de calibration géométrique et de localisation.
7. **Appariement d'images multiples** : appariement basé sur SIFT, chaînage et outils d'alignement.

## Structure du projet

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
│   │   ├── light_source_location_estimator_v1.4.ipynb
│   │   ├── light_source_location_estimator_varied_heights_v1.1.4.ipynb
│   │   └── light_source_location_estimator_varied_heights_v1.1.7.ipynb
│   ├── multiple_match/
│   │   ├── multiple_all_combination_v2.ipynb
│   │   ├── multiple_match.cpp
│   │   ├── multiple_match_centeralized_v1.6.ipynb
│   │   └── multiple_match_chain_v1.5.ipynb
│   └── reconstruction/
│       ├── dataset_prep.ipynb
│       ├── lensless.yaml
│       └── lensless-dropout-one-led-mahuichong.ipynb
└── i18n/
```

### Notes par module

- `camera/` : scripts/ressources liés à l'utilisation de la caméra pour la capture d'échantillons en haute résolution.
- `light_source/` : scripts/ressources pour le contrôle et l'optimisation de la source lumineuse.
- `reconstruction/` : scripts/ressources pour la reconstruction computationnelle.
- `three_axis_cnc/` : scripts/ressources pour le positionnement/contrôle CNC à trois axes.
- `notebooks/` : espace de travail technique principal pour les expériences et méthodes.

## Notebooks

Le répertoire `notebooks` contient des notebooks Jupyter qui documentent les méthodes expérimentales centrales. Ces notebooks fournissent le code, les visualisations et les notes méthodologiques pour chaque domaine.

### `light_source_location`

Contient les notebooks liés à l'estimation de la position de la source lumineuse. Ces méthodes supportent la calibration géométrique de la source et la fidélité de reconstruction.

### `multiple_match`

Contient des notebooks et scripts pour l'appariement d'images/modèles et l'alignement afin d'appuyer des workflows d'enregistrement robuste.

### `reconstruction`

Contient des notebooks liés à la reconstruction depuis des images capturées, y compris les scripts de prétraitement et d'expériences.

## Prérequis

- OS : Linux/macOS recommandé pour les workflows Conda et OpenCV actuels.
- Python : environnement ciblé **Python 3.7**.
- Conda : nécessaire pour reproduire l'environnement `lensless` documenté.
- Jupyter Notebook/Lab.
- Chaîne d'outils C++ optionnelle pour `multiple_match.cpp` :
  - `g++` avec prise en charge C++17.
  - OpenCV 4.x avec les modules contrib (`opencv2/xfeatures2d.hpp` / SIFT).

## Installation

### 1) Cloner

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) Créer l'environnement notebook

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) Démarrer Jupyter

```bash
jupyter notebook
```

## Utilisation

Ce dépôt est principalement utilisé en ouvrant les notebooks et en exécutant les cellules dans l'ordre documenté.

### Piste de reconstruction

- Ouvrez `notebooks/reconstruction/dataset_prep.ipynb` pour la préparation du jeu de données.
- Ouvrez `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` pour les expériences de reconstruction/entraînement.

### Piste de localisation de source lumineuse

- Ouvrez les notebooks sous `notebooks/light_source_location/`.

### Piste de correspondance multiple

- Ouvrez les notebooks sous `notebooks/multiple_match/`.
- Utilitaire optionnel : `notebooks/multiple_match/multiple_match.cpp`.

## Configuration

### Environnement Conda

Spécification principale de l'environnement :

- `notebooks/reconstruction/lensless.yaml`

Dépendances notables :

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- dépendances de vision par ordinateur proches d'OpenCV dans les notebooks

### Données et chemins

- **Hypothèse :** les jeux de données sont locaux et ne sont pas déclarés de manière centralisée à la racine du dépôt.
- **Hypothèse :** l'utilitaire de matching C++ attend un répertoire `all/` (relatif à son chemin d'exécution) contenant des images lisibles en niveaux de gris.

Si votre configuration locale diffère, mettez à jour les cellules de chemins des notebooks et le répertoire d'entrée C++ en conséquence.

## Exemples

### Exécuter l'utilitaire de matching

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

Comportement attendu :

- Lit les images de `all/`
- Calcule des appariements SIFT chaînés entre les images
- Écrit une image de sortie comme `result_<timestamp>.png`

### Lancer un notebook spécifique

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## Notes de développement

- Aucun manifeste de packaging au niveau racine (`pyproject.toml`, `requirements.txt`, `setup.py`) ou harness CI/tests n'est actuellement présent.
- Le travail est d'abord orienté expérimentation ; les notebooks sont la source de vérité des algorithmes actuels.
- `camera/`, `light_source/`, `reconstruction/` et `three_axis_cnc/` contiennent des descriptions au niveau des composants et constituent de bons points d'extension pour des runbooks.
- `i18n/` est préparé pour la documentation spécifique à chaque langue.

## Dépannage

- **Problèmes de résolution Conda :** mettez à jour Conda, vérifiez l'ordre des canaux, puis relancez la création d'environnement.
- **Incompatibilité de noyau dans les notebooks :** vérifiez que Jupyter utilise bien l'environnement `lensless`.
- **Erreurs de compilation OpenCV/SIFT :** installez les modules OpenCV contrib et validez la disponibilité de `opencv2/xfeatures2d.hpp`.
- **Erreurs de fichiers manquants dans les notebooks :** vérifiez les jeux de données attendus et les chemins relatifs aux notebooks.
- **Le matcher ne lit aucune image :** assurez-vous que `notebooks/multiple_match/all/` existe avec des fichiers image valides.

## Feuille de route

- Étendre les runbooks au niveau module dans `camera/`, `light_source/`, `reconstruction/` et `three_axis_cnc/`.
- Documenter les contrats de jeux de données et fournir des références d'échantillons reproductibles.
- Ajouter des wrappers de scripts pour les pipelines notebook principaux.
- Ajouter des contrôles de validation pour les sorties de reconstruction et d'appariement.
- Finaliser les README multilingues sous `i18n/`.

## Participation

Nous accueillons la collaboration et les contributions.

- Ouvrez une issue pour lancer une discussion.
- Soumettez une pull request pour des changements ciblés de documentation ou expérimentaux.
- Contactez les mainteneurs pour les changements matériels/protocolaires avant les refactors importants.

## Contribution

1. Forkez le dépôt.
2. Créez une branche de fonctionnalité.
3. Gardez les changements ciblés et documentés (en particulier pour les notebooks).
4. Ouvrez une pull request avec la motivation, la méthode et les notes de validation.

## Licence

Aucun fichier de licence n'est actuellement présent à la racine du dépôt.

**Hypothèse/Action nécessaire :** ajouter un fichier `LICENSE` et mettre à jour cette section avec l'identifiant SPDX exact.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## Contact

Pour toute question supplémentaire ou intérêt de collaboration, veuillez nous contacter à `contact@lazealoptix.com`.
