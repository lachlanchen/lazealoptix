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

> 🌐 **Multilingual status:** `i18n/` ist vorhanden und für sprachspezifische README-Dateien vorgesehen. Verlinkte lokalisierte Dokumente sind geplant/werden umgesetzt.

## ✨ Kurzüberblick

| Fokus | Ort |
|---|---|
| Kern-Workflow | `notebooks/` |
| Umgebungsspezifikation | `notebooks/reconstruction/lensless.yaml` |
| Modulnotizen | `camera/`, `light_source/`, `reconstruction/`, `three_axis_cnc/` |
| Einstiegsdokumentation | `i18n/README.*.md` |

<table width="100%">
  <tr>
    <td align="center" valign="middle" width="50%">
      <img src="./prototype_individual.jpg" alt="Prototype für Einzelpersonen" style="width: 90%" />
    </td>
    <td align="center" valign="middle" width="50%">
      <img src="./prototype_institute.png" alt="Prototype für Institutionen" style="width: 90%" />
    </td>
  </tr>
</table>

*Prototype für private Nutzung (links) und institutionelle Nutzung (rechts)*

## Überblick

Lazeal OptiX ist ein Forschungs-/Prototyp-Projekt für lensless-Imaging-Workflows in diagnosesnahen Anwendungen im Gesundheitsbereich. Das Repository ist aktuell notebook-zentriert und experimentell ausgelegt und soll fortgeschrittene Diagnoseansätze in ressourcenbeschränkten Umgebungen zugänglicher machen.

Zentrale Ideen sind:

- Lensless-Bildrekonstruktion,
- Lichtquellenlokalisierung,
- Mehrbildabgleich und -ausrichtung.

Das Repository wird hauptsächlich über Jupyter-Notebooks unter `notebooks/` betreut, wobei modulspezifischer Kontext in eigenen Verzeichnissen abgelegt ist.

### Schnappschuss des Projektstatus

| Bereich | Aktueller Status |
|---|---|
| Projektreife | Forschungsprototyp |
| Primäres Ausführungsmodell | Jupyter-Notebook-Workflows |
| Haupt-Experimentbereiche | Rekonstruktion, Lichtquellenlokalisierung, Mehrbildabgleich |
| Packaging/CI im Root | Derzeit nicht deklariert |
| Mehrsprachige Dokumentation | `i18n/`-Struktur vorhanden |

## Funktionen

1. **Fortgeschrittene Mikroskopie-Konzepte**: moderne Optik und Bildaufnahme-Muster für detaillierte Analysen.
2. **Biochemischer / diagnostischer Kontext**: experimentelle Workflows zur Erkennung gesundheitlicher Indikatoren.
3. **Wohnraumtauglicher Ansatz**: auf zugängliche Nutzung und praktische Bereitstellung ausgelegt.
4. **Laptop-zentrierte Nutzung**: Notebooks sind der primäre Ausführungsweg.
5. **Lensless-Rekonstruktionswerkzeuge**: Rechenpipelines für hochauflösende Rekonstruktion.
6. **Werkzeuge zur Lichtquellenlokalisierung**: Experimente zur Lokalisierung und geometrischen Kalibrierung der Quelle.
7. **Mehrbildabgleich**: SIFT-basiertes Matching, Verkettung und Ausrichtungs-Hilfsroutinen.

## Projektstruktur

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

### Modulhinweise

- `camera/`: Skripte/Ressourcen zur Kameranutzung für hochauflösende Probenerfassung.
- `light_source/`: Skripte/Ressourcen für Lichtquellensteuerung und -optimierung.
- `reconstruction/`: Skripte/Ressourcen für rechnergestützte Rekonstruktion.
- `three_axis_cnc/`: Skripte/Ressourcen für Dreiachs-CNC-Positionierung und -steuerung.
- `notebooks/`: Hauptarbeitsbereich für Experimente und Methoden.

## Notebooks

Das Verzeichnis `notebooks` enthält Jupyter-Notebooks, die die wichtigsten experimentellen Methoden dokumentieren. Diese Notebooks liefern Code, Visualisierungen und Methodennotizen für jedes Themengebiet.

### `light_source_location`

Enthält Notebooks zur Schätzung von Lichtquellenpositionen. Diese Methoden unterstützen die Geometrie-Kalibrierung der Quelle und die Rekonstruktionsgenauigkeit.

### `multiple_match`

Enthält Notebooks und Skripte für Bild-/Mustermatching und Ausrichtung, um robuste Registrierungs-Workflows zu unterstützen.

### `reconstruction`

Enthält Notebooks zur Rekonstruktion aus aufgenommenen Bildern, inklusive Preprocessing und Experimentskripten.

## Voraussetzungen

- OS: Linux/macOS empfohlen für aktuelle Conda- und OpenCV-Workflows.
- Python: Umgebung ist auf **Python 3.7** ausgelegt.
- Conda: notwendig, um die dokumentierte `lensless`-Umgebung nachzubilden.
- Jupyter Notebook/Lab.
- Optionaler C++-Toolchain für `multiple_match.cpp`:
  - `g++` mit C++17-Unterstützung.
  - OpenCV 4.x mit Contrib-Modulen (`opencv2/xfeatures2d.hpp` / SIFT).

## Installation

### 1) Klonen

```bash

git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) Notebook-Umgebung einrichten

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) Jupyter starten

```bash
jupyter notebook
```

## Nutzung

Dieses Repository wird primär über das Öffnen von Notebooks und das Ausführen der Zellen in dokumentierter Reihenfolge genutzt.

### Rekonstruktionspfad

- Öffne `notebooks/reconstruction/dataset_prep.ipynb` für die Datensatzvorbereitung.
- Öffne `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` für Rekonstruktions-/Trainingsversuche.

### Spur der Lichtquellenlokalisierung

- Öffne Notebooks unter `notebooks/light_source_location/`.

### Spur für mehrfachen Bildabgleich

- Öffne Notebooks unter `notebooks/multiple_match/`.
- Optionale Hilfskomponente: `notebooks/multiple_match/multiple_match.cpp`.

## Konfiguration

### Conda-Umgebung

Primäre Umgebungsdefinition:

- `notebooks/reconstruction/lensless.yaml`

Auffällige Abhängigkeiten:

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- `opencv`-bezogene Computer-Vision-Abhängigkeiten in den Notebooks

### Daten und Pfade

- **Annahme:** Datensätze sind lokal gespeichert und werden im Repository-Root nicht zentral beschrieben.
- **Annahme:** Das C++-Matching-Utility erwartet ein `all/`-Verzeichnis (relativ zu seinem Ausführungspfad) mit lesbaren Graustufen-Bildern.

Wenn deine lokale Einrichtung abweicht, passe Notebook-Pfadzellen und das C++-Eingabeverzeichnis entsprechend an.

## Beispiele

### Matching-Utility ausführen

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

Erwartetes Verhalten:

- Liest Bilder aus `all/`
- Berechnet verkettete SIFT-basierte Matches über die Bilder
- Schreibt ein Ausgabebild wie `result_<timestamp>.png`

### Ein bestimmtes Notebook starten

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## Entwicklungshinweise

- Kein Root-level Packaging-Manifest (`pyproject.toml`, `requirements.txt`, `setup.py`) oder CI/Test-Harness ist derzeit vorhanden.
- Die Arbeit ist experimentell ausgerichtet; Notebooks sind die Source-of-Truth für aktuelle Algorithmen.
- `camera/`, `light_source/`, `reconstruction/` und `three_axis_cnc/` enthalten beschreibende Modulkontexte und sind gute Einstiegspunkte für Runbooks.
- `i18n/` ist für sprachspezifische Dokumentation vorbereitet.

## Fehlerbehebung

- **Conda-Lö­sungsprobleme:** Aktualisiere Conda, prüfe die Kanal-Reihenfolge, und versuche die Umgebungsanlage erneut.
- **Kernel-Inkompatibilität in Notebooks:** Stelle sicher, dass Jupyter den `lensless`-Kernel verwendet.
- **OpenCV/SIFT-Kompilierfehler:** Installiere OpenCV-Contrib-Module und prüfe die Verfügbarkeit von `opencv2/xfeatures2d.hpp`.
- **Notebook-Datei nicht gefunden:** Prüfe die erwarteten Datensätze und relative Pfade in Notebooks.
- **Matcher liest keine Bilder:** Stelle sicher, dass `notebooks/multiple_match/all/` existiert und gültige Bilddateien enthält.

## Roadmap

- Modulbezogene Runbooks in `camera/`, `light_source/`, `reconstruction/` und `three_axis_cnc/` ausbauen.
- Datensatzverträge dokumentieren und reproduzierbare Referenzen zu Beispieldaten bereitstellen.
- Skript-Wrapper für wichtige Notebook-Pipelines ergänzen.
- Validierungsprüfungen für Rekonstruktions- und Matching-Ausgaben hinzufügen.
- Vollständige mehrsprachige README-Dateien unter `i18n/` abschließen.

## Mitwirken

Wir freuen uns auf Zusammenarbeit und Beiträge.

- Eröffne ein Issue für Diskussionen.
- Reiche einen Pull Request für klar abgegrenzte Dokumentations- oder Experimentänderungen ein.
- Kontaktiere die Maintainer vor größeren Änderungen an Hardware- und Protokollaspekten.

## Beitrag leisten

1. Forke das Repository.
2. Erstelle einen Feature-Branch.
3. Halte Änderungen fokussiert und dokumentiert (insbesondere bei Notebooks).
4. Öffne einen Pull Request mit Motivation, Methode und ggf. Validierungsnotizen.

## Lizenz

Im Repository-Root ist derzeit keine Lizenzdatei vorhanden.

**Annahme/Aktion erforderlich:** Füge eine `LICENSE`-Datei hinzu und aktualisiere diesen Abschnitt mit dem exakten SPDX-Bezeichner.

## Kontakt

Für weitere Anfragen oder Kooperationsinteresse wende dich bitte an `contact@lazealoptix.com`.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
