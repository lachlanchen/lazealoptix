[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


> 🌐 **Mehrsprachiger Status:** `i18n/` ist vorhanden und für sprachspezifische README-Dateien reserviert. Verlinkte lokalisierte Dokumente sind geplant/in Arbeit.

<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt-Banner" />
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
      <img src="./prototype_individual.jpg" alt="Prototyp für Einzelpersonen" style="width: 90%" />
    </td>
    <td align="center" valign="middle" width="50%">
      <img src="./prototype_institute.png" alt="Prototyp für Institutionen" style="width: 90%" />
    </td>
  </tr>
</table>

*Prototyp für individuelle Nutzung (links) und institutionelle Nutzung (rechts)*

## Überblick

Lazeal OptiX ist ein innovatives Gesundheits-Technologieprojekt. Im Kern des Projekts steht die Entwicklung eines Geräts, das Nutzerinnen und Nutzern fortschrittliche Diagnostik im Komfort ihres Zuhauses bietet. Mithilfe moderner Mikroskopie und biochemischer Analysetechniken soll das Gerät die Früherkennung einer Vielzahl von Gesundheitsproblemen erleichtern und so zu besseren Behandlungsergebnissen beitragen.

Das Projekt Lazeal OptiX basiert auf dem Anspruch, Leiden zu verringern und Gesundheitsdiagnostik für alle zugänglicher zu machen. Indem wir Menschen mit den nötigen Werkzeugen ausstatten, damit sie ihre Gesundheit besser selbst steuern können, wollen wir zu einer gesünderen Gesellschaft beitragen.

Das Repository ist derzeit forschungs-/prototyporientiert und notebook-zentriert. Die meisten Implementierungsdetails und Experimente werden in Jupyter-Notebooks unter `notebooks/` dokumentiert.

### Auf einen Blick

| Bereich | Aktueller Status |
|---|---|
| Projektreife | Forschungsprototyp |
| Primäres Ausführungsmodell | Jupyter-Notebook-Workflows |
| Haupt-Experimentbereiche | Rekonstruktion, Lichtquellenlokalisierung, Mehrbild-Matching |
| Packaging/CI im Root | Derzeit nicht deklariert |
| Mehrsprachige Doku | `i18n/`-Verzeichnisgerüst vorhanden |

## Funktionen

1. **Fortgeschrittene Mikroskopie:** Nutzung moderner Mikroskopietechniken für detaillierte Analysen.
2. **Biochemische Analyse:** Tiefgehende biochemische Analyse ermöglicht die Erkennung verschiedener Gesundheitsindikatoren.
3. **Benutzerfreundlich:** Für den Heimgebrauch konzipiert, mit einfacher und zugänglicher Benutzeroberfläche.
4. **Kompakt und erschwinglich:** Lazeal OptiX ist kompakt und preislich zugänglich und bringt fortgeschrittene Diagnostik zu alltäglichen Nutzerinnen und Nutzern.
5. **Lensless-Rekonstruktions-Workflows:** Notebook-basierte Pipelines für Computational Imaging und Rekonstruktion.
6. **Experimente zur Lichtquellenlokalisierung:** Optimierungs-Notebooks zur Schätzung der Lichtquellenposition.
7. **Mehrbild-Matching-Utilities:** Notebook- und C++-OpenCV-Workflows für Feature-Matching/Ausrichtung.

## Repository-Struktur

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

### Modulhinweise

- `camera/`: Skripte/Ressourcen im Zusammenhang mit der Kameranutzung für hochauflösende Probenaufnahme.
- `light_source/`: Skripte/Ressourcen für Lichtquellensteuerung und -optimierung.
- `reconstruction/`: Skripte/Ressourcen für rechnergestützte Rekonstruktion.
- `three_axis_cnc/`: Skripte/Ressourcen für Positionierung/Steuerung mit Dreiachs-CNC.
- `notebooks/`: primärer technischer Arbeitsbereich für Experimente und Methoden.

## Notebooks

Das Verzeichnis `notebooks` enthält Jupyter-Notebooks, die verschiedene Aspekte des Lazeal-OptiX-Projekts dokumentieren. Diese Notebooks enthalten Code, Visualisierungen und detaillierte Erläuterungen zu den Projektmethoden. Sie dienen als interaktive Möglichkeit, das Projekt zu erkunden und zu verstehen.

### `light_source_location`

Das Verzeichnis `light_source_location` enthält Notebooks zur Schätzung von Lichtquellenpositionen. Diese Notebooks enthalten Algorithmen und Methoden zur präzisen Schätzung der Position der Lichtquelle, was ein zentraler Aspekt des Lazeal-OptiX-Projekts ist.

### `multiple_match`

Das Verzeichnis `multiple_match` enthält Notebooks und Skripte zum Matching mehrerer Bilder oder Muster. Dieser Teil des Projekts umfasst komplexe Algorithmen zur präzisen Zuordnung und Ausrichtung von Bildern, die für die Rekonstruktion hochauflösender Bilder aus dem lensless Bildgebungssystem erforderlich ist.

### `reconstruction`

Das Verzeichnis `reconstruction` enthält Notebooks zur Rekonstruktion von Bildern, die vom Lazeal-OptiX-Gerät erfasst wurden. Diese Notebooks dokumentieren die fortgeschrittenen rechnergestützten Techniken, die zur Rekonstruktion hochauflösender Bilder aus dem lensless Bildgebungssystem eingesetzt werden.

## Voraussetzungen

- OS: Linux/macOS empfohlen für aktuelle Notebook- und OpenCV-Workflows.
- Python: Die bereitgestellte Umgebungsdatei ist auf **Python 3.7** ausgelegt.
- Conda: Erforderlich, um die dokumentierte `lensless`-Umgebung nachzubilden.
- Jupyter Notebook/Lab.
- Optionales C++-Toolchain für `multiple_match.cpp`:
  - `g++` mit C++17-Unterstützung.
  - OpenCV 4.x mit Contrib-Modulen (`opencv2/xfeatures2d.hpp` / SIFT).

## Installation

### 1) Klonen

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) Notebook-Umgebung erstellen (empfohlen)

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) Jupyter starten

```bash
jupyter notebook
```

## Nutzung

Dieses Repository wird hauptsächlich verwendet, indem Notebooks geöffnet und Zellen der Reihe nach ausgeführt werden.

### Rekonstruktions-Track

- Öffne `notebooks/reconstruction/dataset_prep.ipynb` für die Datensatzvorbereitung.
- Öffne `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` für Rekonstruktions-/Trainings-Experimente.

### Track zur Lichtquellenlokalisierung

- Öffne Notebooks unter `notebooks/light_source_location/`.

### Multiple-Match-Track

- Öffne Notebooks unter `notebooks/multiple_match/`.
- Optionales C++-Utility: `notebooks/multiple_match/multiple_match.cpp`.

## Konfiguration

### Conda-Umgebung

Die primäre Umgebungsdefinition liegt unter:

- `notebooks/reconstruction/lensless.yaml`

Auffällige Abhängigkeitssignale aus dieser Datei umfassen:

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- `opencv`-nahe Computer-Vision-Workflow-Abhängigkeiten in Notebooks

### Daten und Pfade

- **Annahme:** Notebooks erwarten lokale Datensätze/Dateien, die im Repository-Root nicht zentral deklariert sind.
- **Annahme:** Das C++-Matching-Utility erwartet ein Verzeichnis `all/` (relativ zu seinem Ausführungspfad), das graustufenlesbare Bilder enthält.

Wenn dein lokales Setup abweicht, aktualisiere entsprechend die Notebook-Pfad-Zellen und das C++-Eingabeverzeichnis.

## Beispiele

### Matching-Utility ausführen (Beispiel)

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

Erwartetes Verhalten:

- Liest Bilder aus `all/`
- Berechnet verkettete SIFT-basierte Matches über mehrere Bilder
- Schreibt ein Ausgabebild mit einem Namen wie `result_<timestamp>.png`

### Ein spezifisches Notebook starten

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## Entwicklungshinweise

- Das Repository enthält derzeit kein Root-Level-Packaging (`pyproject.toml`, `requirements.txt` oder `setup.py`) und kein CI-/Test-Harness auf Root-Ebene.
- Die Arbeit ist experiment-first: Notebooks sind die Source-of-Truth für die meisten Algorithmen.
- `camera/`, `light_source/`, `reconstruction/` und `three_axis_cnc/` liefern derzeit übergeordnete Modulbeschreibungen und können im Laufe der Zeit um Runbooks erweitert werden.
- `i18n/` existiert und ist für mehrsprachige README-Varianten reserviert.

## Fehlerbehebung

- **Conda-Löseprobleme:** Conda aktualisieren und die Umgebungserstellung erneut versuchen.
- **Kernel-Mismatch in Notebooks:** sicherstellen, dass der aktive Kernel bei Bedarf zu `lensless` passt.
- **OpenCV/SIFT-Kompilierfehler:** OpenCV-Contrib-Module installieren und Verfügbarkeit von `opencv2/xfeatures2d.hpp` prüfen.
- **Notebook-Datei-nicht-gefunden-Fehler:** Datensatzpfade und von Notebook-Zellen erwartete relative Verzeichnisse prüfen.
- **C++-Matcher liest keine Bilder:** prüfen, ob `notebooks/multiple_match/all/` existiert und gültige Bilddateien enthält.

## Roadmap

- Modulbezogene Runbooks in `camera/`, `light_source/`, `reconstruction/` und `three_axis_cnc/` erweitern.
- Datensatzverträge dokumentieren und reproduzierbare Verweise auf Beispieldaten bereitstellen.
- Reproduzierbare Skripte für zentrale Notebook-Pipelines ergänzen.
- Test-/Validierungsprüfungen für Rekonstruktions- und Matching-Ausgaben ergänzen.
- Mehrsprachige README-Dateien unter `i18n/` vervollständigen.

## Mitwirken

Wir begrüßen Zusammenarbeit und Beiträge. Wenn du am Lazeal-OptiX-Projekt mitwirken möchtest, kannst du gerne ein Issue oder einen Pull Request einreichen oder uns direkt kontaktieren.

## Beitrag leisten

1. Forke das Repository.
2. Erstelle einen Feature-Branch.
3. Halte Änderungen begrenzt und dokumentiert (insbesondere bei Notebooks).
4. Öffne einen Pull Request mit Beschreibung von Motivation, Methode und Validierung.

Wenn du größere Hardware-/Protokolländerungen planst, empfiehlt sich zur Abstimmung zuerst ein Issue.

## Unterstützung

In diesem Repository sind derzeit keine dedizierten Finanzierungs-/Sponsoring-Metadaten deklariert.

Falls sich das ändert, sollten Sponsoring- und Spendeninformationen hier ergänzt werden, ohne bestehende technische Dokumentation zu entfernen.

## Lizenz

Im Repository-Root ist derzeit keine Lizenzdatei vorhanden.

**Annahme/Aktion erforderlich:** Eine `LICENSE`-Datei hinzufügen und diesen Abschnitt mit der exakten SPDX-Kennung aktualisieren.

## Kontakt

Für weitere Anfragen oder Interesse an einer Zusammenarbeit kontaktiere uns bitte unter `contact@lazealoptix.com`.
