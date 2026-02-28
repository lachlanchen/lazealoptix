[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


> 🌐 **Estado multilingüe:** `i18n/` existe y está reservado para archivos README específicos por idioma. Los documentos localizados enlazados están planificados/en progreso.

<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="Banner de LazyingArt" />
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
      <img src="./prototype_individual.jpg" alt="Prototipo para uso individual" style="width: 90%" />
    </td>
    <td align="center" valign="middle" width="50%">
      <img src="./prototype_institute.png" alt="Prototipo para instituciones" style="width: 90%" />
    </td>
  </tr>
</table>

*Prototipo para uso individual (izquierda) y uso institucional (derecha)*

## Descripción general

Lazeal OptiX es un proyecto innovador de tecnología sanitaria. El núcleo del proyecto es el desarrollo de un dispositivo que ofrece diagnósticos avanzados a los usuarios desde la comodidad de sus hogares. Mediante técnicas avanzadas de microscopía y análisis bioquímico, el dispositivo busca facilitar la detección temprana de una variedad de problemas de salud, contribuyendo a mejores resultados asistenciales.

El proyecto Lazeal OptiX nace del compromiso de reducir el sufrimiento y hacer que los diagnósticos de salud sean más accesibles para todos. Al dotar a las personas de herramientas para tomar control de su salud, aspiramos a contribuir a una sociedad más saludable.

Actualmente, el repositorio está orientado a investigación/prototipo y centrado en notebooks. La mayoría de los detalles de implementación y experimentos se registran en notebooks de Jupyter dentro de `notebooks/`.

### Vista rápida

| Área | Estado actual |
|---|---|
| Madurez del proyecto | Prototipo de investigación |
| Modelo principal de ejecución | Flujos de trabajo con notebooks de Jupyter |
| Principales dominios experimentales | Reconstrucción, localización de fuente de luz, emparejamiento de múltiples imágenes |
| Empaquetado/CI en la raíz | No declarado actualmente |
| Documentación multilingüe | Existe la estructura del directorio `i18n/` |

## Características

1. **Microscopía avanzada:** Aprovecha técnicas avanzadas de microscopía para análisis detallado.
2. **Análisis bioquímico:** El análisis bioquímico en profundidad permite detectar diversos indicadores de salud.
3. **Fácil de usar:** Diseñado para uso doméstico, ofrece una interfaz simple y accesible.
4. **Compacto y asequible:** Lazeal OptiX es compacto y de precio asequible, acercando diagnósticos avanzados a usuarios cotidianos.
5. **Flujos de reconstrucción lensless:** Pipelines de imagen computacional y reconstrucción basados en notebooks.
6. **Experimentos de localización de fuente de luz:** Notebooks de optimización para estimar la posición de la fuente de luz.
7. **Utilidades de emparejamiento de múltiples imágenes:** Flujos de trabajo en notebooks y C++ con OpenCV para matching/alineación de características.

## Estructura del repositorio

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

### Notas de módulos

- `camera/`: scripts/recursos relacionados con el uso de cámara para captura de muestras en alta resolución.
- `light_source/`: scripts/recursos para control y optimización de la fuente de luz.
- `reconstruction/`: scripts/recursos para reconstrucción computacional.
- `three_axis_cnc/`: scripts/recursos para posicionamiento/control CNC de tres ejes.
- `notebooks/`: espacio técnico principal para experimentos y métodos.

## Notebooks

El directorio `notebooks` contiene notebooks de Jupyter que documentan varios aspectos del proyecto Lazeal OptiX. Estos notebooks incluyen código, visualizaciones y explicaciones detalladas de las metodologías del proyecto. Sirven como una forma interactiva de explorar y comprender el proyecto.

### `light_source_location`

El directorio `light_source_location` contiene notebooks relacionados con la estimación de ubicaciones de fuentes de luz. Estos notebooks presentan algoritmos y métodos utilizados para estimar con precisión la posición de la fuente de luz, un aspecto crucial del proyecto Lazeal OptiX.

### `multiple_match`

El directorio `multiple_match` contiene notebooks y scripts relacionados con el emparejamiento de múltiples imágenes o patrones. Esta parte del proyecto implica algoritmos complejos para emparejar y alinear imágenes con precisión, lo cual es necesario para la reconstrucción de imágenes de alta resolución a partir del sistema de imagen lensless.

### `reconstruction`

El directorio `reconstruction` contiene notebooks relacionados con la reconstrucción de imágenes capturadas por el dispositivo Lazeal OptiX. Estos notebooks documentan técnicas computacionales avanzadas utilizadas para reconstruir imágenes de alta resolución a partir del sistema de imagen lensless.

## Requisitos previos

- SO: Linux/macOS recomendados para los flujos actuales con notebooks y OpenCV.
- Python: el archivo de entorno proporcionado apunta a **Python 3.7**.
- Conda: requerido para reproducir el entorno documentado `lensless`.
- Jupyter Notebook/Lab.
- Cadena de herramientas C++ opcional para `multiple_match.cpp`:
  - `g++` con soporte C++17.
  - OpenCV 4.x con módulos contrib (`opencv2/xfeatures2d.hpp` / SIFT).

## Instalación

### 1) Clonar

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) Crear el entorno de notebooks (recomendado)

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) Iniciar Jupyter

```bash
jupyter notebook
```

## Uso

Este repositorio se utiliza principalmente abriendo notebooks y ejecutando celdas en secuencia.

### Ruta de reconstrucción

- Abrir `notebooks/reconstruction/dataset_prep.ipynb` para la preparación del conjunto de datos.
- Abrir `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` para experimentos de reconstrucción/entrenamiento.

### Ruta de localización de fuente de luz

- Abrir notebooks bajo `notebooks/light_source_location/`.

### Ruta de multiple match

- Abrir notebooks bajo `notebooks/multiple_match/`.
- Utilidad C++ opcional: `notebooks/multiple_match/multiple_match.cpp`.

## Configuración

### Entorno Conda

La especificación principal del entorno está en:

- `notebooks/reconstruction/lensless.yaml`

Señales de dependencias destacables de este archivo:

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- Dependencias de visión por computador relacionadas con `opencv` en notebooks

### Datos y rutas

- **Suposición:** los notebooks esperan datasets/archivos locales que no están declarados de forma central en la raíz del repositorio.
- **Suposición:** la utilidad de matching en C++ espera un directorio `all/` (relativo a su ruta de ejecución) que contenga imágenes legibles en escala de grises.

Si tu entorno local difiere, actualiza en consecuencia las celdas de rutas en notebooks y el directorio de entrada C++.

## Ejemplos

### Ejecutar la utilidad de matching (ejemplo)

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

Comportamiento esperado:

- Lee imágenes desde `all/`
- Calcula coincidencias encadenadas basadas en SIFT entre imágenes
- Escribe una imagen de salida con nombre como `result_<timestamp>.png`

### Iniciar un notebook específico

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## Notas de desarrollo

- Actualmente, el repositorio no tiene empaquetado a nivel raíz (`pyproject.toml`, `requirements.txt` o `setup.py`) ni un arnés de CI/pruebas en la raíz.
- El trabajo es primero experimental: los notebooks son la fuente de verdad para la mayoría de algoritmos.
- `camera/`, `light_source/`, `reconstruction/` y `three_axis_cnc/` actualmente ofrecen descripciones de módulos de alto nivel y pueden ampliarse con runbooks con el tiempo.
- `i18n/` existe y está reservado para variantes multilingües del README.

## Solución de problemas

- **Problemas al resolver Conda:** actualiza Conda y vuelve a intentar crear el entorno.
- **Desajuste de kernel en notebooks:** asegúrate de que el kernel activo coincida con `lensless` cuando sea necesario.
- **Errores de compilación OpenCV/SIFT:** instala módulos OpenCV contrib y verifica la disponibilidad de `opencv2/xfeatures2d.hpp`.
- **Errores de archivo no encontrado en notebooks:** revisa las rutas de datasets y directorios relativos esperados por las celdas.
- **El matcher C++ no lee imágenes:** verifica que exista `notebooks/multiple_match/all/` y que contenga archivos de imagen válidos.

## Hoja de ruta

- Ampliar runbooks a nivel de módulo en `camera/`, `light_source/`, `reconstruction/` y `three_axis_cnc/`.
- Documentar contratos de datasets y proporcionar referencias a datos de ejemplo reproducibles.
- Añadir scripts reproducibles para pipelines clave en notebooks.
- Añadir verificaciones de prueba/validación para salidas de reconstrucción y matching.
- Completar archivos README multilingües bajo `i18n/`.

## Participar

Damos la bienvenida a colaboraciones y contribuciones. Si te interesa participar en el proyecto Lazeal OptiX, no dudes en enviar un issue o un pull request, o en contactarnos directamente.

## Contribuciones

1. Haz un fork del repositorio.
2. Crea una rama de funcionalidad.
3. Mantén los cambios acotados y documentados (especialmente para notebooks).
4. Abre un pull request describiendo motivación, método y validación.

Si planeas cambios importantes de hardware/protocolo, se recomienda abrir primero un issue para alineación.

## Soporte

Actualmente no hay metadatos dedicados de financiación/patrocinio declarados en este repositorio.

Si esto cambia, los detalles de patrocinio y donaciones deberían añadirse aquí sin eliminar la documentación técnica existente.

## Licencia

Actualmente no hay un archivo de licencia presente en la raíz del repositorio.

**Suposición/Acción necesaria:** añadir un archivo `LICENSE` y actualizar esta sección con el identificador SPDX exacto.

## Contacto

Para más consultas o intereses de colaboración, contáctanos en `contact@lazealoptix.com`.
