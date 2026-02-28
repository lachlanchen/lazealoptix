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

> 🌐 **Estado multilingüe:** `i18n/` está presente y reservado para archivos README específicos por idioma. Los documentos localizados enlazados están planificados/en progreso.

## ✨ Visión general

| Enfoque | Ubicación |
|---|---|
| Flujo principal | `notebooks/` |
| Especificación de entorno | `notebooks/reconstruction/lensless.yaml` |
| Notas de componentes | `camera/`, `light_source/`, `reconstruction/`, `three_axis_cnc/` |
| Documentación de entrada | `i18n/README.*.md` |

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

Lazeal OptiX es un proyecto de investigación/prototipo para flujos de imagen sin lentes en diagnósticos de salud-adjuntos. Actualmente el repositorio está orientado a notebooks y es experimental, y pretende hacer más accesibles enfoques diagnósticos avanzados en entornos con recursos limitados.

Las ideas principales incluyen:

- reconstrucción de imagen sin lentes,
- localización de la fuente de luz,
- coincidencia y alineación de múltiples imágenes.

El repositorio se mantiene principalmente mediante notebooks de Jupyter en `notebooks/`, con contexto específico por módulo almacenado en directorios dedicados.

### Estado del repositorio

| Área | Estado actual |
|---|---|
| Madurez del proyecto | Prototipo de investigación |
| Modelo principal de ejecución | Flujos de trabajo en notebooks de Jupyter |
| Dominios experimentales principales | Reconstrucción, localización de fuente de luz, coincidencia de imágenes múltiples |
| Empaquetado/CI en la raíz | No declarado actualmente |
| Documentación multilingüe | Existe el scaffold del directorio `i18n/` |

## Características

1. **Conceptos avanzados de microscopía**: óptica avanzada y patrones de captura de imágenes para análisis detallado.
2. **Contexto bioquímico/diagnóstico**: flujos experimentales orientados a la detección de indicadores de salud.
3. **Diseño orientado al hogar**: pensado para un uso accesible y despliegue práctico.
4. **Experiencia centrada en laptop**: los notebooks proporcionan la ruta principal de ejecución.
5. **Utilidades de reconstrucción sin lentes**: pipelines computacionales para reconstrucción de alta resolución.
6. **Herramientas de localización de fuente de luz**: experimentos de localización de la fuente y calibración geométrica.
7. **Coincidencia de múltiples imágenes**: utilidades basadas en SIFT para coincidencia en cadena y alineación.

## Estructura del proyecto

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

### Notas de módulos

- `camera/`: scripts/recursos relacionados con el uso de cámara para captura de muestras de alta resolución.
- `light_source/`: scripts/recursos para control y optimización de la fuente de luz.
- `reconstruction/`: scripts/recursos para reconstrucción computacional.
- `three_axis_cnc/`: scripts/recursos para posicionamiento/control CNC de tres ejes.
- `notebooks/`: espacio técnico principal para experimentos y métodos.

## Notebooks

El directorio `notebooks` contiene notebooks de Jupyter que documentan los métodos experimentales principales. Estos notebooks proveen código, visualizaciones y notas metodológicas para cada área.

### `light_source_location`

Incluye notebooks relacionados con la estimación de localizaciones de fuentes de luz. Estos métodos apoyan la calibración geométrica de la fuente y la fidelidad de la reconstrucción.

### `multiple_match`

Incluye notebooks y scripts para coincidencia y alineación de imágenes/patrones para soportar flujos de registro robustos.

### `reconstruction`

Incluye notebooks relacionados con la reconstrucción a partir de imágenes capturadas, incluyendo preprocesamiento y scripts de experimento.

## Requisitos previos

- SO: Linux/macOS recomendado para los flujos actuales de Conda y OpenCV.
- Python: el entorno objetivo es **Python 3.7**.
- Conda: necesario para reproducir el entorno documentado `lensless`.
- Jupyter Notebook/Lab.
- Cadena de herramientas C++ opcional para `multiple_match.cpp`:
  - `g++` con soporte C++17.
  - OpenCV 4.x con módulos `contrib` (`opencv2/xfeatures2d.hpp` / SIFT).

## Instalación

### 1) Clonar

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) Crear el entorno de notebooks

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) Iniciar Jupyter

```bash
jupyter notebook
```

## Uso

Este repositorio se usa principalmente abriendo notebooks y ejecutando celdas en el orden documentado.

### Ruta de reconstrucción

- Abre `notebooks/reconstruction/dataset_prep.ipynb` para la preparación de conjuntos de datos.
- Abre `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` para experimentos de reconstrucción/entrenamiento.

### Ruta de localización de fuente de luz

- Abre los notebooks en `notebooks/light_source_location/`.

### Ruta de múltiples coincidencias

- Abre los notebooks en `notebooks/multiple_match/`.
- Utilidad opcional: `notebooks/multiple_match/multiple_match.cpp`.

## Configuración

### Entorno Conda

Especificación principal del entorno:

- `notebooks/reconstruction/lensless.yaml`

Dependencias destacadas:

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- dependencias de visión por computador con `opencv` en notebooks

### Datos y rutas

- **Supuesto:** los conjuntos de datos son locales y no se declaran de forma centralizada en la raíz del repositorio.
- **Supuesto:** la utilidad de coincidencia en C++ espera un directorio `all/` (relativo a su ruta de ejecución) con imágenes legibles en escala de grises.

Si tu configuración local difiere, actualiza las celdas de rutas del notebook y el directorio de entrada de C++ en consecuencia.

## Ejemplos

### Ejecutar la utilidad de coincidencia

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

Comportamiento esperado:

- Lee imágenes desde `all/`
- Calcula coincidencias en cadena basadas en SIFT entre imágenes
- Genera una imagen de salida con un nombre como `result_<timestamp>.png`

### Iniciar un notebook específico

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## Notas de desarrollo

- No existe en la raíz un manifiesto de empaquetado (`pyproject.toml`, `requirements.txt`, `setup.py`) ni un sistema de CI/pruebas actualmente.
- El trabajo es primero experimental; los notebooks son la fuente de verdad de los algoritmos actuales.
- `camera/`, `light_source/`, `reconstruction/` y `three_axis_cnc/` contienen descripciones a nivel de componente y son buenos puntos de extensión para runbooks.
- `i18n/` está preparado para documentación específica por idioma.

## Solución de problemas

- **Problemas al resolver Conda:** actualiza Conda, verifica el orden de canales y reintenta la creación del entorno.
- **Desajuste de kernel en notebooks:** confirma que Jupyter usa el entorno `lensless`.
- **Errores de compilación OpenCV/SIFT:** instala los módulos de OpenCV contrib y valida la disponibilidad de `opencv2/xfeatures2d.hpp`.
- **Errores de archivo no encontrado en notebooks:** verifica las rutas esperadas de datasets y rutas relativas del notebook.
- **El matcher no lee imágenes:** asegúrate de que existe `notebooks/multiple_match/all/` con archivos de imagen válidos.

## Hoja de ruta

- Ampliar runbooks a nivel de módulo en `camera/`, `light_source/`, `reconstruction/` y `three_axis_cnc/`.
- Documentar contratos de conjunto de datos y proveer referencias reproducibles de datos de ejemplo.
- Añadir envolturas de script para pipelines clave de notebooks.
- Añadir verificaciones de validación para salidas de reconstrucción y coincidencia.
- Completar archivos README multilingües en `i18n/`.

## Participación

Damos la bienvenida a colaboraciones y aportes.

- Abre un issue para discutir.
- Envía un pull request para cambios acotados de documentación o experimentos.
- Contacta a los mantenedores para cambios de hardware o protocolo antes de refactorizaciones grandes.

## Contribuciones

1. Haz un fork del repositorio.
2. Crea una rama de característica.
3. Mantén los cambios acotados y documentados (especialmente para notebooks).
4. Abre un pull request con motivación, método y cualquier nota de validación.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License

No existe actualmente un archivo de licencia en la raíz del repositorio.

**Suposición/acción necesaria:** añade un archivo `LICENSE` y actualiza esta sección con el identificador SPDX exacto.

## Contact

Para más consultas o interés en colaborar, escríbenos a `contact@lazealoptix.com`.
