[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


> 🌐 **حالة تعدد اللغات:** المجلد `i18n/` موجود ومخصص لملفات README الخاصة بكل لغة. المستندات المترجمة المرتبطة مخطط لها/قيد التنفيذ.

<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
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
      <img src="./prototype_individual.jpg" alt="Prototype for Individuals" style="width: 90%" />
    </td>
    <td align="center" valign="middle" width="50%">
      <img src="./prototype_institute.png" alt="Prototype for Institutions" style="width: 90%" />
    </td>
  </tr>
</table>

*نموذج أولي للاستخدام الفردي (يسارًا) وللاستخدام المؤسسي (يمينًا)*

## نظرة عامة

Lazeal OptiX هو مشروع مبتكر في تقنيات الرعاية الصحية. يتمحور المشروع حول تطوير جهاز يقدّم تشخيصات متقدمة للمستخدمين من منازلهم. وبالاعتماد على تقنيات المجهر المتقدمة والتحليل الكيميائي الحيوي، يهدف الجهاز إلى تسهيل الاكتشاف المبكر لمجموعة متنوعة من المشكلات الصحية، بما يسهم في تحسين النتائج الصحية.

وُلِد مشروع Lazeal OptiX من التزام بتقليل المعاناة وجعل التشخيص الصحي أكثر إتاحة للجميع. ومن خلال تزويد الأفراد بالأدوات اللازمة للتحكم في صحتهم، نسعى للمساهمة في بناء مجتمع أكثر صحة.

هذا المستودع موجّه حاليًا للأبحاث/النماذج الأولية ويرتكز على دفاتر Jupyter. يتم تتبّع معظم تفاصيل التنفيذ والتجارب داخل دفاتر Jupyter ضمن `notebooks/`.

### لمحة سريعة

| المجال | الحالة الحالية |
|---|---|
| نضج المشروع | نموذج أولي بحثي |
| نموذج التنفيذ الأساسي | سير عمل قائم على دفاتر Jupyter |
| مجالات التجارب الرئيسية | إعادة البناء، تحديد موقع مصدر الضوء، مطابقة صور متعددة |
| التحزيم/التكامل المستمر في الجذر | غير مُعلن حاليًا |
| التوثيق متعدد اللغات | يوجد هيكل مجلد `i18n/` |

## الميزات

1. **مجهرية متقدمة:** الاستفادة من تقنيات مجهرية متقدمة للتحليل التفصيلي.
2. **تحليل كيميائي حيوي:** تحليل كيميائي حيوي متعمّق يتيح كشف مؤشرات صحية مختلفة.
3. **سهل الاستخدام:** مصمم للاستخدام المنزلي بواجهة بسيطة وسهلة الوصول.
4. **مدمج وميسور التكلفة:** Lazeal OptiX مدمج وتكلفته مناسبة، ما يوفّر التشخيصات المتقدمة للمستخدمين يوميًا.
5. **سير عمل إعادة بناء بدون عدسة:** خطوط معالجة تصوير حاسوبي وإعادة بناء مبنية على الدفاتر.
6. **تجارب تحديد موقع مصدر الضوء:** دفاتر تحسين لتقدير موضع مصدر الضوء.
7. **أدوات مطابقة صور متعددة:** سير عمل دفاتر وC++ OpenCV لمطابقة/محاذاة السمات.

## بنية المستودع

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

### ملاحظات الوحدات

- `camera/`: سكربتات/موارد مرتبطة باستخدام الكاميرا لالتقاط عينات عالية الدقة.
- `light_source/`: سكربتات/موارد للتحكم في مصدر الضوء وتحسينه.
- `reconstruction/`: سكربتات/موارد لإعادة البناء الحاسوبي.
- `three_axis_cnc/`: سكربتات/موارد خاصة بالتموضع/التحكم عبر CNC ثلاثي المحاور.
- `notebooks/`: مساحة العمل التقنية الأساسية للتجارب والطرق.

## الدفاتر

يحتوي مجلد `notebooks` على دفاتر Jupyter توثّق جوانب مختلفة من مشروع Lazeal OptiX. تتضمن هذه الدفاتر الشيفرة، والتصورات، وشروحات تفصيلية لمنهجيات المشروع. وهي توفّر طريقة تفاعلية لاستكشاف المشروع وفهمه.

### `light_source_location`

يحتوي مجلد `light_source_location` على دفاتر مرتبطة بتقدير مواقع مصادر الضوء. تتضمن هذه الدفاتر خوارزميات وطرقًا تُستخدم لتقدير موضع مصدر الضوء بدقة، وهو جانب مهم في مشروع Lazeal OptiX.

### `multiple_match`

يحتوي مجلد `multiple_match` على دفاتر وسكربتات متعلقة بمطابقة صور أو أنماط متعددة. يتضمن هذا الجزء من المشروع خوارزميات معقدة لمطابقة الصور ومحاذاتها بدقة، وهو أمر ضروري لإعادة بناء صور عالية الدقة من نظام التصوير بدون عدسة.

### `reconstruction`

يحتوي مجلد `reconstruction` على دفاتر مرتبطة بإعادة بناء الصور الملتقطة بواسطة جهاز Lazeal OptiX. توثّق هذه الدفاتر التقنيات الحاسوبية المتقدمة المستخدمة لإعادة بناء صور عالية الدقة من نظام التصوير بدون عدسة.

## المتطلبات المسبقة

- نظام التشغيل: يُنصح بـ Linux/macOS لسير العمل الحالي المعتمد على الدفاتر وOpenCV.
- Python: ملف البيئة المرفق يستهدف **Python 3.7**.
- Conda: مطلوب لإعادة إنتاج بيئة `lensless` الموثقة.
- Jupyter Notebook/Lab.
- سلسلة أدوات C++ اختيارية لـ `multiple_match.cpp`:
  - `g++` مع دعم C++17.
  - OpenCV 4.x مع وحدات contrib (`opencv2/xfeatures2d.hpp` / SIFT).

## التثبيت

### 1) الاستنساخ

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) إنشاء بيئة الدفاتر (موصى به)

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) تشغيل Jupyter

```bash
jupyter notebook
```

## الاستخدام

يُستخدم هذا المستودع أساسًا عبر فتح الدفاتر وتشغيل الخلايا بالتسلسل.

### مسار إعادة البناء

- افتح `notebooks/reconstruction/dataset_prep.ipynb` لإعداد مجموعة البيانات.
- افتح `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` لتجارب إعادة البناء/التدريب.

### مسار تحديد موقع مصدر الضوء

- افتح الدفاتر ضمن `notebooks/light_source_location/`.

### مسار المطابقة المتعددة

- افتح الدفاتر ضمن `notebooks/multiple_match/`.
- أداة C++ اختيارية: `notebooks/multiple_match/multiple_match.cpp`.

## الإعداد

### بيئة Conda

يوجد تعريف البيئة الأساسي في:

- `notebooks/reconstruction/lensless.yaml`

من أبرز الإشارات إلى الاعتماديات في هذا الملف:

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- اعتماديات سير عمل رؤية حاسوبية مرتبطة بـ `opencv` داخل الدفاتر

### البيانات والمسارات

- **افتراض:** تتوقع الدفاتر وجود مجموعات بيانات/ملفات محلية غير معلنة مركزيًا في جذر المستودع.
- **افتراض:** تتوقع أداة المطابقة C++ وجود دليل `all/` (نسبيًا لمسار التنفيذ) يحتوي على صور قابلة للقراءة بتدرج رمادي.

إذا كان إعدادك المحلي مختلفًا، فحدّث خلايا المسارات في الدفاتر ودليل الإدخال الخاص بـ C++ وفقًا لذلك.

## أمثلة

### تشغيل أداة المطابقة (مثال)

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

السلوك المتوقع:

- قراءة الصور من `all/`
- حساب مطابقات متسلسلة قائمة على SIFT عبر الصور
- كتابة صورة خرج باسم مثل `result_<timestamp>.png`

### تشغيل دفتر محدد

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## ملاحظات التطوير

- لا يحتوي المستودع حاليًا على تحزيم في الجذر (`pyproject.toml` أو `requirements.txt` أو `setup.py`) ولا على حزمة CI/اختبارات في الجذر.
- العمل تجريبي أولًا: الدفاتر هي مصدر الحقيقة لمعظم الخوارزميات.
- توفّر `camera/` و`light_source/` و`reconstruction/` و`three_axis_cnc/` حاليًا أوصافًا عالية المستوى للوحدات ويمكن توسيعها لاحقًا بأدلة تشغيل.
- المجلد `i18n/` موجود ومخصص لنسخ README متعددة اللغات.

## استكشاف الأخطاء وإصلاحها

- **مشاكل حل اعتماديات Conda:** حدّث Conda ثم أعد محاولة إنشاء البيئة.
- **عدم تطابق النواة في الدفاتر:** تأكد أن النواة النشطة تطابق `lensless` عند الحاجة.
- **أخطاء تجميع OpenCV/SIFT:** ثبّت وحدات OpenCV contrib وتحقق من توفر `opencv2/xfeatures2d.hpp`.
- **أخطاء عدم العثور على ملفات في الدفاتر:** تحقق من مسارات البيانات والأدلة النسبية المتوقعة في خلايا الدفاتر.
- **أداة المطابقة C++ لا تقرأ صورًا:** تحقق من وجود `notebooks/multiple_match/all/` واحتوائه على ملفات صور صالحة.

## خارطة الطريق

- توسيع أدلة التشغيل على مستوى الوحدات في `camera/` و`light_source/` و`reconstruction/` و`three_axis_cnc/`.
- توثيق عقود مجموعات البيانات وتوفير مؤشرات لبيانات تجريبية قابلة لإعادة الإنتاج.
- إضافة سكربتات قابلة لإعادة الإنتاج لخطوط المعالجة الرئيسية في الدفاتر.
- إضافة فحوصات اختبار/تحقق لمخرجات إعادة البناء والمطابقة.
- استكمال ملفات README متعددة اللغات ضمن `i18n/`.

## المشاركة

نرحب بالتعاون والمساهمات. إذا كنت مهتمًا بالمشاركة في مشروع Lazeal OptiX، فلا تتردد في إرسال issue أو pull request أو التواصل معنا مباشرة.

## المساهمة

1. اعمل Fork للمستودع.
2. أنشئ فرع ميزة.
3. حافظ على نطاق تغييرات واضح وموثق (خاصة في الدفاتر).
4. افتح pull request يصف الدافع والمنهجية والتحقق.

إذا كنت تخطط لتغييرات كبيرة في العتاد/البروتوكول، يُنصح بفتح issue أولًا لضمان التوافق.

## الدعم

لا توجد حاليًا بيانات مخصصة للتمويل/الرعاية معلنة في هذا المستودع.

إذا تغيّر ذلك، فيجب إضافة تفاصيل الرعاية والتبرعات هنا دون إزالة التوثيق التقني الحالي.

## الترخيص

لا يوجد حاليًا ملف ترخيص في جذر المستودع.

**افتراض/إجراء مطلوب:** أضف ملف `LICENSE` وحدّث هذا القسم بمعرّف SPDX الدقيق.

## التواصل

للاستفسارات الإضافية أو الاهتمام بالتعاون، يُرجى التواصل عبر `contact@lazealoptix.com`.
