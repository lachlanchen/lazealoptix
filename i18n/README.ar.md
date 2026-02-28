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

> 🌐 **حالة التعدد اللغوي:** المجلد `i18n/` موجود ومخصص لملفات README لكل لغة. مستندات اللغات المرتبطة مخطط لها/قيد الإعداد.

## ✨ نظرة سريعة

| التركيز | الموقع |
|---|---|
| مسار العمل الأساسي | `notebooks/` |
| مواصفات البيئة | `notebooks/reconstruction/lensless.yaml` |
| ملاحظات المكوّنات | `camera/`، `light_source/`، `reconstruction/`، `three_axis_cnc/` |
| مراجع الدخول | `i18n/README.*.md` |

<table width="100%">
  <tr>
    <td align="center" valign="middle" width="50%">
      <img src="./prototype_individual.jpg" alt="نموذج أولي للاستخدام الفردي" style="width: 90%" />
    </td>
    <td align="center" valign="middle" width="50%">
      <img src="./prototype_institute.png" alt="نموذج أولي للمؤسسات" style="width: 90%" />
    </td>
  </tr>
</table>

*النموذج الفردي من اليسار والمؤسسي من اليمين*

## نظرة عامة

Lazeal OptiX هو مشروع بحثي/نمذجي لسلاسل عمل التصوير عديم العدسة في التطبيقات القريبة من التشخيص الطبي. المستودع حاليا يعتمد على دفاتر الملاحظات وهو تجريبي، ويهدف إلى جعل الأساليب التشخيصية المتقدمة أكثر قابلية للوصول في البيئات ذات الموارد المحدودة.

تشمل الأفكار الأساسية:

- إعادة بناء الصور عديمة العدسة,
- تحديد موقع مصدر الضوء,
- المطابقة والمحاذاة بين صور متعددة.

المستودع يُدار بالأساس عبر دفاتر Jupyter في `notebooks/`، مع تخزين سياق كل وحدة ضمن الأدلة المخصصة.

### لقطة حالة المستودع

| المجال | الحالة الحالية |
|---|---|
| درجة نضج المشروع | نموذج بحثي |
| نموذج التشغيل الأساسي | مسارات عمل Jupyter Notebook |
| مجالات التجارب الأساسية | إعادة البناء، تحديد موقع مصدر الضوء، مطابقة عدة صور |
| الحزم/CI في الجذر | غير معلنة حالياً |
| وثائق متعددة اللغات | بنية المجلد `i18n/` موجودة |

## الميزات

1. **مفاهيم مجهرية متقدمة**: بصرية متقدمة وأنماط التقاط صور للتحليل التفصيلي.
2. **سياق بيولوجي / تشخيصي**: سلاسل عمل تجريبية موجهة لاكتشاف مؤشرات صحية.
3. **اتجاه يناسب الاستخدام المنزلي**: مصمم لاستخدام ميسّر ونشر عملي.
4. **تجربة موجهة إلى الحاسوب المحمول**: دفاتر الملاحظات توفر المسار الأساسي للتنفيذ.
5. **أدوات إعادة البناء عديمة العدسة**: خطوط أنابيب حسابية لإعادة البناء بدقة عالية.
6. **أدوات تحديد مصدر الضوء**: تجارب خاصة بمعايرة هندسة المصدر.
7. **مطابقة صور متعددة**: مطابقة SIFT المتسلسلة، الربط، وأدوات المحاذاة.

## هيكل المشروع

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

### ملاحظات الوحدات

- `camera/`: سكربتات/موارد مرتبطة باستخدام الكاميرا لالتقاط عينات بدقة عالية.
- `light_source/`: سكربتات/موارد للتحكم في مصدر الضوء وتحسينه.
- `reconstruction/`: سكربتات/موارد لإعادة البناء الحسابية.
- `three_axis_cnc/`: سكربتات/موارد للتحكم/الموضعية بمحور ثلاثي للمحركات CNC.
- `notebooks/`: مساحة العمل التقنية الأساسية للتجارب والأساليب.

## دفاتر الملاحظات

دليل `notebooks` يحتوي على دفاتر Jupyter التي توثّق الطرق التجريبية الأساسية. هذه الدفاتر توفر كودًا، مرئيات، وملاحظات منهجية لكل منطقة.

### `light_source_location`

يحتوي دفاتر تتعلق بتقدير مواضع مصادر الضوء. هذه الطرق تدعم معايرة هندسة المصدر وتحسين دقة إعادة البناء.

### `multiple_match`

يحتوي دفاتر وسكربتات لمطابقة الصور/الأنماط ومحاذاتها لدعم تدفقات تسجيل أكثر موثوقية.

### `reconstruction`

يحتوي دفاتر تتعلق بإعادة البناء من الصور الملتقطة، بما في ذلك إعدادات ما قبل المعالجة وسكربتات التجارب.

## المتطلبات المسبقة

- نظام التشغيل: Linux/macOS موصى به لبيئات Conda وOpenCV الحالية.
- Python: البيئة تستهدف **Python 3.7**.
- Conda: مطلوبة لإعادة إنشاء بيئة `lensless` الموثقة.
- Jupyter Notebook/Lab.
- سلسلة أدوات C++ اختيارية لـ `multiple_match.cpp`:
  - `g++` بدعم C++17.
  - OpenCV 4.x مع وحدة contrib (`opencv2/xfeatures2d.hpp` / SIFT).

## التثبيت

### 1) الاستنساخ

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) إنشاء بيئة الدفاتر

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) تشغيل Jupyter

```bash
jupyter notebook
```

## طريقة الاستخدام

يُستخدم هذا المستودع أساسًا عبر فتح دفاتر الملاحظات وتشغيل الخلايا بالترتيب الموثق.

### مسار إعادة البناء

- افتح `notebooks/reconstruction/dataset_prep.ipynb` لإعداد مجموعة البيانات.
- افتح `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` لتجارب إعادة البناء/التدريب.

### مسار تحديد موقع مصدر الضوء

- افتح دفاتر العمل ضمن `notebooks/light_source_location/`.

### مسار المطابقة المتعددة

- افتح دفاتر العمل ضمن `notebooks/multiple_match/`.
- أداة إضافية: `notebooks/multiple_match/multiple_match.cpp`.

## الإعدادات

### بيئة Conda

المواصفات الرئيسية للبيئة:

- `notebooks/reconstruction/lensless.yaml`

تشمل التبعيات البارزة:

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- تبعيات سير عمل الرؤية الحاسوبية المرتبطة بـ OpenCV داخل الدفاتر

### البيانات والمسارات

- **افتراض:** توجد مجموعات البيانات محليًا وليست معلنة مركزيًا في جذر المستودع.
- **افتراض:** أداة المطابقة في C++ تتوقع وجود دليل `all/` (نسبي لمسار التنفيذ) يحتوي على صور قابلة للقراءة بدرجات رمادية.

إذا اختلف إعدادك المحلي، حدّث خلايا المسارات داخل الدفاتر ودليل إدخال C++ حسب الحالة.

## أمثلة

### تشغيل أداة المطابقة

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

السلوك المتوقع:

- تقرأ الصور من `all/`
- تحسب تطابقات SIFT المتسلسلة عبر الصور
- تكتب صورة مخرجات شبيهة بـ `result_<timestamp>.png`

### تشغيل دفتر عمل محدد

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## ملاحظات التطوير

- لا يوجد حاليًا ملف توزيع جذري على مستوى الجذر (`pyproject.toml`، `requirements.txt`، `setup.py`) ولا بنية CI/اختبارات.
- العمل قائم على التجربة أولاً؛ دفاتر الملاحظات هي مصدر الحقيقة الحالي للخوارزميات.
- تحتوي المجلدات `camera/`، `light_source/`، `reconstruction/`، و`three_axis_cnc/` على أوصاف على مستوى الوحدات وتعتبر نقاط تمدد ممتازة لدلائل التشغيل.
- المجلد `i18n/` مُعد بالفعل لتوثيق متعدد اللغات.

## استكشاف الأخطاء وإصلاحها

- **مشاكل حل Conda:** حدّث Conda، تحقق من ترتيب القنوات، ثم أعد محاولة إنشاء البيئة.
- **عدم تطابق النواة في دفاتر الملاحظات:** تأكد من أن Jupyter يستخدم البيئة `lensless`.
- **أخطاء تجميع OpenCV/SIFT:** ثبّت وحدات contrib الخاصة بـ OpenCV وتأكد من توافر `opencv2/xfeatures2d.hpp`.
- **أخطاء عدم العثور على ملفات الدفاتر:** تحقق من مسارات مجموعات البيانات المتوقعة ومسارات الملفات النسبية داخل الدفتر.
- **عدم قراءة أي صور بواسطة المطابقة:** تأكد من وجود `notebooks/multiple_match/all/` مع ملفات صور صالحة.

## خارطة الطريق

- توسيع دلائل التشغيل على مستوى الوحدات في `camera/` و`light_source/` و`reconstruction/` و`three_axis_cnc/`.
- توثيق اتفاقيات مجموعات البيانات وتوفير مراجع بيانات تجريبية قابلة لإعادة الاستخدام.
- إضافة أغلفة سكربتية لخطوط أنابيب الدفاتر الرئيسية.
- إضافة فحوص تحقق لمخرجات إعادة البناء والمطابقة.
- إكمال ملفات README متعددة اللغات داخل `i18n/`.

## كيف تشارك

نرحب بالتعاون والمساهمات.

- افتح تذكرة نقاش (issue) للنقاش.
- أرسل طلب سحب (pull request) لتغييرات موثقة ونطاقية، وخصوصًا في دفاتر العمل.
- تواصل مع المشرفين قبل القيام بتغييرات كبيرة في العتاد أو بروتوكول العمل.

## المساهمة

1. استنسخ المستودع (Fork).
2. أنشئ فرعًا للميزة الجديدة.
3. التزم بنطاق واضح وموثق (خاصةً لدفاتر الملاحظات).
4. افتح طلب سحب مع الدافع والطريقة وأي ملاحظات تحقق.

## الترخيص

لا يوجد ملف ترخيص حاليًا في جذر المستودع.

**افتراض/إجراء مطلوب:** أضف ملف `LICENSE` وحدث هذا القسم بمعرف SPDX الدقيق.

## التواصل

للاستفسارات الإضافية أو فرص التعاون، يرجى التواصل عبر `contact@lazealoptix.com`.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
