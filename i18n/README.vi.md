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

> 🌐 **Tình trạng đa ngôn ngữ:** `i18n/` đã có mặt và dành riêng cho các file README theo từng ngôn ngữ. Tài liệu nội dung địa phương hóa đã được lên kế hoạch/đang trong quá trình triển khai.

## ✨ Tổng quan nhanh

| Trọng tâm | Vị trí |
|---|---|
| Dòng công việc chính | `notebooks/` |
| Đặc tả môi trường | `notebooks/reconstruction/lensless.yaml` |
| Ghi chú thành phần | `camera/`, `light_source/`, `reconstruction/`, `three_axis_cnc/` |
| Tài liệu khởi động | `i18n/README.*.md` |

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

*Nguyên mẫu cho sử dụng cá nhân (bên trái) và sử dụng cho tổ chức (bên phải)*

## Tổng quan

Lazeal OptiX là một dự án nghiên cứu/ nguyên mẫu cho quy trình hình ảnh không kính trong các ứng dụng chẩn đoán liên quan y tế. Hiện tại kho lưu trữ chủ yếu theo hướng notebook-centric và mang tính thử nghiệm, nhằm mục tiêu giúp các phương pháp chẩn đoán tiên tiến tiếp cận tốt hơn trong các bối cảnh hạn chế.

Những ý tưởng cốt lõi bao gồm:

- tái tạo ảnh không kính,
- định vị nguồn sáng,
- ghép và căn chỉnh nhiều ảnh.

Kho lưu trữ này chủ yếu được duy trì qua các Jupyter notebook trong `notebooks/`, với ngữ cảnh theo từng module được lưu trong các thư mục chuyên biệt.

### Ảnh chụp nhanh trạng thái kho lưu trữ

| Khu vực | Tình trạng hiện tại |
|---|---|
| Độ chín muồi dự án | Nguyên mẫu nghiên cứu |
| Mô hình thực thi chính | Quy trình làm việc bằng notebook |
| Miền thử nghiệm chính | Tái tạo, định vị nguồn sáng, ghép nhiều ảnh |
| Packaging/CI tại root | Hiện chưa được khai báo |
| Tài liệu đa ngôn ngữ | Thư mục `i18n/` đã được chuẩn bị |

## Tính năng

1. **Khái niệm kính hiển vi nâng cao**: quang học và mẫu thu ảnh nâng cao phục vụ phân tích chi tiết.
2. **Ngữ cảnh sinh hóa / chẩn đoán**: quy trình thử nghiệm hướng tới phát hiện chỉ báo sức khỏe.
3. **Hướng đến người dùng tại nhà**: thiết kế nhằm sử dụng dễ tiếp cận và triển khai thực tế.
4. **Trải nghiệm ưu tiên máy tính xách tay**: notebook là con đường thực thi chính.
5. **Công cụ tái tạo không kính**: pipeline tính toán cho tái tạo độ phân giải cao.
6. **Công cụ định vị nguồn sáng**: thí nghiệm về xác định vị trí nguồn và hiệu chuẩn hình học.
7. **Ghép nhiều ảnh**: công cụ ghép bằng SIFT, nối chuỗi và căn chỉnh.

## Cấu trúc dự án

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

### Ghi chú module

- `camera/`: các script/tài nguyên liên quan đến việc sử dụng camera cho chụp mẫu độ phân giải cao.
- `light_source/`: các script/tài nguyên cho điều khiển và tối ưu nguồn sáng.
- `reconstruction/`: các script/tài nguyên cho tái tạo tính toán.
- `three_axis_cnc/`: các script/tài nguyên cho định vị/điều khiển CNC ba trục.
- `notebooks/`: không gian làm việc kỹ thuật chính cho thí nghiệm và phương pháp.

## Notebooks

Thư mục `notebooks` chứa các Jupyter notebook ghi lại các phương pháp thí nghiệm cốt lõi. Các notebook này cung cấp mã, trực quan hóa và ghi chú phương pháp cho từng mảng.

### `light_source_location`

Chứa các notebook liên quan đến ước lượng vị trí nguồn sáng. Các phương pháp này hỗ trợ hiệu chuẩn hình học nguồn và độ chính xác tái tạo.

### `multiple_match`

Chứa notebook và script cho ghép/căn chỉnh hình ảnh/chấm mẫu nhằm hỗ trợ quy trình đăng ký (registration) ổn định.

### `reconstruction`

Chứa notebook liên quan đến tái tạo từ ảnh đã thu thập, bao gồm tiền xử lý và script thí nghiệm.

## Điều kiện tiên quyết

- Hệ điều hành: Linux/macOS được khuyến nghị cho quy trình Conda và OpenCV hiện tại.
- Python: môi trường nhắm đến **Python 3.7**.
- Conda: cần thiết để tái tạo môi trường `lensless` đã được mô tả.
- Jupyter Notebook/Lab.
- Bộ công cụ C++ tùy chọn cho `multiple_match.cpp`:
  - `g++` với hỗ trợ C++17.
  - OpenCV 4.x cùng các module góp phần (`opencv2/xfeatures2d.hpp` / SIFT).

## Cài đặt

### 1) Clone

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) Tạo môi trường notebook

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) Khởi động Jupyter

```bash
jupyter notebook
```

## Cách sử dụng

Kho lưu trữ này chủ yếu được sử dụng bằng cách mở notebook và chạy từng cell theo thứ tự đã tài liệu hóa.

### Dòng tái tạo

- Mở `notebooks/reconstruction/dataset_prep.ipynb` để chuẩn bị bộ dữ liệu.
- Mở `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` cho các thí nghiệm tái tạo/huấn luyện.

### Dòng định vị nguồn sáng

- Mở các notebook trong `notebooks/light_source_location/`.

### Dòng ghép nhiều ảnh

- Mở các notebook trong `notebooks/multiple_match/`.
- Tiện ích tùy chọn: `notebooks/multiple_match/multiple_match.cpp`.

## Cấu hình

### Môi trường Conda

Đặc tả môi trường chính:

- `notebooks/reconstruction/lensless.yaml`

Các phụ thuộc đáng chú ý bao gồm:

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- `opencv` và các phụ thuộc workflow thị giác máy tính liên quan trong notebook

### Dữ liệu và đường dẫn

- **Giả định:** các bộ dữ liệu được lưu tại máy cục bộ và không được khai báo tập trung tại root của repository.
- **Giả định:** tiện ích khớp C++ mong đợi một thư mục `all/` (tương đối với đường dẫn chạy của nó) chứa ảnh đọc được ở thang độ xám.

Nếu cấu hình cục bộ của bạn khác, hãy cập nhật lại các ô đường dẫn trong notebook và thư mục đầu vào của C++ cho phù hợp.

## Ví dụ

### Chạy tiện ích ghép ảnh

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

Kết quả dự kiến:

- Đọc ảnh từ `all/`
- Tính toán chuỗi ghép dựa trên SIFT giữa các ảnh
- Ghi ảnh kết quả dạng `result_<timestamp>.png`

### Khởi chạy một notebook cụ thể

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## Ghi chú phát triển

- Hiện không có manifest đóng gói cấp root (`pyproject.toml`, `requirements.txt`, `setup.py`) hay bộ công cụ CI/test.
- Công việc theo hướng thí nghiệm trước; notebook là nguồn sự thật (source-of-truth) cho thuật toán hiện tại.
- `camera/`, `light_source/`, `reconstruction/`, và `three_axis_cnc/` chứa mô tả theo từng thành phần và là điểm mở rộng tốt cho runbooks.
- `i18n/` đã sẵn sàng cho tài liệu theo ngôn ngữ.

## Khắc phục sự cố

- **Vấn đề giải dependency Conda:** cập nhật Conda, kiểm tra thứ tự channel, và thử tạo môi trường lại.
- **Sai mismatch kernel trong notebook:** xác nhận Jupyter đang dùng môi trường `lensless`.
- **Lỗi biên dịch OpenCV/SIFT:** cài đặt OpenCV có module contrib và xác minh khả dụng của `opencv2/xfeatures2d.hpp`.
- **Lỗi file-not-found của notebook:** kiểm tra dataset và đường dẫn tương đối theo notebook.
- **Matcher không đọc được ảnh:** đảm bảo `notebooks/multiple_match/all/` tồn tại và chứa ảnh hợp lệ.

## Lộ trình

- Mở rộng các runbook theo module trong `camera/`, `light_source/`, `reconstruction/`, và `three_axis_cnc/`.
- Tài liệu hóa hợp đồng dữ liệu và cung cấp tham chiếu mẫu dữ liệu có thể tái lập.
- Thêm script wrapper cho các pipeline notebook chính.
- Thêm kiểm tra xác thực cho đầu ra tái tạo và ghép ảnh.
- Hoàn thiện README đa ngôn ngữ trong `i18n/`.

## Tham gia đóng góp

Chúng tôi hoan nghênh hợp tác và đóng góp.

- Mở issue để thảo luận.
- Tạo pull request cho các thay đổi tài liệu có phạm vi rõ ràng hoặc thay đổi thực nghiệm.
- Liên hệ với nhóm duy trì trước khi đề xuất thay đổi lớn ở mức phần cứng và giao thức.

## Giấy phép

Hiện không có file giấy phép nào trong root của repository.

**Giả định/Hành động cần thiết:** thêm file `LICENSE` và cập nhật mục này theo mã SPDX chính xác.

## Liên hệ

Để được trao đổi thêm hoặc quan tâm hợp tác, hãy liên hệ qua `contact@lazealoptix.com`.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
