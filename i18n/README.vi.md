[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


> 🌐 **Trạng thái đa ngôn ngữ:** `i18n/` đã tồn tại và được dành riêng cho các tệp README theo từng ngôn ngữ. Các tài liệu bản địa hóa được liên kết đang ở trạng thái kế hoạch/đang thực hiện.

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

*Mẫu thử cho sử dụng cá nhân (trái) và sử dụng tổ chức (phải)*

## Tổng quan

Lazeal OptiX là một dự án công nghệ y tế đổi mới. Trọng tâm của dự án là phát triển một thiết bị cung cấp khả năng chẩn đoán tiên tiến cho người dùng ngay tại nhà. Bằng cách sử dụng các kỹ thuật kính hiển vi tiên tiến và phân tích sinh hóa, thiết bị hướng tới việc hỗ trợ phát hiện sớm nhiều vấn đề sức khỏe, từ đó góp phần cải thiện kết quả chăm sóc y tế.

Dự án Lazeal OptiX ra đời từ cam kết giảm bớt đau khổ và giúp chẩn đoán sức khỏe dễ tiếp cận hơn cho mọi người. Bằng việc trang bị cho cá nhân các công cụ để chủ động kiểm soát sức khỏe, chúng tôi mong muốn góp phần tạo nên một xã hội khỏe mạnh hơn.

Kho mã hiện định hướng nghiên cứu/mẫu thử và lấy notebook làm trung tâm. Phần lớn chi tiết triển khai và thí nghiệm được theo dõi trong các Jupyter notebook dưới `notebooks/`.

### Tóm tắt nhanh

| Khu vực | Trạng thái hiện tại |
|---|---|
| Mức độ hoàn thiện dự án | Mẫu thử nghiên cứu |
| Mô hình thực thi chính | Quy trình Jupyter notebook |
| Miền thí nghiệm chính | Tái tạo ảnh, định vị nguồn sáng, so khớp đa ảnh |
| Packaging/CI ở thư mục gốc | Chưa được khai báo |
| Tài liệu đa ngôn ngữ | Có sẵn khung thư mục `i18n/` |

## Tính năng

1. **Kính hiển vi tiên tiến:** Tận dụng các kỹ thuật kính hiển vi tiên tiến để phân tích chi tiết.
2. **Phân tích sinh hóa:** Phân tích sinh hóa chuyên sâu giúp phát hiện nhiều chỉ dấu sức khỏe khác nhau.
3. **Thân thiện với người dùng:** Được thiết kế để dùng tại nhà, cung cấp giao diện đơn giản và dễ tiếp cận.
4. **Nhỏ gọn và chi phí hợp lý:** Lazeal OptiX có thiết kế nhỏ gọn, mức giá hợp lý, đưa khả năng chẩn đoán tiên tiến đến người dùng hằng ngày.
5. **Quy trình tái tạo không thấu kính:** Các pipeline ảnh tính toán và tái tạo dựa trên notebook.
6. **Thí nghiệm định vị nguồn sáng:** Notebook tối ưu hóa cho ước lượng vị trí nguồn sáng.
7. **Tiện ích so khớp đa ảnh:** Quy trình notebook và C++ OpenCV cho so khớp/căn chỉnh đặc trưng.

## Cấu trúc kho mã

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

### Ghi chú mô-đun

- `camera/`: script/tài nguyên liên quan đến sử dụng camera để chụp mẫu độ phân giải cao.
- `light_source/`: script/tài nguyên cho điều khiển và tối ưu nguồn sáng.
- `reconstruction/`: script/tài nguyên cho tái tạo tính toán.
- `three_axis_cnc/`: script/tài nguyên cho định vị/điều khiển CNC ba trục.
- `notebooks/`: không gian làm việc kỹ thuật chính cho thí nghiệm và phương pháp.

## Notebooks

Thư mục `notebooks` chứa các Jupyter notebook ghi lại nhiều khía cạnh của dự án Lazeal OptiX. Các notebook này bao gồm mã nguồn, trực quan hóa và giải thích chi tiết về phương pháp của dự án. Chúng là cách tương tác để khám phá và hiểu dự án.

### `light_source_location`

Thư mục `light_source_location` chứa các notebook liên quan đến ước lượng vị trí nguồn sáng. Các notebook này trình bày thuật toán và phương pháp dùng để ước lượng chính xác vị trí nguồn sáng, một thành phần quan trọng của dự án Lazeal OptiX.

### `multiple_match`

Thư mục `multiple_match` chứa notebook và script liên quan đến so khớp nhiều ảnh hoặc mẫu. Phần này của dự án sử dụng các thuật toán phức tạp để so khớp và căn chỉnh ảnh chính xác, điều cần thiết cho việc tái tạo ảnh độ phân giải cao từ hệ thống ảnh không thấu kính.

### `reconstruction`

Thư mục `reconstruction` chứa các notebook liên quan đến tái tạo ảnh được chụp bởi thiết bị Lazeal OptiX. Các notebook này ghi lại các kỹ thuật tính toán tiên tiến dùng để tái tạo ảnh độ phân giải cao từ hệ thống ảnh không thấu kính.

## Điều kiện tiên quyết

- OS: Khuyến nghị Linux/macOS cho quy trình notebook và OpenCV hiện tại.
- Python: Tệp môi trường được cung cấp nhắm tới **Python 3.7**.
- Conda: Bắt buộc để tái tạo môi trường `lensless` đã được mô tả.
- Jupyter Notebook/Lab.
- Bộ công cụ C++ tùy chọn cho `multiple_match.cpp`:
  - `g++` hỗ trợ C++17.
  - OpenCV 4.x kèm contrib modules (`opencv2/xfeatures2d.hpp` / SIFT).

## Cài đặt

### 1) Clone

```bash
git clone https://github.com/lachlanchen/lazealoptix.git
cd lazealoptix
```

### 2) Tạo môi trường notebook (khuyến nghị)

```bash
conda env create -f notebooks/reconstruction/lensless.yaml
conda activate lensless
```

### 3) Khởi chạy Jupyter

```bash
jupyter notebook
```

## Cách sử dụng

Kho mã này chủ yếu được sử dụng bằng cách mở notebook và chạy các cell theo thứ tự.

### Nhánh tái tạo

- Mở `notebooks/reconstruction/dataset_prep.ipynb` để chuẩn bị dữ liệu.
- Mở `notebooks/reconstruction/lensless-dropout-one-led-mahuichong.ipynb` cho thí nghiệm tái tạo/huấn luyện.

### Nhánh định vị nguồn sáng

- Mở các notebook dưới `notebooks/light_source_location/`.

### Nhánh multiple match

- Mở các notebook dưới `notebooks/multiple_match/`.
- Tiện ích C++ tùy chọn: `notebooks/multiple_match/multiple_match.cpp`.

## Cấu hình

### Môi trường Conda

Đặc tả môi trường chính nằm tại:

- `notebooks/reconstruction/lensless.yaml`

Một số tín hiệu phụ thuộc đáng chú ý từ tệp này bao gồm:

- `python=3.7`
- `pytorch=1.9.0`
- `pyro-ppl`
- Các phụ thuộc quy trình thị giác máy tính liên quan `opencv` trong notebook

### Dữ liệu và đường dẫn

- **Giả định:** notebook kỳ vọng dữ liệu/tệp cục bộ chưa được khai báo tập trung ở thư mục gốc của kho.
- **Giả định:** tiện ích so khớp C++ kỳ vọng có thư mục `all/` (tương đối với đường dẫn thực thi) chứa ảnh có thể đọc ở thang xám.

Nếu thiết lập cục bộ của bạn khác, hãy cập nhật các cell đường dẫn trong notebook và thư mục đầu vào C++ cho phù hợp.

## Ví dụ

### Chạy tiện ích so khớp (ví dụ)

```bash
cd notebooks/multiple_match
g++ -std=c++17 multiple_match.cpp -o multiple_match `pkg-config --cflags --libs opencv4`
./multiple_match
```

Hành vi dự kiến:

- Đọc ảnh từ `all/`
- Tính toán chuỗi so khớp dựa trên SIFT giữa các ảnh
- Ghi ảnh đầu ra có tên dạng `result_<timestamp>.png`

### Mở một notebook cụ thể

```bash
conda activate lensless
jupyter notebook notebooks/reconstruction/dataset_prep.ipynb
```

## Ghi chú phát triển

- Kho mã hiện chưa có packaging ở mức thư mục gốc (`pyproject.toml`, `requirements.txt`, hoặc `setup.py`) và chưa có CI/test harness ở thư mục gốc.
- Cách làm việc ưu tiên thí nghiệm: notebook là nguồn sự thật cho phần lớn thuật toán.
- `camera/`, `light_source/`, `reconstruction/`, và `three_axis_cnc/` hiện cung cấp mô tả mô-đun mức cao và có thể mở rộng bằng runbook theo thời gian.
- `i18n/` đã tồn tại và được dành cho các biến thể README đa ngôn ngữ.

## Khắc phục sự cố

- **Lỗi giải phụ thuộc Conda:** cập nhật Conda rồi thử tạo lại môi trường.
- **Sai kernel trong notebook:** đảm bảo kernel đang dùng khớp với `lensless` khi cần.
- **Lỗi biên dịch OpenCV/SIFT:** cài OpenCV contrib modules và kiểm tra tính sẵn có của `opencv2/xfeatures2d.hpp`.
- **Notebook lỗi không tìm thấy tệp:** kiểm tra đường dẫn dữ liệu và các thư mục tương đối mà cell notebook kỳ vọng.
- **Bộ so khớp C++ không đọc được ảnh:** xác minh `notebooks/multiple_match/all/` tồn tại và chứa tệp ảnh hợp lệ.

## Lộ trình

- Mở rộng runbook cấp mô-đun trong `camera/`, `light_source/`, `reconstruction/`, và `three_axis_cnc/`.
- Tài liệu hóa hợp đồng dữ liệu và cung cấp tham chiếu dữ liệu mẫu có thể tái lập.
- Bổ sung script tái lập cho các pipeline notebook trọng yếu.
- Bổ sung kiểm tra test/xác thực cho đầu ra tái tạo và so khớp.
- Hoàn thiện các tệp README đa ngôn ngữ dưới `i18n/`.

## Tham gia đóng góp

Chúng tôi hoan nghênh cộng tác và đóng góp. Nếu bạn muốn tham gia dự án Lazeal OptiX, hãy gửi issue hoặc pull request, hoặc liên hệ trực tiếp với chúng tôi.

## Đóng góp

1. Fork repository.
2. Tạo một nhánh tính năng.
3. Giữ phạm vi thay đổi rõ ràng và có tài liệu (đặc biệt với notebook).
4. Mở pull request mô tả động lực, phương pháp và cách xác thực.

Nếu bạn dự định thay đổi lớn về phần cứng/giao thức, nên mở issue trước để thống nhất.

## Hỗ trợ

Hiện chưa có metadata tài trợ/tài chính chuyên dụng được khai báo trong kho mã này.

Nếu điều này thay đổi, thông tin tài trợ và quyên góp nên được thêm vào đây mà không xóa tài liệu kỹ thuật hiện có.

## Giấy phép

Hiện chưa có tệp giấy phép ở thư mục gốc của kho mã.

**Giả định/Hành động cần thiết:** thêm tệp `LICENSE` và cập nhật mục này với mã định danh SPDX chính xác.

## Liên hệ

Để biết thêm thông tin hoặc trao đổi hợp tác, vui lòng liên hệ `contact@lazealoptix.com`.
