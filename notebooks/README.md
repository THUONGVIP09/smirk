# Notebooks Kaggle của dự án (theo từng giai đoạn)

Tất cả notebook được chạy trên **Kaggle** (máy cá nhân không đủ GPU). Chạy theo thứ tự thư mục `01 → 02 → 03`.
Đầu ra của mỗi notebook được lưu thành **Kaggle Dataset (Private)** để notebook sau dùng làm Input.

Script Python mà các notebook dùng nằm trong [`datasets/preprocess_scripts/`](../datasets/preprocess_scripts/)
(được chép vào notebook bằng các cell `%%writefile`) và trong [`src/`](../src/) (notebook giai đoạn 03 clone repo từ GitHub).

```
01_data_preparation  →  02_data_cleaning  →  03_smirk_setup  →  04_baseline_evaluation  →  (05 huấn luyện: sắp làm)
 crop / chuẩn hóa        landmark + lọc        cài SMIRK, demo,     SMIRK_em1 trên ảnh sạch
 ảnh 256x256 + nhãn      + chia train/val/test  che khuất nhân tạo   vs ảnh bị che (số liệu)
```

## 01_data_preparation: chuẩn hóa dữ liệu

Đưa từng bộ dữ liệu về cùng định dạng: ảnh mặt 256×256 PNG + `meta.csv` (nhãn cảm xúc theo cùng một thứ tự
0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral, 7 contempt).

| Notebook | Làm gì | Input (Kaggle) | Output → Dataset | Kết quả |
|---|---|---|---|---|
| `kaggle_crop_expw.ipynb` | Giải nén ExpW (7z 8 phần), crop từng mặt theo bounding box, bỏ mặt < 160 px | Expression in-the-Wild (ExpW) | `expw-crops` | 25.611 / 91.793 mặt |
| `kaggle_prepare_affectnet.ipynb` | Chuẩn hóa AffectNet (bản Kaggle, thư mục `0..7`), kiểm tra thứ tự nhãn bằng lưới ảnh | AffectNet (Kaggle, không chính thức) | `affectnet-crops` | 40.928 ảnh |
| `kaggle_prepare_rafdb.ipynb` | Tải RAF-DB chính thức từ Google Drive (link trong email RAF, **không đưa link vào repo**), crop theo bounding box, đổi nhãn RAF (1..7) sang thứ tự chung, bỏ mặt < 80 px | Link Drive RAF-DB | `rafdb-crops` | 11.850 ảnh (9.472 train / 2.378 test) |

Script: `crop_expw.py`, `prepare_affectnet.py`, `prepare_rafdb.py`. Cài đặt Kaggle: Accelerator None, Internet On.

## 02_data_cleaning: landmark, lọc ảnh sạch, chia dữ liệu

| Notebook | Làm gì | Input | Output → Dataset |
|---|---|---|---|
| `kaggle_clean_faces.ipynb` | MediaPipe 478 landmark → đo độ nét, viền đen, mặt bị cắt, face parsing BiSeNet (kính, mũ, vật che vùng mắt–mũi–miệng) → lọc theo ngưỡng → chia train/val/test theo lớp | các dataset `*-crops` đang gắn làm Input (tự nhận diện) | `clean-faces` (ExpW + AffectNet), `clean-faces-rafdb` (RAF-DB) |

- Cài đặt: **GPU T4 ×1**, Internet On. `LIMIT = 300` để chạy thử nhanh, `LIMIT = 0` để chạy toàn bộ.
- Ngưỡng (CELL 9): `--min_face_size 80 --max_black_fraction 0.8 --max_landmarks_outside 0.05 --min_sharpness 30 --max_occluded_fraction 0.12 --max_glasses_fraction 0.01 --max_hat_fraction 0.02`.
  Đổi ngưỡng chỉ cần chạy lại CELL 9 (vài giây), không phải chạy lại MediaPipe / face parsing.
- Output: `clean_meta.csv` (danh sách ảnh sạch + split), `landmarks.npz` (landmark ảnh sạch, dùng làm nhãn khi train),
  `rejected.csv` (ảnh bị loại + lý do), `*_landmarks.npz` / `*_quality.csv` (toàn bộ ảnh, để lọc lại).
- Kết quả ExpW + AffectNet: **53.294 ảnh sạch** (AffectNet 35.465, ExpW 17.829), train / val / test = 46.259 / 2.487 / 4.548.

Script: `extract_mediapipe_landmarks.py`, `compute_face_quality.py`, `filter_and_split.py`.

## 03_smirk_setup: cài SMIRK trên Kaggle, demo, che khuất nhân tạo

| Notebook | Làm gì | Input | Output |
|---|---|---|---|
| `kaggle_smirk_setup.ipynb` | Clone repo (nhánh `occlusion-aware`), build PyTorch3D, tải FLAME 2020 (tài khoản trong Kaggle Secrets) và `SMIRK_em1.pt`, demo tái tạo 3D trên ảnh test, demo **SMIRK trên ảnh sạch vs ảnh bị che** | `clean-faces`, `affectnet-crops`, `expw-crops`, (tùy chọn) `pytorch3d-wheel` | `wheels/pytorch3d-*.whl` → dataset `pytorch3d-wheel` (lần sau khỏi build lại 20–30 phút) |

- Cài đặt: **GPU T4 ×1**, Internet On, Secrets `FLAME_USERNAME`, `FLAME_PASSWORD`.
- Code SMIRK được clone từ GitHub: **phải push code trước** khi chạy, nếu có sửa trong `src/`.
- Module che khuất: [`src/utils/occlusion.py`](../src/utils/occlusion.py) (che ngẫu nhiên miệng / mắt / lông mày / mũi;
  hình dạng, kích thước, màu sắc ngẫu nhiên; `occlusion_rng(tên_ảnh)` cho tập test cố định).

## 04_baseline_evaluation: đánh giá SMIRK gốc khi bị che khuất

| Notebook | Làm gì | Input | Output |
|---|---|---|---|
| `kaggle_baseline_evaluation.ipynb` | Chạy `SMIRK_em1.pt` trên **toàn bộ ảnh test**, mỗi ảnh ở 6 kịch bản: `clean`, `random` (1–2 bộ phận ngẫu nhiên), và che đúng `mouth` / `eyes` / `brows` / `nose` (cố định theo tên ảnh) | `clean-faces` (+ `clean-faces-rafdb`), `expw-crops`, `affectnet-crops` (+ `rafdb-crops`) | `baseline_smirk_em1/`: `summary.csv`, `per_image.csv`, `features.npz`, `config.json` |

- Cài đặt: **GPU T4 ×1**, Internet On, Secrets FLAME. **Không cần PyTorch3D** (không render).
- Chỉ số (script [`src/evaluation/occlusion_benchmark.py`](../src/evaluation/occlusion_benchmark.py)):
  `exp_dist` / `exp_rel` (sai lệch tham số biểu cảm so với ảnh sạch), `expr_mm` (sai lệch mesh biểu cảm, mm),
  `lmk_px` (lỗi landmark so với landmark MediaPipe của ảnh sạch, pixel trong ảnh 224),
  `exp_scale` (‖ψ bị che‖ / ‖ψ sạch‖, > 1 nghĩa là biểu cảm bị phóng đại khi bị che, kiểm chứng nhận định ở mục 4.5 của bài báo),
  `emotion_mean_class_acc` (bộ phân loại **MLP** như giao thức EMOCA/SMIRK, train trên ảnh sạch của tập train,
  8 lớp với AffectNet, 7 lớp với ExpW / RAF-DB).
- Cùng script này sẽ dùng lại để đánh giá các mô hình huấn luyện ở giai đoạn 05 (chỉ đổi `--checkpoint`).

## Lưu ý chung

- **Bản quyền:** FLAME, EMOCA, SMIRK, AffectNet, RAF-DB chỉ dùng cho nghiên cứu, không phân phối lại. Mọi notebook và
  dataset trên Kaggle để **Private**; không commit dữ liệu, mô hình hay link tải riêng vào repo (repo đang Public).
- **Tạo dataset từ output:** dùng *Save Version → Save & Run All*, rồi tạo dataset từ **trang version đã chạy xong**
  (tab Output). Tạo từ khung Output trong editor chỉ liệt kê ~500 file và hay lỗi `Unexpected end of JSON input`.
- Kaggle chỉ cho **1 phiên GPU tương tác** cùng lúc: tắt phiên cũ trong *Manage active sessions*.
