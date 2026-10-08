# Phân tích dự án SMIRK — góc nhìn kỹ sư Thị giác máy tính

> Tài liệu nội bộ để nắm dự án trước khi phát triển tiếp trên branch `occlusion-aware`.
> Các đường dẫn trong tài liệu là tương đối so với thư mục gốc của repo.

## Mục lục
1. [Kiến thức nền tảng cần nắm](#1-kiến-thức-nền-tảng-cần-nắm)
2. [Bức tranh tổng thể của SMIRK](#2-bức-tranh-tổng-thể-của-smirk)
3. [Sơ đồ phụ thuộc giữa các file](#3-sơ-đồ-phụ-thuộc-giữa-các-file)
4. [Phân tích chi tiết từng file](#4-phân-tích-chi-tiết-từng-file)
5. [Luồng dữ liệu & shape tensor](#5-luồng-dữ-liệu--shape-tensor)
6. [Dự án đang dừng ở mức nào](#6-dự-án-đang-dừng-ở-mức-nào)
7. [Lỗi / điểm yếu phát hiện trong code](#7-lỗi--điểm-yếu-phát-hiện-trong-code)
8. [Tình trạng máy của bạn & những gì cần bổ sung](#8-tình-trạng-máy-của-bạn--những-gì-cần-bổ-sung)
9. [Lộ trình đề xuất cho bước tiếp theo (occlusion-aware)](#9-lộ-trình-đề-xuất-cho-bước-tiếp-theo-occlusion-aware)

---

## 1. Kiến thức nền tảng cần nắm

### 1.1 Bài toán: Monocular 3D Face Reconstruction
Từ **một ảnh 2D**, ước lượng hình học 3D của khuôn mặt. Đây là bài toán *ill-posed*: nhiều hình 3D khác nhau có thể cho ra cùng một ảnh, do sự nhập nhằng giữa độ sâu, ánh sáng và albedo. Vì vậy người ta dùng **mô hình thống kê (3D Morphable Model – 3DMM)** làm prior. Mạng chỉ cần hồi quy ra vài trăm tham số, thay vì hàng nghìn đỉnh.

### 1.2 FLAME – mô hình đầu tham số hóa
FLAME (Li et al., 2017) là 3DMM được dùng trong dự án:
- **5023 đỉnh, 9976 tam giác** và 5 khớp (global, neck, jaw, 2 mắt).
- Công thức:
  `T(β, ψ, θ) = T̄ + B_S(β) + B_E(ψ) + B_P(θ)`, sau đó `M = LBS(T, J(β), θ, W)`
  - `β` (shape, tối đa 300): danh tính, tức hình dạng khuôn mặt của từng người.
  - `ψ` (expression, tối đa 100, SMIRK dùng **50**): biểu cảm.
  - `θ` (pose): xoay toàn cục, neck, **jaw** và 2 nhãn cầu, biểu diễn bằng axis-angle và chuyển sang ma trận xoay bằng công thức **Rodrigues**.
  - **LBS (Linear Blend Skinning)**: mỗi đỉnh được biến đổi theo tổ hợp tuyến tính của các ma trận khớp, với trọng số `W`.
- SMIRK bổ sung **eyelid blendshapes** (`assets/l_eyelid.npy`, `r_eyelid.npy`), vì bản FLAME gốc nhắm mắt kém.

### 1.3 Camera & chiếu
Dự án dùng **phép chiếu trực giao có tỉ lệ (weak-perspective)**: `x_2d = s · (x_3d[:2] + t)`, với `cam = [s, tx, ty]`. Cách này đơn giản và ổn định cho ảnh mặt đã crop, nhưng không mô hình hóa được méo phối cảnh khi chụp gần.

### 1.4 Landmark
- **FAN (68 điểm, chuẩn iBUG 300W)**: 17 điểm viền hàm và 51 điểm trong. Trên FLAME, 17 điểm viền là *dynamic* (thay đổi theo góc quay đầu).
- **MediaPipe FaceMesh (478 điểm)**: SMIRK dùng **105 điểm** có ánh xạ barycentric sang bề mặt FLAME.
- Landmark loss là tín hiệu giám sát "yếu": rẻ, nhưng không đủ để bắt được biểu cảm tinh tế (má phồng, môi cuộn...). Đó chính là động cơ ra đời của SMIRK.

### 1.5 Analysis-by-Synthesis & Differentiable Rendering
- **Analysis-by-synthesis**: ước lượng tham số, *tổng hợp* lại ảnh, rồi so với ảnh thật để lấy tín hiệu học.
- Cách truyền thống (DECA, EMOCA) dùng **differentiable rasterizer** kết hợp albedo và lighting (Spherical Harmonics) để render ảnh "giống thật". Hạn chế là albedo tuyến tính trông giả, và sai số render dễ bị "đổ" nhầm vào hình học.
- **Đóng góp của SMIRK — Analysis-by-*Neural*-Synthesis**: thay photometric renderer bằng **mạng image-to-image (UNet)**. Mạng này nhận hình học đã render dạng mesh xám, cùng **một ít pixel thật lấy mẫu thưa** trên mặt, rồi sinh lại ảnh thật. Vì generator chỉ có rất ít thông tin pixel, nó *buộc phải dựa vào hình học* để tái tạo, nên gradient từ reconstruction loss thực sự sửa được expression.

### 1.6 Các loss/kỹ thuật khác cần biết
| Kỹ thuật | Ý nghĩa |
|---|---|
| **Perceptual loss (VGG16)** | So sánh ảnh trong không gian đặc trưng của CNN, giúp ảnh sắc nét hơn so với chỉ dùng L1. |
| **Emotion/Expression loss (EMOCA ResNet50)** | So sánh đặc trưng cảm xúc giữa ảnh tái tạo và ảnh thật. |
| **ArcFace / MICA** | ArcFace cho embedding danh tính 512 chiều. MICA ánh xạ embedding đó sang shape FLAME metric, dùng làm "pseudo-GT" cho shape. |
| **Cycle consistency** | Từ tham số p' (đã tăng cường), sinh ảnh I', rồi encode I' ra p''. Yêu cầu p'' ≈ p'. Nhờ đó encoder học được cả những biểu cảm *không có trong dataset*. |
| **Convex hull mask** | Vùng mặt được lấy từ bao lồi của landmark, dùng để che mặt trước khi đưa vào generator. |
| **Barycentric sampling** | Lấy mẫu điểm ngẫu nhiên trên tam giác mesh bằng tọa độ barycentric (α, β, γ). |
| **Backface culling bằng normal** | Chỉ lấy mẫu trên các tam giác quay về phía camera (normal z < 0.05 trong hệ trục của renderer). |

---

## 2. Bức tranh tổng thể của SMIRK

```
                ┌───────────── SmirkEncoder ─────────────┐
 Ảnh 224x224 ──►│ PoseEncoder   (MobileNetV3-small) → pose(3)+cam(3)
                │ ShapeEncoder  (MobileNetV3-large) → shape(300)
                │ ExprEncoder   (MobileNetV3-large) → exp(50)+eyelid(2)+jaw(3)
                └────────────────────┬───────────────────┘
                                     ▼
                              FLAME (LBS) ──► vertices(5023), landmarks FAN/MP
                                     ▼
                     Renderer (PyTorch3D rasterizer, mesh xám + đèn)
                                     ▼
          rendered_img ─┐
                        ├─ concat (6 kênh) ─► SmirkGenerator (UNet) ─► ảnh tái tạo
 masked_img ────────────┘   (ảnh gốc bị che vùng mặt + ~1% pixel mặt được giữ lại)
```

**Hai giai đoạn huấn luyện:**
1. **Pretrain** (`configs/config_pretrain.yaml`): train cả 3 encoder bằng landmark loss, MICA shape loss và expression regularization. Không dùng generator.
2. **Train chính** (`configs/config_train.yaml`): đóng băng pose và shape, chỉ train **expression encoder cùng generator**, qua 2 path:
   - **Path 1 – Reconstruction**: tái tạo lại chính ảnh đầu vào.
   - **Path 2 – Cycle**: tăng cường expression, sinh ảnh mới, encode lại và tính cycle loss.

---

## 3. Sơ đồ phụ thuộc giữa các file

```
train.py
 ├── datasets/data_utils.py  (load_dataloaders)
 │     ├── datasets/lrs3_dataset.py ─┐
 │     ├── datasets/mead_dataset.py  │
 │     ├── datasets/mead_sides_dataset.py ├──► datasets/base_dataset.py (BaseDataset, create_mask)
 │     ├── datasets/ffhq_dataset.py  │
 │     ├── datasets/celeba_dataset.py┘  (đọc datasets/identity_CelebA.txt)
 │     └── datasets/mixed_dataset_sampler.py
 └── src/smirk_trainer.py (SmirkTrainer)
       ├── src/base_trainer.py (BaseTrainer: optimizer, save/load, visualize)
       │     ├── src/losses/VGGPerceptualLoss.py
       │     ├── src/losses/ExpressionLoss.py ──► src/losses/resnet.py  (+ assets/ResNet50 của EMOCA)
       │     ├── src/models/MICA/mica.py ──► src/models/MICA/arcface.py (+ assets/mica.tar)
       │     └── src/utils/utils.py (vẽ landmark, make grid)
       ├── src/smirk_encoder.py (timm backbones)
       ├── src/smirk_generator.py
       ├── src/FLAME/FLAME.py ──► src/FLAME/lbs.py
       │     (assets/FLAME2020/generic_model.pkl, landmark_embedding.npy,
       │      mediapipe_landmark_embedding.npz, l/r_eyelid.npy)
       ├── src/renderer/renderer.py ──► src/renderer/util.py, pytorch3d
       │     (assets/head_template.obj, FLAME_masks/FLAME_masks.pkl)
       ├── src/utils/masking.py ──► renderer/util.py, FLAME/lbs.py
       │     (assets/FLAME_masks/FLAME_masks_triangles.npy)
       └── src/utils/utils.py (load_templates → assets/expression_templates_famos)

demo.py / demo_video.py
 ├── utils/mediapipe_utils.py (assets/face_landmarker.task)
 ├── src/smirk_encoder.py, src/smirk_generator.py
 ├── src/FLAME/FLAME.py, src/renderer/renderer.py
 ├── src/utils/masking.py
 └── datasets/base_dataset.py (create_mask)

datasets/preprocess_scripts/apply_mediapipe_to_dataset.py  → sinh landmark MediaPipe (.npy)
datasets/preprocess_scripts/apply_fan_to_dataset.py        → sinh landmark FAN (.npy) bằng ibug face_alignment
quick_install.sh → tải FLAME, face_landmarker, checkpoint, templates, EMOCA, MICA
```

---

## 4. Phân tích chi tiết từng file

### 4.1 Entry points

#### [train.py](../train.py)
- `parse_args()`: đọc YAML bằng OmegaConf, rồi cho phép ghi đè từ CLI (ví dụ `train.lr=1e-4`). Vì dùng `set_struct(True)`, chỉ ghi đè được các key **đã khai báo** trong YAML.
- Tạo `log_path/train_images` và `val_images`, đồng thời lưu lại config.
- Vòng lặp: **mỗi epoch gọi lại `configure_optimizers`** (tức là reset cosine scheduler theo epoch). Sau đó chạy phase `train` rồi `val`. Mỗi batch gọi `set_freeze_status`, rồi `trainer.step`. Cứ `visualize_every` batch thì lưu ảnh lưới một lần. Cứ `save_every` epoch thì lưu checkpoint `model_{epoch}.pt`.
- `create_base_encoder()`: sao chép encoder sau khi load. Bản sao này dùng để regularize so với model gốc (nếu bật) và để so sánh khi visualize.

#### [demo.py](../demo.py)
1. Chạy MediaPipe (`utils/mediapipe_utils.run_mediapipe`) để lấy 478 điểm.
2. `crop_face` (scale 1.4) ước lượng *similarity transform*, rồi `warp` ra ảnh 224×224.
3. Chạy encoder, FLAME và renderer để có `rendered_img`.
4. `--render_orig`: warp ngược ảnh render về kích thước ảnh gốc.
5. `--use_smirk_generator`: tạo hull mask, lấy mẫu điểm trên mesh (`mask_ratio 0.01×5`, chọn ngẫu nhiên số điểm thực dùng `rbound`), dựng `masked_img` rồi cho qua generator.
6. Lưu ảnh lưới [gốc | render | (tái tạo)].

#### [demo_video.py](../demo_video.py)
Giống `demo.py` nhưng chạy từng frame và ghi ra mp4. **Không có làm mượt theo thời gian**. Nếu một frame không phát hiện được mặt thì chương trình `exit()` luôn.

### 4.2 Mô hình

#### [src/smirk_encoder.py](../src/smirk_encoder.py)
- `create_backbone`: gọi `timm.create_model(..., features_only=True)` và lấy feature map cuối, sau đó global average pool.
- `PoseEncoder` (`tf_mobilenetv3_small_minimal_100`) có đầu ra Linear 6 chiều: `pose_params[:3]` và `cam[3:]`. Khởi tạo bias `cam_s = 7` để mặt ban đầu có kích thước hợp lý, các trọng số còn lại gần 0, tương ứng mặt nhìn thẳng.
- `ShapeEncoder` (`large`): Linear 300 chiều, khởi tạo bằng 0, tức bắt đầu từ mặt trung bình.
- `ExpressionEncoder` (`large`): Linear `n_exp + 5` chiều:
  - `expression_params`: 50 chiều, không ràng buộc.
  - `eyelid_params`: clamp trong [0, 1].
  - `jaw_params`: `[ReLU(mở hàm), clamp(±0.2), clamp(±0.2)]`. Hàm chỉ mở xuống và chỉ lệch ngang nhẹ, là ràng buộc giải phẫu.
- `SmirkEncoder`: gộp 3 dict đầu ra lại. **Mỗi nhánh có backbone riêng**, nên có thể đóng băng từng nhánh độc lập.
- Lưu ý: file config có khai báo `arch.backbone_*` nhưng **code đang hard-code tên backbone**, nên đổi config sẽ không có tác dụng.

#### [src/smirk_generator.py](../src/smirk_generator.py)
- UNet 4 tầng (`init_features=32` khi dùng, tức 32→64→128→256, bottleneck 512) cộng thêm **5 ResnetBlock** ở bottleneck (reflect padding, BatchNorm).
- Đầu vào 6 kênh = `rendered_img (3) ‖ masked_img (3)`. Đầu ra 3 kênh qua `sigmoid`, nằm trong [0,1].
- Có đoạn code bị comment về `use_mask`: tác giả từng thử trộn ảnh đầu ra với nền.

#### [src/FLAME/FLAME.py](../src/FLAME/FLAME.py) & [src/FLAME/lbs.py](../src/FLAME/lbs.py)
- Đầu file có monkey-patch `np.bool = np.bool_ ...` để `chumpy` chạy được với numpy mới.
- `__init__`: đọc `generic_model.pkl`. **Cắt `shapedirs` thành `[:n_shape]` và `[300:300+n_exp]`**. Nạp posedirs, J_regressor, lbs_weights, eyelid, landmark embedding (static, dynamic, full 68 3D, và MediaPipe 105).
- `forward(param_dict, zero_expression, zero_shape, zero_pose)` ([FLAME.py:232](../src/FLAME/FLAME.py#L232)):
  - Pad expression/shape bằng 0 nếu thiếu chiều.
  - `full_pose = [global(3), neck(3), jaw(3), eyes(6)]`. Neck và mắt luôn bằng 0.
  - Gọi `lbs(...)` để ra vertices, rồi **cộng offset eyelid** (tuyến tính theo `eyelid_params`).
  - Landmark: 17 điểm viền *dynamic*, chọn theo góc yaw bằng `_find_dynamic_lmk_idx_and_bcoords`, cộng 51 điểm static, được `landmarks_fan` (68). Ngoài ra có `landmarks_fan_3d` và `landmarks_mp` (105).
  - `zero_pose` đặt pose cố định `(0.2, -0.7, 0)` để visualize mặt nghiêng.
- `lbs.py`: code chuẩn của SMPL/FLAME gồm `blend_shapes`, `batch_rodrigues`, `batch_rigid_transform`, `vertices2landmarks` (nội suy barycentric).

#### [src/renderer/renderer.py](../src/renderer/renderer.py) & [src/renderer/util.py](../src/renderer/util.py)
- Đọc `head_template.obj` để lấy UV (không dùng texture). Màu xám cố định 180/255.
- `render_full_head=False`: chỉ giữ vùng `flame_masks['face']`, và `keep_vertices_and_update_faces` đánh lại chỉ số tam giác.
- `forward(vertices, cam, **landmarks)`: `batch_orth_proj` ([util.py:64](../src/renderer/util.py#L64)), **lật trục y, z** cho khớp hệ tọa độ ảnh, chiếu cả landmark, rồi render.
- `render`: rasterize bằng PyTorch3D, **5 đèn định hướng cố định**, shading Lambert. Ảnh ra có nền = 0.
- Có sẵn code SH lighting và render_multiface nhưng không dùng tới.

#### [src/models/MICA/mica.py](../src/models/MICA/mica.py), [arcface.py](../src/models/MICA/arcface.py)
- ArcFace IResNet100 cho embedding 512 chiều (chuẩn hóa L2). `MappingNetwork` (3 lớp ẩn) ra 300 shape FLAME.
- Đầu vào: ảnh 112×112 đã căn theo 5 điểm ArcFace (tạo trong `BaseDataset.prepare_data`), đổi sang BGR và chuẩn hóa về [-1,1].
- `calculate_mica_shape_loss` = MSE giữa shape dự đoán và shape của MICA. **Chỉ dùng ở pretrain.**

### 4.3 Loss

#### [src/losses/VGGPerceptualLoss.py](../src/losses/VGGPerceptualLoss.py)
L1 trên 4 block đặc trưng VGG16 (relu1_2, relu2_2, relu3_3, relu4_3). Code giả định input nằm trong [-1,1] (`x*0.5+0.5`) trong khi ảnh thực tế nằm trong [0,1]. Xem [mục 7](#7-lỗi--điểm-yếu-phát-hiện-trong-code).

#### [src/losses/ExpressionLoss.py](../src/losses/ExpressionLoss.py) & [resnet.py](../src/losses/resnet.py)
Backbone ResNet50 emotion của EMOCA, bỏ lớp fc/linear, so đặc trưng (l2/l1/cos). Bị hard-code `.cuda()`.

### 4.4 Tiện ích

#### [src/utils/masking.py](../src/utils/masking.py) — *trái tim của "neural synthesis"*
| Hàm | Chức năng |
|---|---|
| `load_probabilities_per_FLAME_triangle` | Gán trọng số lấy mẫu cho từng vùng của 9976 tam giác. Mắt, trán và mặt được 1.0, môi và mũi 0.5, tai, cổ và nhãn cầu 0. |
| `mesh_based_mask_uniform_faces` | Chỉ giữ tam giác quay về camera (normal z < 0.05), nhân thêm với **diện tích chiếu xy**, lấy mẫu `multinomial` được `mask_ratio·224²` điểm (≈ 501 điểm khi ratio 0.01), cộng barycentric ngẫu nhiên, rồi đổi sang tọa độ pixel. Có thể truyền `coords` để **dùng lại đúng các điểm trên mesh** cho một mesh khác. Đây là cơ sở của path 2. |
| `transfer_pixels(img, p1, p2)` | Chép pixel tại vị trí `p1` (mesh gốc) sang `p2` (mesh đã biến dạng), tức là *warp thưa theo mesh*. |
| `masking` | Dilate hull mask (`max_pool` với bán kính `wr`), trừ thêm vùng mesh đã render, nhân với ảnh, rồi điền các điểm thưa. Các điểm này có nhiễu nhân (σ = 0.05) và có thể bị khoét thêm các ô 11×11 ngẫu nhiên. |
| `random_barycentric`, `triangle_area`, `point2ind` | Hàm phụ trợ. |

#### [src/utils/utils.py](../src/utils/utils.py)
`load_templates` nạp các expression template từ FaMoS (12 lớp như `kissing`, `blow_cheeks`, `high_smile`...) để tiêm vào path 2. Ngoài ra có các hàm freeze/unfreeze, vẽ landmark và dựng lưới ảnh.

#### [utils/mediapipe_utils.py](../utils/mediapipe_utils.py)
Khởi tạo `FaceLandmarker` **ngay khi import** (cần `assets/face_landmarker.task`), `num_faces=1`, ngưỡng confidence 0.1. Trả về mảng (478, 3) theo pixel, hoặc `None`.

### 4.5 Trainer

#### [src/base_trainer.py](../src/base_trainer.py)
- `configure_optimizers`: Adam cho encoder với `lr × 0.25`, chỉ gồm các nhánh có `optimize_*=True`. Adam cho generator với `lr` và `betas=(0.5, 0.999)`. Cả hai dùng **CosineAnnealing theo iteration trong 1 epoch**.
- `setup_losses`: VGG luôn được tạo. Emotion và MICA chỉ được tạo khi trọng số > 0.
- `save_model`: chỉ lưu các key `smirk_encoder.*` và `smirk_generator.*`. `load_model` dùng `strict=False`.
- `set_freeze_status` ([base_trainer.py:258](../src/base_trainer.py#L258)): **batch chẵn đóng băng encoder, batch lẻ đóng băng generator** trong path 2.
- `create_visualizations` / `save_visualizations`: lưới ảnh gồm ảnh gốc kèm landmark (xanh lá = dự đoán MP, đỏ = GT MP, tím = dự đoán FAN, trắng = GT FAN), MICA, base, render, overlay, masked, reconstructed, loss map và 2nd path.

#### [src/smirk_trainer.py](../src/smirk_trainer.py)
**`step1` – Path 1 (reconstruction)** ([L34](../src/smirk_trainer.py#L34)):
- Chạy encoder, FLAME và renderer.
- Landmark loss:
  - `landmark_loss_fan`: **chỉ dùng 17 điểm viền hàm** (FAN tốt ở viền, còn phần trong mặt đã có MediaPipe lo). Bỏ qua mẫu không có FAN, ví dụ MEAD sides.
  - `landmark_loss_mp`: 105 điểm.
- Regularization: expression, shape và jaw được kéo về 0, hoặc về base model nếu `use_base_model_for_regularization`.
- Nếu bật generator: tính `rendered_mask`, lấy mẫu điểm, `transfer_pixels(img, p, p)`, `masking`, rồi generator. Loss gồm L1, VGG và **emotion loss**. Riêng emotion loss được tính trên một lần forward generator ở chế độ *đóng băng + eval*, để gradient cảm xúc **chỉ chảy về encoder**.
- Tổng: `shape + expression + landmark + generator losses` (có bật/tắt theo `optimize_*`).

**`step2` – Path 2 (cycle)** ([L184](../src/smirk_trainer.py#L184)):
- Nhân bản batch `Ke` lần, rồi chia ngẫu nhiên thành **4 nhóm** tăng cường:
  1. **Random**: cộng nhiễu Gaussian (hệ số 1–3) vào khoảng 50% số chiều, clamp ±4.
  2. **Permutation**: hoán đổi expression giữa các mẫu trong batch, nhân hệ số 0.25–1.5.
  3. **Template injection**: gán template FaMoS nhân hệ số 0.25–1.5.
  4. **Zero expression**: expression = 0 cộng nhiễu nhỏ, jaw = 0, eyelid ngẫu nhiên.
- Với mọi nhóm: jaw được cộng nhiễu (với xác suất 50%) và clamp độ mở hàm trong [0, 0.5]. Eyelid cộng nhiễu ±0.25.
- Render mesh gốc và mesh mới. Lấy mẫu điểm trên mesh gốc, **dùng cùng tọa độ barycentric trên mesh mới**, rồi `transfer_pixels` để chuyển pixel thật sang vị trí mới.
- Generator sinh ảnh "khuôn mặt thật mang biểu cảm mới". Ảnh này đi qua encoder để ra `recon_feats`.
- `cycle_loss = MSE(exp) + 10·MSE(jaw) + 10·MSE(eyelid) [+ MSE(shape) nếu generator đang train]`.

**`step`** ([L349](../src/smirk_trainer.py#L349)): chạy path 1 rồi backward và step. Nếu `cycle_loss > 0`, chạy tiếp path 2 với chế độ freeze xen kẽ, clip grad generator 0.1, rồi backward và step. Cuối cùng log và bước scheduler.

### 4.6 Dữ liệu

#### [datasets/base_dataset.py](../datasets/base_dataset.py)
- `create_mask` ([L9](../datasets/base_dataset.py#L9)): bao lồi của 478 điểm MediaPipe. Giá trị **1 = ngoài mặt, 0 = trong mặt**.
- `__getitem__`: lặp lại, chọn index ngẫu nhiên mới nếu lỗi hoặc thiếu FAN 68 điểm.
- `prepare_data` ([L124](../datasets/base_dataset.py#L124)):
  1. Chuyển BGR sang RGB. Crop theo MediaPipe với scale ngẫu nhiên trong [1.2, 1.8] khi train, cố định 1.6 khi test.
  2. Biến đổi FAN và MP theo `tform`, tạo hull mask, chọn **105 điểm MP** (`mediapipe_indices`).
  3. Augment bằng albumentations: màu sắc, blur, noise và **ShiftScaleRotate**, áp đồng thời lên ảnh, mask và 2 bộ keypoint.
  4. Chuẩn hóa landmark về [-1, 1].
  5. Tạo `img_mica` 112×112 bằng căn chỉnh ArcFace 5 điểm từ FAN.
  - Trả về dict: `img, landmarks_fan, flag_landmarks_fan, landmarks_mp, mask, img_mica`.

| File | Dataset | Ghi chú |
|---|---|---|
| [lrs3_dataset.py](../datasets/lrs3_dataset.py) | LRS3 (video nói) | Lấy frame ngẫu nhiên. Landmark FAN dạng `.pkl` có nội suy. Split được cache vào `assets/LRS3_lists.pkl`. **Dataset đã bị gỡ khỏi web.** |
| [mead_dataset.py](../datasets/mead_dataset.py) | MEAD front | Split theo subject giống paper (37/5/5). |
| [mead_sides_dataset.py](../datasets/mead_sides_dataset.py) | MEAD góc 30°/60° | **Không có FAN** (`landmarks_fan=None`), chỉ có MP. Cache vào `assets/MEAD_lists.pkl`. |
| [ffhq_dataset.py](../datasets/ffhq_dataset.py) | FFHQ256 | Chỉ dùng cho train. |
| [celeba_dataset.py](../datasets/celeba_dataset.py) | CelebA aligned | Nhóm ảnh theo identity (`identity_CelebA.txt`), mỗi lần chọn ngẫu nhiên 1 ảnh của 1 người. |
| [mixed_dataset_sampler.py](../datasets/mixed_dataset_sampler.py) | — | Mỗi batch trộn theo tỉ lệ cố định (LRS3 0.2, MEAD 0.1, FFHQ 0.3, CelebA 0.3, MEAD sides 0.1). Một epoch có `samples_per_epoch=50000`, lấy mẫu có hoàn lại. |
| [data_utils.py](../datasets/data_utils.py) | — | `load_dataloaders`: tập train trộn 5 dataset, tập val chỉ gồm LRS3 và MEAD. Có hàm nội suy landmark cho frame bị thiếu và tạo list LRS3. |

#### [datasets/preprocess_scripts/](../datasets/preprocess_scripts/)
- `apply_mediapipe_to_dataset.py`: chạy đa tiến trình, xử lý được cả ảnh và video, lưu `.npy` (478×3) hoặc (T×478×3).
- `apply_fan_to_dataset.py`: RetinaFace kết hợp FAN (`2dfan2_alt`) của [ibug face_alignment](https://github.com/hhj1897/face_alignment). **Chỉ xử lý ảnh** (dùng `cv2.imread`), nên video (LRS3/MEAD) phải có landmark `.pkl` từ nguồn khác.

### 4.7 Config
| Key quan trọng | Pretrain | Train |
|---|---|---|
| `lr` | 2e-4 | 1e-3 (encoder thực tế ×0.25) |
| `num_epochs` | 300 | 50 |
| `optimize_pose/shape/expression` | T/T/T | F/F/T |
| `enable_fuse_generator` | False | True |
| `landmark_loss` | 100 | 100 |
| `mica_loss` | 10 | 0 |
| `reconstruction / vgg` | 0 / 0 | 10 / 10 |
| `emotion_loss` | 0 | 0 (README khuyến nghị đặt **1.0**) |
| `cycle_loss` | 0 | 1.0 |
| `mask_ratio`, `mask_dilation_radius` | — | 0.01, 10 |

### 4.8 Assets
| Có sẵn trong repo | Phải tải (qua `quick_install.sh`) |
|---|---|
| `FLAME_masks/` (vùng mặt), `head_template.obj`, `l/r_eyelid.npy`, `landmark_embedding.npy`, `mediapipe_landmark_embedding/` | `FLAME2020/generic_model.pkl` (cần tài khoản), `face_landmarker.task`, `pretrained_models/SMIRK_em1.pt`, `expression_templates_famos/`, `ResNet50/` (EMOCA, cần tài khoản), `mica.tar` |

---

## 5. Luồng dữ liệu & shape tensor

| Tensor | Shape | Miền giá trị |
|---|---|---|
| `batch['img']` | B×3×224×224 | [0,1] RGB |
| `batch['mask']` | B×1×224×224 | 1 = ngoài mặt |
| `batch['landmarks_fan']` | B×68×2 | [-1,1] |
| `batch['landmarks_mp']` | B×105×2 | [-1,1] |
| `batch['img_mica']` | B×3×112×112 | [0,1] |
| `pose_params` / `cam` | B×3 / B×3 | cam = [s, tx, ty] |
| `shape_params` | B×300 | |
| `expression_params` | B×50 | |
| `jaw_params` / `eyelid_params` | B×3 / B×2 | |
| `vertices` | B×5023×3 | không gian FLAME (mét) |
| `transformed_vertices` | B×5023×3 | NDC [-1,1] |
| `rendered_img` | B×3×224×224 | nền = 0 |
| `npoints` | B×N×2 (N ≈ 501) | pixel |
| đầu vào generator | B×6×224×224 | |

---

## 6. Dự án đang dừng ở mức nào

### 6.1 Những gì tác giả đã công bố
- **Code chính thức của bài CVPR 2024**. Commit cuối cùng là **2024-05-31**, sau đó không còn cập nhật. Branch `occlusion-aware` hiện **trùng hoàn toàn với `main`**, tức chưa có thay đổi nào của bạn.
- **Đã train xong và phát hành checkpoint `SMIRK_em1.pt`** (Google Drive, tải qua `quick_install.sh`). Checkpoint gồm **encoder + generator**, train với `emotion_loss = 1.0` (hậu tố `em1`). Đây là kết quả tương ứng với paper.
- Có đủ code để **suy luận** (ảnh/video) và **huấn luyện 2 giai đoạn**.

### 6.2 Những gì **KHÔNG có** trong repo
| Thiếu | Hệ quả |
|---|---|
| Checkpoint sau pretrain (`first_stage_pretrained_encoder.pt`) | Muốn train lại phải tự chạy pretrain, **hoặc** fine-tune tiếp từ `SMIRK_em1.pt` (khuyến nghị). |
| Code đánh giá / benchmark (paper đánh giá bằng emotion recognition, lip-reading trên ảnh tái tạo và user study) | Chưa có cách đo định lượng xem cải tiến của bạn có tốt hơn hay không. **Đây là việc phải làm đầu tiên.** |
| Validation có metric | Vòng val chỉ in loss, không chọn best model. |
| Landmark FAN cho video | Script FAN chỉ chạy trên ảnh. |
| Xử lý **che khuất** (tay, kính, khẩu trang, tóc) | Hull mask giả định toàn bộ mặt đều nhìn thấy. Pixel của vật che vẫn bị lấy mẫu và vẫn bị ép tái tạo. → **Khoảng trống cho branch này.** |
| Tính nhất quán theo thời gian (temporal) | Chạy video từng frame nên bị rung. Config có `K` nhưng không dùng. |
| Texture / albedo / lighting | Chỉ ra hình học, không có màu da. |
| Logging (wandb/tensorboard), multi-GPU, AMP | `use_wandb` được khai báo nhưng không dùng. |

---

## 7. Lỗi / điểm yếu phát hiện trong code

| # | Vị trí | Vấn đề | Mức độ |
|---|---|---|---|
| 1 | [base_trainer.py:58](../src/base_trainer.py#L58) | Kiểm tra `hasattr(self, 'fuse_generator_optimizer')`, trong khi thuộc tính thật tên là `smirk_generator_optimizer`. Hậu quả: **optimizer của generator bị tạo lại mỗi epoch, mất moment Adam.** | Trung bình |
| 2 | [apply_mediapipe_to_dataset.py:69-72](../datasets/preprocess_scripts/apply_mediapipe_to_dataset.py#L69-L72) | Với video, frame không phát hiện được mặt sẽ **bị bỏ qua** thay vì chèn giá trị rỗng. Mảng landmark vì vậy lệch chỉ số so với frame (`mediapipe_landmarks[frame_idx]` bị sai hoặc tràn chỉ số). | **Cao** nếu dữ liệu có frame mất mặt |
| 3 | [VGGPerceptualLoss.py:24](../src/losses/VGGPerceptualLoss.py#L24) | Giả định input trong [-1,1], nhưng ảnh thực tế trong [0,1], nên chuẩn hóa ImageNet bị lệch. Loss vẫn hoạt động nhưng không đúng chuẩn. | Thấp |
| 4 | [smirk_trainer.py:41](../src/smirk_trainer.py#L41) & [L66](../src/smirk_trainer.py#L66) | `base_encoder` bị chạy 2 lần mỗi step, tốn tính toán vô ích. | Thấp |
| 5 | [smirk_trainer.py:79](../src/smirk_trainer.py#L79) vs [L290](../src/smirk_trainer.py#L290) | Hai công thức `rendered_mask` khác nhau: `any>0` ở path 1, `all>0` ở path 2. Khác biệt nằm ở pixel viền. | Thấp |
| 6 | [lrs3_dataset.py:95](../datasets/lrs3_dataset.py#L95) | `get_LRS3_test` truyền tham số `sample_full_video_for_testing` mà `LRS3Dataset` không nhận, nên gọi là lỗi ngay. | Thấp (hàm không được dùng) |
| 7 | `datasets/*` dùng `np.random` | Nếu PyTorch DataLoader fork worker mà không đặt `worker_init_fn`, các worker **dùng chung trạng thái NumPy RNG**. Khi đó frame_idx và scale crop có thể trùng nhau giữa các worker. | Trung bình |
| 8 | [ExpressionLoss.py:30](../src/losses/ExpressionLoss.py#L30) | Hard-code `.cuda()`, nên không chạy được trên CPU hoặc GPU khác `cuda:0`. | Thấp |
| 9 | `smirk_encoder.py` | Tên backbone bị hard-code, nên `arch.backbone_*` trong config vô tác dụng. | Thấp |
| 10 | `demo_video.py` | Frame mất mặt thì chương trình `exit()`. Không có làm mượt landmark hay tham số theo thời gian. | Trung bình (khi dùng thực tế) |

---

## 8. Tình trạng máy của bạn & những gì cần bổ sung

**Kiểm tra hiện tại:**
- GPU: **NVIDIA GTX 1650 – 4 GB VRAM**, driver 595.97 (WSL2).
- Conda env `smirk`: có Python 3.9.25 nhưng **chưa cài torch, numpy, pytorch3d...**
- Ổ đĩa trống ~938 GB (đủ cho dataset).
- **Chưa tải asset nào** (FLAME, face_landmarker, checkpoint...).

**Đánh giá phần cứng:**
- **Suy luận (demo)**: chạy được trên 1650.
- **Train đúng cấu hình paper** (batch 32, gồm 3 MobileNetV3, UNet 32-feature, VGG16, EMOCA ResNet50 và cycle path nhân đôi batch): **không vừa 4 GB**. Có 3 lựa chọn:
  - (a) giảm `batch_size` xuống 4–8 kèm gradient accumulation, AMP, và tắt emotion loss khi debug;
  - (b) thuê GPU ≥ 24 GB (A5000/3090/4090/A100) cho các lần train thật;
  - (c) chỉ fine-tune ngắn từ `SMIRK_em1.pt`.

**Danh sách cần bổ sung (checklist):**
- [ ] Cài môi trường: `pip install -r requirements.txt`, sau đó cài pytorch3d bản wheel `py39_cu117_pyt201`. Có thể phải build từ source nếu wheel không khớp.
- [ ] Đăng ký tài khoản **FLAME** (flame.is.tue.mpg.de) và **EMOCA** (emoca.is.tue.mpg.de), rồi chạy `bash quick_install.sh`.
- [ ] Chạy `demo.py` với `samples/test_image2.png` để xác nhận môi trường hoạt động.
- [ ] Dữ liệu: FFHQ256 và CelebA dễ lấy. MEAD phải gửi yêu cầu. LRS3 đã bị gỡ, thay bằng LRS2 hoặc VoxCeleb2. Sau đó chạy tiền xử lý MediaPipe + FAN, và **sửa lỗi #2** trước khi xử lý video.
- [ ] Cập nhật đường dẫn `dataset.*` trong config. Có thể đặt `percentage = 0` cho những dataset chưa có. Lưu ý: vẫn cần sửa `load_dataloaders` để bỏ qua dataset rỗng, vì hiện tại code luôn khởi tạo cả 5 dataset.
- [ ] **Bộ đánh giá** (cần có trước khi cải tiến), ví dụ:
  - NME landmark trên tập test MEAD và trên tập **bị che khuất** (COFW, hoặc che nhân tạo);
  - độ nhất quán expression giữa ảnh sạch và ảnh bị che của cùng một khuôn mặt (MSE tham số, khoảng cách đỉnh);
  - emotion recognition (EMOCA/AffectNet) trên ảnh tái tạo;
  - so sánh định tính với checkpoint gốc.

---

## 9. Lộ trình đề xuất cho bước tiếp theo (occlusion-aware)

Tên branch cho thấy mục tiêu là **SMIRK chịu được che khuất**. Hiện có 3 điểm trong pipeline bị che khuất làm hỏng:

1. **Encoder**: chưa từng thấy ảnh bị che trong lúc train, nên khi mặt bị che, expression dự đoán dễ bị "kéo" theo vật che.
2. **Reconstruction loss (path 1)**: bắt generator tái tạo cả vật che, nên gradient đẩy hình học sai về phía vật che.
3. **Lấy mẫu pixel (`mesh_based_mask_uniform_faces` + `transfer_pixels`)**: pixel của tay hoặc kính bị lấy như thể là da. Ở path 2, các pixel này còn bị warp sang vị trí mới.

**Kế hoạch đề xuất:**

| Giai đoạn | Nội dung | File chính cần sửa/thêm |
|---|---|---|
| **0. Baseline** | Cài môi trường, chạy demo, xây bộ đánh giá occlusion (mục 8) và đo `SMIRK_em1.pt` gốc. | `demo.py`, thêm `eval/` |
| **1. Mask che khuất** | Lấy mặt nạ "da nhìn thấy được" bằng một mô hình face parsing hoặc occlusion segmentation (BiSeNet face-parsing, FaRL, hoặc model occlusion-seg). Có thể tính offline khi tiền xử lý, hoặc online. Thêm `batch['visible_mask']`. | `datasets/base_dataset.py`, `preprocess_scripts/` |
| **2. Tăng cường che khuất nhân tạo** | Dán tay, kính, khẩu trang, vật thể lên ảnh (theo kiểu *copy-paste occlusion*, ví dụ bộ dữ liệu occluder của Voo et al. 2022), đồng thời cập nhật `visible_mask`. Encoder nhận ảnh bị che, **còn loss tính so với ảnh sạch hoặc chỉ tính trên vùng nhìn thấy**. | `base_dataset.py` (albumentations/custom) |
| **3. Loss nhận biết che khuất** | Nhân `visible_mask` vào L1/VGG ở path 1. Trọng số landmark theo độ nhìn thấy (điểm nằm trong vùng bị che thì weight = 0). | `smirk_trainer.step1` |
| **4. Lấy mẫu nhận biết che khuất** | Đặt `face_probabilities = 0` cho điểm rơi vào vùng bị che, hoặc lọc `npoints` theo `visible_mask` trước `transfer_pixels`. | `src/utils/masking.py`, `smirk_trainer.step1/step2` |
| **5. Cycle có che khuất** | Ở path 2, đưa vật che vào ảnh do generator sinh ra trước khi encode lại. Encoder buộc phải ra đúng `flame_feats` dù bị che. Đây là tín hiệu tự giám sát mạnh nhất, không cần nhãn. | `smirk_trainer.step2` |
| **6. Consistency loss** | Cùng một ảnh, bản sạch và bản bị che phải cho ra expression giống nhau (teacher là `base_encoder` hoặc EMA trên ảnh sạch, student nhận ảnh bị che). | `smirk_trainer`, `base_trainer.create_base_encoder` |
| **7. Fine-tune** | Resume từ `SMIRK_em1.pt` (`resume=... load_encoder=True load_fuse_generator=True`), bỏ qua pretrain. Train ngắn rồi so sánh với baseline. | `configs/` (thêm `config_occlusion.yaml`) |

**Nên sửa trước khi làm các bước trên**: lỗi #1 (optimizer generator), #2 (lệch chỉ số landmark video) và #7 (seed worker), vì chúng ảnh hưởng trực tiếp đến kết quả train.
