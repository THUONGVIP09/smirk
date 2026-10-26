"""
Training data for occlusion-aware SMIRK: clean faces and their pre-generated synthetic occlusions.

Every sample gives the encoder input (an occluded variant from datasets/preprocess_scripts/make_occluded_dataset.py
with probability p_occluded, otherwise the clean image) and the targets from the CLEAN image: the image itself (for the
reconstruction / emotion losses), the 105 MediaPipe landmarks used by SMIRK, the face hull mask and the MICA crop.
Clean and occluded images are pixel-aligned, so the same crop and augmentation are applied to both.

Keys (as in BaseDataset.prepare_data, plus the occlusion ones):
    img                  clean face, 3 x S x S in [0, 1]                    (targets)
    img_input            occluded face (or the clean one), 3 x S x S         (encoder input)
    occlusion_mask       1 x S x S, 1 = occluded
    landmarks_mp         105 x 2 in [-1, 1], from the clean image
    landmarks_mp_visible 105, False where the landmark is under the occluder
    landmarks_fan / flag_landmarks_fan   zeros / False (no FAN landmarks for these datasets)
    mask                 1 x S x S face hull mask, 0 inside the face (as in BaseDataset)
    img_mica             3 x 112 x 112 ArcFace crop of the clean face (MICA loss, from-scratch pretraining)
    is_occluded, occluded_area
"""
import os
import csv
import numpy as np
import cv2
import torch

from src.utils.occluded_store import OccludedStore

# indices of the MediaPipe landmarks that SMIRK uses (same as mediapipe_indices in datasets/base_dataset.py)
SMIRK_MEDIAPIPE_INDICES = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 46, 52, 53,
        55, 63, 65, 66, 70, 105, 107, 249, 263, 362, 373, 374, 380,
       381, 382, 384, 385, 386, 387, 388, 390, 398, 466, 7, 33, 133,
       144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
       168, 6, 197, 195, 5, 4, 129, 98, 97, 2, 326, 327, 358,
         0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84,
        87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
       308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
       415]

# ArcFace 5-point template (as BaseDataset.arcface_dst) and the MediaPipe points that approximate it
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
IMAGE_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]   # subject's right eye
IMAGE_RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]


def crop_matrix(landmarks, scale, size):
    """Affine matrix of the SMIRK face crop (same as BaseDataset.crop_face / demo.py)."""
    left, top = landmarks.min(axis=0)
    right, bottom = landmarks.max(axis=0)
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    s = int(old_size * scale)
    src = np.float32([[center[0] - s / 2, center[1] - s / 2], [center[0] - s / 2, center[1] + s / 2],
                      [center[0] + s / 2, center[1] - s / 2]])
    dst = np.float32([[0, 0], [0, size - 1], [size - 1, 0]])
    return cv2.getAffineTransform(src, dst)


def mica_crop(image_rgb, landmarks):
    """112x112 ArcFace-aligned crop for MICA, with the 5 points taken from the MediaPipe landmarks."""
    points = np.stack([landmarks[IMAGE_LEFT_EYE].mean(0), landmarks[IMAGE_RIGHT_EYE].mean(0),
                       landmarks[1], landmarks[61], landmarks[291]]).astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(points, ARCFACE_DST)
    if M is None:
        M = np.float32([[112 / image_rgb.shape[1], 0, 0], [0, 112 / image_rgb.shape[0], 0]])
    return cv2.warpAffine(image_rgb.astype(np.float32) / 255, M, (112, 112), borderValue=0.0)


def build_augmentation():
    """Photometric + geometric augmentation applied identically to the clean and the occluded image."""
    import albumentations as A
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.9),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        additional_targets={'occluded': 'image', 'occ_mask': 'mask'})


class OccludedFaceDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dirs, image_dirs, occluded_dirs, split, image_size=224, train_scale=(1.2, 1.8),
                 test_scale=1.6, p_occluded=0.5, max_occluded_area=0.7, augment=None):
        """
        clean_dirs:    folders with clean_meta.csv and landmarks.npz (outputs of the cleaning step)
        image_dirs:    {dataset: folder with the 256x256 crops}
        occluded_dirs: {dataset: folder with occluded_meta.csv / images.bin / masks.bin}
        split:         'train', 'val' or 'test'
        """
        self.split, self.image_size, self.p_occluded = split, image_size, p_occluded
        self.train = split == 'train'
        self.scale = train_scale if self.train else test_scale
        self.image_dirs = image_dirs
        self.stores = {name: OccludedStore(folder) for name, folder in occluded_dirs.items()}

        self.rows, self.landmarks = [], {}
        for folder in clean_dirs:
            with open(os.path.join(folder, 'clean_meta.csv')) as f:
                self.rows += [r for r in csv.DictReader(f) if r['split'] == split and r['dataset'] in image_dirs]
            data = np.load(os.path.join(folder, 'landmarks.npz'))
            self.landmarks.update(zip(data['keys'], data['landmarks']))

        # occluded variants of every face, without the nearly fully occluded ones
        self.variants = []
        for r in self.rows:
            store = self.stores.get(r['dataset'])
            keep = []
            for v in (store.variants(r['dataset'], r['name']) if store else []):
                if float(store.info(r['dataset'], r['name'], v)['occluded_area']) <= max_occluded_area:
                    keep.append(v)
            self.variants.append(keep)

        self.augment = (build_augmentation() if self.train else None) if augment is None else augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        dataset, name = r['dataset'], r['name']
        clean = np.ascontiguousarray(cv2.imread(os.path.join(self.image_dirs[dataset], name + '.png'))[..., ::-1])
        landmarks = self.landmarks[f'{dataset}/{name}'][:, :2].astype(np.float32)

        # encoder input: a random occluded variant (train), or a fixed choice (val / test: every other face occluded)
        variants = self.variants[index]
        if self.train:
            use_occluded = len(variants) > 0 and np.random.rand() < self.p_occluded
            variant = variants[np.random.randint(len(variants))] if use_occluded else None
        else:
            use_occluded = len(variants) > 0 and self.p_occluded > 0 and index % 2 == 0
            variant = variants[0] if use_occluded else None
        if use_occluded:
            occluded, occ_mask = self.stores[dataset].get(dataset, name, variant)
            area = float(self.stores[dataset].info(dataset, name, variant)['occluded_area'])
        else:
            occluded, occ_mask, area = clean, np.zeros(clean.shape[:2], np.float32), 0.0

        # SMIRK crop (random scale when training), the same for the clean and the occluded image
        scale = np.random.uniform(*self.scale) if self.train else self.scale
        M = crop_matrix(landmarks, scale, self.image_size)
        S = (self.image_size, self.image_size)
        clean_crop = cv2.warpAffine(clean, M, S, borderValue=0)
        occluded_crop = cv2.warpAffine(occluded, M, S, borderValue=0)
        occ_mask_crop = cv2.warpAffine(occ_mask, M, S, borderValue=0)
        crop_landmarks = landmarks @ M[:, :2].T + M[:, 2]

        hull = np.ones(S, np.uint8)  # 1 outside the face, as create_mask in datasets/base_dataset.py
        cv2.fillConvexPoly(hull, cv2.convexHull(crop_landmarks.astype(np.int32)), 0)
        landmarks_mp = crop_landmarks[SMIRK_MEDIAPIPE_INDICES]

        if self.augment is not None:
            out = self.augment(image=clean_crop, occluded=occluded_crop, mask=1 - hull, occ_mask=occ_mask_crop,
                               keypoints=landmarks_mp)
            clean_crop, occluded_crop = out['image'], out['occluded']
            hull, occ_mask_crop = 1 - out['mask'], out['occ_mask']
            landmarks_mp = np.array(out['keypoints'], dtype=np.float32).reshape(-1, 2)

        # landmarks under the occluder (looked up in the occlusion mask)
        xy = np.round(landmarks_mp).astype(int)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < self.image_size) & (xy[:, 1] >= 0) & (xy[:, 1] < self.image_size)
        hidden = np.zeros(len(xy), bool)
        hidden[inside] = occ_mask_crop[xy[inside, 1], xy[inside, 0]] > 0.5

        to_tensor = lambda image: torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255
        return {
            'img': to_tensor(clean_crop),
            'img_input': to_tensor(occluded_crop),
            'occlusion_mask': torch.from_numpy(np.ascontiguousarray(occ_mask_crop)).float()[None],
            'landmarks_mp': torch.from_numpy(landmarks_mp / self.image_size * 2 - 1).float(),
            'landmarks_mp_visible': torch.from_numpy(~hidden),
            'landmarks_fan': torch.zeros(68, 2),
            'flag_landmarks_fan': False,
            'mask': torch.from_numpy(np.ascontiguousarray(hull)).float()[None],
            'img_mica': torch.from_numpy(mica_crop(clean, landmarks).transpose(2, 0, 1).copy()),
            'is_occluded': use_occluded,
            'occluded_area': area,
        }
