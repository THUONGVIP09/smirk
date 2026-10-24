"""
Pre-generated synthetic occlusions of a face dataset (written by datasets/preprocess_scripts/make_occluded_dataset.py).

A store folder holds a few big files instead of tens of thousands of small ones (Kaggle datasets handle that badly):
    occluded_meta.csv   one row per occluded variant: dataset, name, variant, split, label, expression, parts, shapes,
                        fills, covered_mouth / _eyes / _brows / _nose, occluded_area, and the byte ranges below
    images.bin          the occluded images, JPEG encoded, concatenated (same 256x256 frame as the clean crops)
    masks.bin           the occlusion masks, PNG encoded (uint8, 255 = occluded), concatenated
    preview/            a few occluded images as plain files, to look at

Usage:
    store = OccludedStore('/kaggle/input/occluded-expw/occluded_expw')
    variants = store.variants('expw', '000123')          # e.g. [0, 1]
    image, mask = store.get('expw', '000123', variant=0)  # HxWx3 uint8 RGB, HxW float32 in [0, 1]
"""
import os
import csv
import numpy as np
import cv2


class OccludedStore:
    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, 'occluded_meta.csv')) as f:
            self.rows = list(csv.DictReader(f))
        self.index = {(r['dataset'], r['name'], int(r['variant'])): r for r in self.rows}
        self._variants = {}
        for r in self.rows:
            self._variants.setdefault((r['dataset'], r['name']), []).append(int(r['variant']))
        self._files = {}  # opened lazily, once per process (DataLoader workers)

    def __len__(self):
        return len(self.rows)

    def variants(self, dataset, name):
        return self._variants.get((dataset, name), [])

    def _read(self, kind, offset, length):
        key = (os.getpid(), kind)
        if key not in self._files:
            self._files[key] = open(os.path.join(self.folder, f'{kind}.bin'), 'rb')
        f = self._files[key]
        f.seek(offset)
        return np.frombuffer(f.read(length), np.uint8)

    def info(self, dataset, name, variant=0):
        return self.index[(dataset, name, variant)]

    def get(self, dataset, name, variant=0):
        r = self.index[(dataset, name, variant)]
        image = cv2.imdecode(self._read('images', int(r['image_offset']), int(r['image_length'])), cv2.IMREAD_COLOR)[..., ::-1]
        mask = cv2.imdecode(self._read('masks', int(r['mask_offset']), int(r['mask_length'])), cv2.IMREAD_GRAYSCALE)
        return np.ascontiguousarray(image), mask.astype(np.float32) / 255
