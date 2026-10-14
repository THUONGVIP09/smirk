"""
Crop the labeled faces of the official RAF-DB basic emotion set (http://whdeng.cn/RAF/model1.html) into the same layout
as the ExpW crops (crop_expw.py).

Expected input (after extracting the official zips):
    Image/original/{train_XXXXX,test_XXXX}.jpg              original images (some contain several faces)
    Annotation/boundingbox/<image name>_boundingbox.txt     "x1 y1 x2 y2" of the labeled face
    EmoLabel/list_patition_label.txt                        "<image name>.jpg <label>", labels 1..7

RAF-DB labels are mapped to the label order used for ExpW / AffectNet:
    RAF 1 Surprise -> 5, 2 Fear -> 2, 3 Disgust -> 1, 4 Happiness -> 3, 5 Sadness -> 4, 6 Anger -> 0, 7 Neutral -> 6
The official train / test split (image name prefix) is kept.

Output:
    out_dir/images/{train_XXXXX,test_XXXX}.png   cropped faces (image_size x image_size)
    out_dir/meta.csv                             name, split, label, expression, raf_label, original box, face size
"""
import os
import csv
import argparse
import numpy as np
import cv2
from tqdm import tqdm

EXPRESSIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
RAF_TO_COMMON = {1: 5, 2: 2, 3: 1, 4: 3, 5: 4, 6: 0, 7: 6}


def crop_face(image, box, scale, image_size):
    # square crop centered on the face box; out-of-image regions are filled with black (same as crop_expw.py)
    left, top, right, bottom = box
    center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
    size = max(right - left, bottom - top) * scale

    s = image_size / size
    M = np.array([[s, 0, image_size / 2.0 - s * center[0]],
                  [0, s, image_size / 2.0 - s * center[1]]], dtype=np.float32)

    return cv2.warpAffine(image, M, (image_size, image_size), flags=cv2.INTER_LINEAR, borderValue=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop the labeled faces of RAF-DB (basic emotions).')
    parser.add_argument('--image_dir', type=str, required=True, help='Folder with the original images (Image/original)')
    parser.add_argument('--bbox_dir', type=str, required=True, help='Folder with the *_boundingbox.txt files')
    parser.add_argument('--label_file', type=str, required=True, help='EmoLabel/list_patition_label.txt')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--scale', type=float, default=2.0, help='Crop size relative to the face box size')
    parser.add_argument('--min_face_size', type=float, default=160, help='Skip faces whose box is smaller than this (pixels)')
    args = parser.parse_args()

    with open(args.label_file) as f:
        labels = [line.split() for line in f if line.strip()]

    os.makedirs(os.path.join(args.out_dir, 'images'), exist_ok=True)
    rows = []
    skipped = {'too_small': 0, 'unreadable': 0, 'no_box': 0}

    for image_name, raf_label in tqdm(labels):
        stem = os.path.splitext(image_name)[0]
        box_path = os.path.join(args.bbox_dir, f'{stem}_boundingbox.txt')
        if not os.path.exists(box_path):
            skipped['no_box'] += 1
            continue
        with open(box_path) as f:
            left, top, right, bottom = [float(v) for v in f.read().split()[:4]]

        if max(right - left, bottom - top) < args.min_face_size:
            skipped['too_small'] += 1
            continue

        image = cv2.imread(os.path.join(args.image_dir, image_name))
        if image is None:
            skipped['unreadable'] += 1
            continue

        cropped = crop_face(image, (left, top, right, bottom), args.scale, args.image_size)
        cv2.imwrite(os.path.join(args.out_dir, 'images', stem + '.png'), cropped)

        label = RAF_TO_COMMON[int(raf_label)]
        rows.append([stem, stem.split('_')[0], label, EXPRESSIONS[label], int(raf_label),
                     left, top, right, bottom, max(right - left, bottom - top)])

    with open(os.path.join(args.out_dir, 'meta.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'split', 'label', 'expression', 'raf_label', 'x1', 'y1', 'x2', 'y2', 'face_size'])
        writer.writerows(rows)

    print(f'Saved {len(rows)} faces, skipped {skipped}')
    for split in ['train', 'test']:
        counts = np.bincount([r[2] for r in rows if r[1] == split], minlength=len(EXPRESSIONS))
        print(f'{split}:', dict(zip(EXPRESSIONS, counts.tolist())))
