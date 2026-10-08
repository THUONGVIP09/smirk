"""
Convert a folder-structured AffectNet copy into the same layout as the ExpW crops (crop_expw.py).

Expected input layout (AffectNet copies found on Kaggle, where faces are already cropped), either
    input_dir/{train,val,test}/{0..7}/*.jpg     (split is kept)
or
    input_dir/{0..7}/*.jpg                      (split is set to 'unsplit')

The folder numbers follow the FER2013 order (same as ExpW) plus contempt:
    0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral, 7 contempt
This mapping was inferred by looking at samples and at the class sizes (disgust 3803 and contempt 3750 train images,
as in the official AffectNet), so check it visually before using the labels (see kaggle_prepare_affectnet.ipynb).

Each image is resized to --image_size x --image_size (the images are square face crops; the few non-square ones are
padded with black first) and saved as PNG.

Output:
    out_dir/images/{split}_{label}_{idx:05d}.png
    out_dir/meta.csv     name, split, label, expression, source file, original width/height
"""
import os
import csv
import argparse
import numpy as np
import cv2
from tqdm import tqdm

EXPRESSIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt']


def to_square(image):
    h, w = image.shape[:2]
    if h == w:
        return image
    size = max(h, w)
    top, left = (size - h) // 2, (size - w) // 2
    return cv2.copyMakeBorder(image, top, size - h - top, left, size - w - left, cv2.BORDER_CONSTANT, value=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert AffectNet (folder per class) to the ExpW-crops layout.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing train/ val/ test/')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the square output images')
    parser.add_argument('--min_face_size', type=float, default=160, help='Skip images smaller than this (in pixels)')
    parser.add_argument('--skip_contempt', action='store_true', help='Drop class 7 (contempt is not in ExpW)')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, 'images'), exist_ok=True)

    # some copies have split folders (input_dir/{train,val,test}/{0..7}), others only class folders (input_dir/{0..7});
    # the latter are marked 'unsplit' and split later
    splits = [s for s in ['train', 'val', 'test'] if os.path.isdir(os.path.join(args.input_dir, s))]
    sources = [(s, os.path.join(args.input_dir, s)) for s in splits] or [('unsplit', args.input_dir)]

    files = []
    for split, split_dir in sources:
        for label in range(len(EXPRESSIONS)):
            folder = os.path.join(split_dir, str(label))
            if not os.path.isdir(folder) or (args.skip_contempt and label == 7):
                continue
            for filename in sorted(os.listdir(folder)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    files.append((split, label, os.path.join(folder, filename)))

    rows = []
    skipped = {'too_small': 0, 'unreadable': 0}
    for idx, (split, label, path) in enumerate(tqdm(files)):
        image = cv2.imread(path)
        if image is None:
            skipped['unreadable'] += 1
            continue

        h, w = image.shape[:2]
        if min(h, w) < args.min_face_size:
            skipped['too_small'] += 1
            continue

        image = cv2.resize(to_square(image), (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)

        name = f'{split}_{label}_{idx:05d}'
        cv2.imwrite(os.path.join(args.out_dir, 'images', name + '.png'), image)
        rows.append([name, split, label, EXPRESSIONS[label], os.path.relpath(path, args.input_dir), w, h])

    with open(os.path.join(args.out_dir, 'meta.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'split', 'label', 'expression', 'source_file', 'orig_width', 'orig_height'])
        writer.writerows(rows)

    print(f'Saved {len(rows)} images, skipped {skipped}')
    for split, _ in sources:
        counts = np.bincount([r[2] for r in rows if r[1] == split], minlength=len(EXPRESSIONS))
        print(f'{split}:', dict(zip(EXPRESSIONS, counts.tolist())))