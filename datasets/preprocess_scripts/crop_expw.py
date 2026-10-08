"""
Crop every labeled face of ExpW (Expression in-the-Wild, http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html)
into its own image.

ExpW images are full scenes that may contain several people. label.lst has one line per face:
    image_name face_id_in_image face_box_top face_box_left face_box_right face_box_bottom face_box_cofidence expression_label
with expression labels 0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral.

Each face is cropped as a square around its box (enlarged by --scale, so that the later MediaPipe-based crop of
BaseDataset still fits), resized to --image_size and saved together with a meta.csv. The crops can then be processed
with apply_mediapipe_to_dataset.py / apply_fan_to_dataset.py.

Output:
    out_dir/images/{idx:06d}.png   cropped faces
    out_dir/meta.csv               name, source image, face id, original box, box confidence, expression label
"""
import os
import csv
import argparse
import numpy as np
import cv2
from tqdm import tqdm

EXPRESSIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def parse_line(line):
    tokens = line.strip().split()
    assert len(tokens) == 8, f'unexpected label format ({len(tokens)} tokens): {line}'

    image_name = tokens[0]
    face_id = int(tokens[1])
    top, left, right, bottom = [float(x) for x in tokens[2:6]]
    confidence = float(tokens[6])
    expression = int(tokens[7])
    return image_name, face_id, (left, top, right, bottom), confidence, expression


def crop_face(image, box, scale, image_size):
    # square crop centered on the face box; out-of-image regions are filled with black
    left, top, right, bottom = box
    center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
    size = max(right - left, bottom - top) * scale

    s = image_size / size
    M = np.array([[s, 0, image_size / 2.0 - s * center[0]],
                  [0, s, image_size / 2.0 - s * center[1]]], dtype=np.float32)

    return cv2.warpAffine(image, M, (image_size, image_size), flags=cv2.INTER_LINEAR, borderValue=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop the labeled faces of ExpW.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with the extracted ExpW images (origin/)')
    parser.add_argument('--label_file', type=str, required=True, help='Path to label.lst')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the square output crops')
    parser.add_argument('--scale', type=float, default=2.0, help='Crop size relative to the face box size')
    parser.add_argument('--min_face_size', type=float, default=160,
                        help='Skip faces whose box (in original pixels) is smaller than this; small faces are too blurry for SMIRK')
    parser.add_argument('--min_confidence', type=float, default=None, help='Optionally skip faces with a lower box confidence')
    args = parser.parse_args()

    with open(args.label_file) as f:
        lines = [line for line in f if line.strip()]

    os.makedirs(os.path.join(args.out_dir, 'images'), exist_ok=True)

    rows = []
    skipped = {'too_small': 0, 'low_confidence': 0, 'unreadable': 0}
    cached_name, cached_image = None, None  # consecutive lines usually refer to the same image

    for idx, line in enumerate(tqdm(lines)):
        image_name, face_id, box, confidence, expression = parse_line(line)
        left, top, right, bottom = box

        if max(right - left, bottom - top) < args.min_face_size:
            skipped['too_small'] += 1
            continue

        if args.min_confidence is not None and confidence < args.min_confidence:
            skipped['low_confidence'] += 1
            continue

        if image_name != cached_name:
            cached_name, cached_image = image_name, cv2.imread(os.path.join(args.image_dir, image_name))
        if cached_image is None:
            skipped['unreadable'] += 1
            continue

        cropped = crop_face(cached_image, box, args.scale, args.image_size)

        name = f'{idx:06d}'
        cv2.imwrite(os.path.join(args.out_dir, 'images', name + '.png'), cropped)
        rows.append([name, image_name, face_id, left, top, right, bottom, confidence, expression, EXPRESSIONS[expression]])

    with open(os.path.join(args.out_dir, 'meta.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'source_image', 'face_id', 'left', 'top', 'right', 'bottom', 'confidence', 'label', 'expression'])
        writer.writerows(rows)

    print(f'Saved {len(rows)} faces, skipped {skipped}')
    counts = np.bincount([r[8] for r in rows], minlength=len(EXPRESSIONS))
    print('Faces per expression:', dict(zip(EXPRESSIONS, counts.tolist())))
