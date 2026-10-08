"""
Crop every annotated face of WFLW (https://wywu.github.io/projects/LAB/WFLW.html) into its own image.

WFLW images are full scenes (from WIDER FACE) that may contain many people, so each annotated face is cropped
around its ground-truth 98 landmarks. The landmarks are transformed into the crop and saved together with the
attribute flags, so that the crops can then be processed with apply_mediapipe_to_dataset.py / apply_fan_to_dataset.py.

Expected input layout (after extracting WFLW_images.tar.gz and WFLW_annotations.tar.gz into --wflw_dir):
    wflw_dir/WFLW_images/<event folders>/<image>.jpg
    wflw_dir/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_{train,test}.txt

Output:
    out_dir/{train,test}/images/{idx:05d}.png       cropped faces (image_size x image_size)
    out_dir/{train,test}/landmarks_98/{idx:05d}.npy  98x2 GT landmarks in crop pixel coordinates
    out_dir/{train,test}/meta.csv                    idx, source image, original bbox, attribute flags
"""
import os
import csv
import argparse
import numpy as np
import cv2
from tqdm import tqdm

ATTRIBUTES = ['pose', 'expression', 'illumination', 'make_up', 'occlusion', 'blur']


def find_annotation_file(wflw_dir, split):
    filename = f'list_98pt_rect_attr_{split}.txt'
    for root, _, files in os.walk(os.path.join(wflw_dir, 'WFLW_annotations')):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f'{filename} not found under {wflw_dir}/WFLW_annotations')


def parse_line(line):
    # 98 * (x, y) landmarks, 4 bbox values (x_min, y_min, x_max, y_max), 6 attribute flags, image name
    tokens = line.strip().split()
    assert len(tokens) == 196 + 4 + 6 + 1, f'unexpected annotation format ({len(tokens)} tokens)'

    landmarks = np.array(tokens[:196], dtype=np.float32).reshape(98, 2)
    bbox = np.array(tokens[196:200], dtype=np.float32)
    attributes = [int(x) for x in tokens[200:206]]
    image_name = tokens[206]
    return landmarks, bbox, attributes, image_name


def crop_face(image, landmarks, scale, image_size):
    # square crop centered on the landmark extent; out-of-image regions are filled with black
    left, top = landmarks.min(axis=0)
    right, bottom = landmarks.max(axis=0)
    center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
    size = max(right - left, bottom - top) * scale

    s = image_size / size
    M = np.array([[s, 0, image_size / 2.0 - s * center[0]],
                  [0, s, image_size / 2.0 - s * center[1]]], dtype=np.float32)

    cropped = cv2.warpAffine(image, M, (image_size, image_size), flags=cv2.INTER_LINEAR, borderValue=0)
    cropped_landmarks = landmarks @ M[:, :2].T + M[:, 2]
    return cropped, cropped_landmarks, size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop the annotated faces of WFLW.')
    parser.add_argument('--wflw_dir', type=str, required=True, help='Directory containing WFLW_images and WFLW_annotations')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the square output crops')
    parser.add_argument('--scale', type=float, default=2.0,
                        help='Crop size relative to the landmark extent. Must leave room for the later MediaPipe-based '
                             'crop of BaseDataset (train scale up to 1.8 w.r.t. MediaPipe landmarks, which also cover the forehead)')
    parser.add_argument('--min_face_size', type=float, default=80,
                        help='Skip faces whose landmark extent (in original pixels) is smaller than this')
    args = parser.parse_args()

    for split in ['train', 'test']:
        annotation_file = find_annotation_file(args.wflw_dir, split)
        with open(annotation_file) as f:
            lines = [line for line in f if line.strip()]

        os.makedirs(os.path.join(args.out_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, split, 'landmarks_98'), exist_ok=True)

        rows = []
        num_skipped = 0
        for idx, line in enumerate(tqdm(lines, desc=split)):
            landmarks, bbox, attributes, image_name = parse_line(line)

            face_size = max(*(landmarks.max(axis=0) - landmarks.min(axis=0)))
            if face_size < args.min_face_size:
                num_skipped += 1
                continue

            image = cv2.imread(os.path.join(args.wflw_dir, 'WFLW_images', image_name))
            if image is None:
                print(f'Could not read {image_name}')
                num_skipped += 1
                continue

            cropped, cropped_landmarks, _ = crop_face(image, landmarks, args.scale, args.image_size)

            name = f'{idx:05d}'
            cv2.imwrite(os.path.join(args.out_dir, split, 'images', name + '.png'), cropped)
            np.save(os.path.join(args.out_dir, split, 'landmarks_98', name + '.npy'), cropped_landmarks.astype(np.float32))
            rows.append([name, image_name, *bbox.tolist(), *attributes])

        with open(os.path.join(args.out_dir, split, 'meta.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'source_image', 'x_min', 'y_min', 'x_max', 'y_max'] + ATTRIBUTES)
            writer.writerows(rows)

        print(f'{split}: saved {len(rows)} faces, skipped {num_skipped} (too small or unreadable)')
