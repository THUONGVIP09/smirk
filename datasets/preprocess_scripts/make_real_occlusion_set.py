"""
Test set of REAL occlusions (glasses, hats, hands / objects) from the faces that the cleaning step rejected.

filter_and_split.py rejects a face as soon as one check fails; here a face is taken when it passes every quality check
(detected, not cut by the border, size, black border, sharpness) and fails only the occlusion checks of the face
parsing (glasses_fraction, hat_fraction, occluded_fraction above the cleaning thresholds). These faces keep their human
expression labels, were never used for training (rejected), and give a test of the emotion probe under occlusions that
the synthetic occluder never produced. There is no clean version of these faces, so only the emotion scores apply.

Each dataset is given as  name=meta.csv,landmarks.npz,quality.csv  (as filter_and_split.py; the landmarks and quality
files of every face are in the output of the cleaning notebook: <name>_landmarks.npz, <name>_quality.csv).

Output (out_dir), like the cleaning output but with its own file name (not taken for clean faces by the notebooks):
    real_occluded_meta.csv  dataset, name, label, expression, split (= test), occlusion (glasses / hat / other),
                            glasses_fraction, hat_fraction, occluded_fraction
    landmarks.npz           keys ("dataset/name") and MediaPipe landmarks (478x3)
"""
import os
import csv
import argparse
import numpy as np


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test set of real occlusions from the rejected faces.')
    parser.add_argument('--dataset', action='append', required=True, help='name=meta.csv,landmarks.npz,quality.csv')
    parser.add_argument('--out_dir', type=str, required=True)
    # quality checks: same defaults as filter_and_split.py
    parser.add_argument('--min_face_size', type=float, default=80)
    parser.add_argument('--max_black_fraction', type=float, default=0.8)
    parser.add_argument('--min_sharpness', type=float, default=30)
    parser.add_argument('--max_landmarks_outside', type=float, default=0.05)
    parser.add_argument('--image_size', type=int, default=256)
    # occlusion checks: a face is occluded when one of them is above the cleaning threshold
    parser.add_argument('--max_occluded_fraction', type=float, default=0.12)
    parser.add_argument('--max_glasses_fraction', type=float, default=0.01)
    parser.add_argument('--max_hat_fraction', type=float, default=0.02)
    args = parser.parse_args()

    rows, keys, kept_landmarks = [], [], []
    for spec in args.dataset:
        dataset, paths = spec.split('=', 1)
        meta_path, landmarks_path, quality_path = paths.split(',')
        meta = {r['name']: r for r in read_csv(meta_path)}
        lmk_data = np.load(landmarks_path)
        all_landmarks = lmk_data['landmarks']
        lmk_index = {n: i for i, n in enumerate(lmk_data['names'])}

        counts = {}
        for q in read_csv(quality_path):
            name = q['name']
            if name not in meta or q['detected'] != 'True' or name not in lmk_index:
                continue
            landmarks = all_landmarks[lmk_index[name]]
            outside = ((landmarks[:, :2] < 0) | (landmarks[:, :2] >= args.image_size)).any(axis=1).mean()
            if outside > args.max_landmarks_outside or not (
                    to_float(q['face_size']) >= args.min_face_size and to_float(q['black_fraction']) <= args.max_black_fraction
                    and to_float(q['sharpness']) >= args.min_sharpness):
                continue
            glasses, hat, occluded = (to_float(q[c]) for c in ['glasses_fraction', 'hat_fraction', 'occluded_fraction'])
            if np.isnan(occluded):  # not parsed
                continue
            if glasses > args.max_glasses_fraction:
                occlusion = 'glasses'
            elif hat > args.max_hat_fraction:
                occlusion = 'hat'
            elif occluded > args.max_occluded_fraction:
                occlusion = 'other'  # hands, objects, hair, ...
            else:
                continue  # a clean face
            m = meta[name]
            rows.append([dataset, name, m['label'], m['expression'], 'test', occlusion, q['glasses_fraction'],
                         q['hat_fraction'], q['occluded_fraction']])
            keys.append(f'{dataset}/{name}')
            kept_landmarks.append(landmarks)
            counts[occlusion] = counts.get(occlusion, 0) + 1
        print(f'{dataset}: {sum(counts.values())} faces with real occlusions {counts}')

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'real_occluded_meta.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'name', 'label', 'expression', 'split', 'occlusion', 'glasses_fraction',
                         'hat_fraction', 'occluded_fraction'])
        writer.writerows(rows)
    np.savez_compressed(os.path.join(args.out_dir, 'landmarks.npz'), keys=np.array(keys),
                        landmarks=np.array(kept_landmarks, dtype=np.float32).reshape(-1, 478, 3))
    print(f'Saved {len(rows)} faces to {args.out_dir}')
