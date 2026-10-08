"""
Keep the clean faces of one or more datasets and split them into train / val / test.

Each dataset is given as  name=meta.csv,landmarks.npz,quality.csv  (outputs of crop_expw.py / prepare_affectnet.py,
extract_mediapipe_landmarks.py and compute_face_quality.py). A face is kept when MediaPipe found it and its quality
metrics pass the thresholds (faces cut by the image border, i.e. with MediaPipe landmarks outside the image, are
rejected as well); NaN parsing metrics (parser found no face) count as failing unless --keep_unparsed.

Splits: datasets whose meta.csv already has train/val/test splits keep them (e.g. AffectNet); the others are split
per expression label with --val_fraction / --test_fraction (fixed seed).

Output (out_dir):
    clean_meta.csv     dataset, name, label, expression, split + quality metrics
    landmarks.npz      keys ("dataset/name") and MediaPipe landmarks (478x3) of the kept faces
    rejected.csv       dataset, name, reason   (first failing check)
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
    parser = argparse.ArgumentParser(description='Filter face crops by quality and split them.')
    parser.add_argument('--dataset', action='append', required=True, help='name=meta.csv,landmarks.npz,quality.csv')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--min_face_size', type=float, default=80, help='MediaPipe landmark box size in pixels')
    parser.add_argument('--max_black_fraction', type=float, default=0.8)
    parser.add_argument('--min_sharpness', type=float, default=30)
    parser.add_argument('--max_occluded_fraction', type=float, default=0.12)
    parser.add_argument('--max_glasses_fraction', type=float, default=0.01)
    parser.add_argument('--max_hat_fraction', type=float, default=0.02)
    parser.add_argument('--max_landmarks_outside', type=float, default=0.05,
                        help='Reject faces cut by the image border: fraction of MediaPipe landmarks outside the image')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the (square) face crops')
    parser.add_argument('--keep_unparsed', action='store_true', help='Keep faces without parsing metrics')
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--test_fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # (column, failing test, reason)
    checks = [
        ('face_size', lambda v: v < args.min_face_size, 'face_too_small'),
        ('black_fraction', lambda v: v > args.max_black_fraction, 'black_border'),
        ('sharpness', lambda v: v < args.min_sharpness, 'blurry'),
        ('glasses_fraction', lambda v: v > args.max_glasses_fraction, 'glasses'),
        ('hat_fraction', lambda v: v > args.max_hat_fraction, 'hat'),
        ('occluded_fraction', lambda v: v > args.max_occluded_fraction, 'occluded'),
    ]
    parsing_columns = {'occluded_fraction', 'glasses_fraction', 'hat_fraction'}

    def landmarks_outside(lmk):
        xy = lmk[:, :2]
        return float(((xy < 0) | (xy >= args.image_size)).any(axis=1).mean())

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    clean_rows, rejected_rows, keys, kept_landmarks = [], [], [], []

    for spec in args.dataset:
        dataset, paths = spec.split('=', 1)
        meta_path, landmarks_path, quality_path = paths.split(',')
        meta = {r['name']: r for r in read_csv(meta_path)}
        quality = {r['name']: r for r in read_csv(quality_path)}
        lmk_data = np.load(landmarks_path)
        lmk_index = {n: i for i, n in enumerate(lmk_data['names'])}

        kept = []
        for name, m in meta.items():
            q = quality.get(name)
            reason = None
            if q is None:
                reason = 'not_processed'  # not in the quality file (e.g. landmarks computed with --limit)
            elif q['detected'] != 'True':
                reason = 'no_face'
            elif landmarks_outside(lmk_data['landmarks'][lmk_index[name]]) > args.max_landmarks_outside:
                reason = 'face_cut'
            else:
                for column, fails, why in checks:
                    value = to_float(q[column])
                    if np.isnan(value):
                        if column in parsing_columns and not args.keep_unparsed:
                            reason = 'not_parsed'
                            break
                        continue
                    if fails(value):
                        reason = why
                        break
            if reason:
                rejected_rows.append([dataset, name, reason])
            else:
                kept.append((name, m, q))

        # split: keep existing train/val/test, otherwise stratified random split per label
        has_split = all(m.get('split') in ('train', 'val', 'test') for _, m, _ in kept) and kept
        splits = {}
        if has_split:
            splits = {name: m['split'] for name, m, _ in kept}
        else:
            for label in sorted({m['label'] for _, m, _ in kept}):
                names = [name for name, m, _ in kept if m['label'] == label]
                rng.shuffle(names)
                n_test, n_val = int(round(len(names) * args.test_fraction)), int(round(len(names) * args.val_fraction))
                for i, name in enumerate(names):
                    splits[name] = 'test' if i < n_test else 'val' if i < n_test + n_val else 'train'

        for name, m, q in kept:
            clean_rows.append([dataset, name, m['label'], m['expression'], splits[name]] +
                              [q[c] for c, _, _ in checks])
            keys.append(f'{dataset}/{name}')
            kept_landmarks.append(lmk_data['landmarks'][lmk_index[name]])

        n_rejected = sum(r[0] == dataset for r in rejected_rows)
        print(f'{dataset}: kept {len(kept)} / {len(meta)} ({100 * len(kept) / max(len(meta), 1):.1f}%), '
              f'splits kept from meta.csv: {bool(has_split)}')
        reasons = {}
        for r in rejected_rows:
            if r[0] == dataset:
                reasons[r[2]] = reasons.get(r[2], 0) + 1
        print('  rejected:', dict(sorted(reasons.items(), key=lambda x: -x[1])), f'total {n_rejected}')

    with open(os.path.join(args.out_dir, 'clean_meta.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'name', 'label', 'expression', 'split'] + [c for c, _, _ in checks])
        writer.writerows(clean_rows)
    with open(os.path.join(args.out_dir, 'rejected.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'name', 'reason'])
        writer.writerows(rejected_rows)
    np.savez_compressed(os.path.join(args.out_dir, 'landmarks.npz'), keys=np.array(keys),
                        landmarks=np.array(kept_landmarks, dtype=np.float32).reshape(-1, 478, 3))
    print(f'Saved {len(clean_rows)} clean faces to {args.out_dir}')
