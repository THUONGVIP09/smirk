"""
Find train faces that are (near-)duplicates of evaluation faces, within and across datasets.

ExpW, AffectNet and RAF-DB were all collected from the web, so the same photo can be in several of them (or twice in
one), e.g. in the RAF-DB test set and in the AffectNet train set. Training on all datasets would then test on faces
seen during training. Every face is compared on its face region (the landmark box, so the different crops of the
datasets do not matter), also mirrored:
    1. candidates: perceptual hash (DCT of the 32x32 face, 64 bits) within --max_hash_distance bits
    2. duplicates: normalized correlation of the 64x64 grayscale faces >= --min_correlation
Look at the pairs saved in the preview to check the thresholds.

Inputs: folders with clean_meta.csv (cleaning output) or real_occluded_meta.csv (make_real_occlusion_set.py) and
landmarks.npz; evaluation faces are the val / test faces and the real occlusion faces.

Output (out_dir):
    duplicates.csv     every candidate pair (eval face, train face), hash distance, correlation, mirrored, duplicate
    exclude_train.csv  dataset, name of the train faces to leave out (data.exclude_files of the training,
                       --exclude of the benchmark: the emotion probe is trained on train faces too)
    preview.jpg        some duplicate pairs and the least similar accepted pairs
"""
import os
import csv
import argparse
from multiprocessing import Pool
import numpy as np
import cv2

FACE_SIZE = 64


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


def load_face(job):
    path, landmarks = job
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return np.zeros((FACE_SIZE, FACE_SIZE), np.uint8)
    return cv2.warpAffine(image, crop_matrix(landmarks, 1.0, FACE_SIZE), (FACE_SIZE, FACE_SIZE), borderValue=0)


def phash(faces):
    """64-bit perceptual hashes as N x 64 signs (+1 / -1)."""
    signs = np.zeros((len(faces), 64), np.float32)
    for i, face in enumerate(faces):
        low = cv2.dct(cv2.resize(face, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].flatten()
        signs[i] = np.where(low > np.median(low[1:]), 1, -1)
    return signs


def normalized(faces):
    x = faces.reshape(len(faces), -1).astype(np.float32)
    x -= x.mean(1, keepdims=True)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def main():
    parser = argparse.ArgumentParser(description='Find train faces that duplicate evaluation faces.')
    parser.add_argument('--meta_dir', action='append', required=True,
                        help='Folder with clean_meta.csv or real_occluded_meta.csv and landmarks.npz (repeatable)')
    parser.add_argument('--image_dir', action='append', required=True, help='dataset=folder with the crop images (repeatable)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--max_hash_distance', type=int, default=12, help='Candidates: hash distance in bits (of 64)')
    parser.add_argument('--min_correlation', type=float, default=0.9, help='Duplicates: correlation of the 64x64 faces')
    parser.add_argument('--candidates_per_face', type=int, default=10)
    parser.add_argument('--num_processes', type=int, default=4)
    args = parser.parse_args()
    image_dirs = dict(spec.split('=', 1) for spec in args.image_dir)

    rows, landmarks = [], {}
    for folder in args.meta_dir:
        for meta, role in [('clean_meta.csv', None), ('real_occluded_meta.csv', 'real')]:
            if os.path.exists(os.path.join(folder, meta)):
                with open(os.path.join(folder, meta)) as f:
                    rows += [{**r, 'split': role or r['split']} for r in csv.DictReader(f) if r['dataset'] in image_dirs]
        data = np.load(os.path.join(folder, 'landmarks.npz'))
        landmarks.update(zip(data['keys'], data['landmarks']))
    print(f'{len(rows)} faces:', {s: sum(r['split'] == s for r in rows) for s in ['train', 'val', 'test', 'real']})

    jobs = [(os.path.join(image_dirs[r['dataset']], r['name'] + '.png'), landmarks[f"{r['dataset']}/{r['name']}"][:, :2])
            for r in rows]
    with Pool(args.num_processes) as pool:
        faces = np.stack(pool.map(load_face, jobs, chunksize=256))
    print('faces loaded')

    train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    evaluation = np.array([i for i, r in enumerate(rows) if r['split'] != 'train'])
    train_hash, train_faces = phash(faces[train]), normalized(faces[train])

    pairs = []
    for mirrored in [False, True]:
        eval_faces = faces[evaluation][:, :, ::-1] if mirrored else faces[evaluation]
        eval_hash, eval_norm = phash(eval_faces), normalized(eval_faces)
        for start in range(0, len(evaluation), 1024):
            distance = (64 - eval_hash[start:start + 1024] @ train_hash.T) / 2   # Hamming distance from the signs
            k = min(args.candidates_per_face, distance.shape[1])
            nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
            for row, columns in enumerate(nearest):
                e = start + row
                for t in columns[distance[row, columns] <= args.max_hash_distance]:
                    pairs.append((evaluation[e], train[t], int(distance[row, t]),
                                  float(eval_norm[e] @ train_faces[t]), mirrored))
    print(f'{len(pairs)} candidate pairs')

    # one row per (eval face, train face): the more similar orientation
    best = {}
    for e, t, distance, correlation, mirrored in pairs:
        if (e, t) not in best or correlation > best[(e, t)][1]:
            best[(e, t)] = (distance, correlation, mirrored)

    os.makedirs(args.out_dir, exist_ok=True)
    table, excluded = [], {}
    for (e, t), (distance, correlation, mirrored) in sorted(best.items(), key=lambda x: -x[1][1]):
        duplicate = correlation >= args.min_correlation
        a, b = rows[e], rows[t]
        table.append([a['dataset'], a['name'], a['split'], b['dataset'], b['name'], distance, round(correlation, 4), mirrored, duplicate])
        if duplicate:
            excluded[(b['dataset'], b['name'])] = True
    with open(os.path.join(args.out_dir, 'duplicates.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eval_dataset', 'eval_name', 'eval_split', 'train_dataset', 'train_name', 'hash_distance',
                         'correlation', 'mirrored', 'duplicate'])
        writer.writerows(table)
    with open(os.path.join(args.out_dir, 'exclude_train.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'name'])
        writer.writerows(sorted(excluded))

    counts = {}
    for r in table:
        if r[-1]:
            key = f'{r[0]} {r[2]} <-> {r[3]} train'
            counts[key] = counts.get(key, 0) + 1
    print('duplicate pairs:', dict(sorted(counts.items())) or 'none')
    print(f'{len(excluded)} train faces to exclude ({100 * len(excluded) / max(len(train), 1):.2f}% of the train faces)')

    # preview: the most similar duplicates, the least similar duplicates (threshold check) and the best rejected pairs
    duplicates = [p for p in sorted(best.items(), key=lambda x: -x[1][1]) if p[1][1] >= args.min_correlation]
    rejected = [p for p in sorted(best.items(), key=lambda x: -x[1][1]) if p[1][1] < args.min_correlation]
    groups = [duplicates[:8], duplicates[-8:], rejected[:8]]
    tiles = []
    for group in groups:
        line = []
        for (e, t), (distance, correlation, mirrored) in group:
            pair = np.concatenate([faces[e][:, ::-1] if mirrored else faces[e], faces[t]], axis=1)
            pair = cv2.resize(pair, (256, 128))
            cv2.putText(pair, f'{correlation:.2f}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
            line.append(np.pad(pair, 4))
        line += [np.zeros((136, 264), np.uint8)] * (8 - len(line))
        tiles.append(np.concatenate(line, axis=1))
    cv2.imwrite(os.path.join(args.out_dir, 'preview.jpg'), np.concatenate(tiles, axis=0))
    print('preview rows: most similar duplicates / least similar duplicates / most similar non-duplicates')


if __name__ == '__main__':
    main()
