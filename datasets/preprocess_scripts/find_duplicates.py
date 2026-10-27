"""
Find train faces that are (near-)duplicates of evaluation faces, within and across datasets.

ExpW, AffectNet and RAF-DB were all collected from the web, so the same photo can be in several of them (or twice in
one), e.g. in the RAF-DB test set and in the AffectNet train set. Training on all datasets would then test on faces
seen during training. Every face is compared on its face region (the landmark box, so the different crops of the
datasets do not matter), also mirrored:
    1. candidates: perceptual hash (DCT of the 32x32 face, 64 bits) within --max_hash_distance bits
    2. similar candidates: correlation of the face details (band-pass filtered center of the 64x64 faces)
       >= --verify_min_detail, or plain pixel correlation >= --verify_min_correlation. Neither score separates the
       duplicates on its own: copies with other resolution / crop / colors / text drop to 0.75, different people of
       similar faces reach 0.8
    3. duplicates: geometric verification of the similar candidates: SIFT points of the two 128x128 face crops that
       agree with one similarity transform close to the identity (RANSAC) >= --min_inliers (12: the pairs with 12-15
       matches are mostly copies, with 11 about half of them). The same photo keeps
       dozens of them under color / crop / resolution changes and added text (26-105 in tests), different photos have
       a few chance matches (0-8; up to 17 for two nearly identical frames of one video). Nearly identical faces
       (detail correlation >= --auto_detail) are duplicates without it (blurred faces can have too few SIFT points).
Look at the pairs around the threshold (notebook) to check it.

Inputs: folders with clean_meta.csv (cleaning output) or real_occluded_meta.csv (make_real_occlusion_set.py) and
landmarks.npz; evaluation faces are the val / test faces and the real occlusion faces.

Output (out_dir):
    duplicates.csv     every candidate pair (eval face, train face), hash distance, correlation, detail_correlation,
                       inliers (-1 = not verified), mirrored, duplicate
    exclude_train.csv  dataset, name of the train faces to leave out (data.exclude_files of the training,
                       --exclude of the benchmark: the emotion probe is trained on train faces too)
    preview.jpg        some duplicate pairs and the least similar accepted pairs
"""
import os
import csv
import pickle
import argparse
from multiprocessing import Pool
import time
import numpy as np
import cv2
from tqdm import tqdm

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


def init_worker():
    cv2.setNumThreads(0)  # OpenCV threads in forked worker processes can hang


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


VERIFY_SIZE = 128
_sift = None


def load_features(job):
    """SIFT keypoints / descriptors of the 128x128 face crop (mirrored if asked) for the geometric verification."""
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=400)
    path, landmarks, mirrored = job
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    face = cv2.warpAffine(image, crop_matrix(landmarks, 1.2, VERIFY_SIZE), (VERIFY_SIZE, VERIFY_SIZE), borderValue=0)
    if mirrored:
        face = np.ascontiguousarray(face[:, ::-1])
    keypoints, descriptors = _sift.detectAndCompute(face, None)
    if descriptors is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.float32([k.pt for k in keypoints]), descriptors


def geometric_inliers(a, b, matcher):
    """Matched SIFT points of two aligned faces that agree with one similarity transform close to the identity
    (same photo: crop / scale / color / text changes keep many of them; different photos: a few chance matches)."""
    (points_a, desc_a), (points_b, desc_b) = a, b
    if len(desc_a) < 2 or len(desc_b) < 2:
        return 0
    matches = [m for m, n in (p for p in matcher.knnMatch(desc_a, desc_b, k=2) if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(matches) < 4:
        return 0
    src = points_a[[m.queryIdx for m in matches]]
    dst = points_b[[m.trainIdx for m in matches]]
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return 0
    scale = np.hypot(M[0, 0], M[1, 0])
    center = np.float32([VERIFY_SIZE / 2, VERIFY_SIZE / 2])
    shift = np.linalg.norm(M[:, :2] @ center + M[:, 2] - center)
    if not (0.8 < scale < 1.25 and shift < 0.15 * VERIFY_SIZE):  # the faces are aligned: the transform must be too
        return 0
    return int(inliers.sum())


def details(faces):
    """Band-pass filtered center of the faces, normalized (N x D): the correlation of two rows compares the details."""
    x = faces.astype(np.float32)
    x = np.stack([cv2.GaussianBlur(f, (0, 0), 1.0) - cv2.GaussianBlur(f, (0, 0), 4.0) for f in x])
    m = FACE_SIZE // 8
    return normalized(x[:, m:-m, m:-m])


def main():
    parser = argparse.ArgumentParser(description='Find train faces that duplicate evaluation faces.')
    parser.add_argument('--meta_dir', action='append', required=True,
                        help='Folder with clean_meta.csv or real_occluded_meta.csv and landmarks.npz (repeatable)')
    parser.add_argument('--image_dir', action='append', required=True, help='dataset=folder with the crop images (repeatable)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--max_hash_distance', type=int, default=12, help='Candidates: hash distance in bits (of 64)')
    parser.add_argument('--verify_min_detail', type=float, default=0.5, help='Verify the candidates with this face detail correlation')
    parser.add_argument('--verify_min_correlation', type=float, default=0.85, help='... or this plain pixel correlation')
    parser.add_argument('--min_inliers', type=int, default=12, help='Duplicates: geometrically consistent SIFT matches')
    parser.add_argument('--auto_detail', type=float, default=0.95, help='Duplicates without verification: face detail correlation')
    parser.add_argument('--cache', default=None, help='.npy file keeping the loaded faces (and <cache>.sift.pkl the SIFT '
                                                       'features): re-runs with other thresholds skip the loading')
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
    print(f'{len(rows)} faces:', {s: sum(r['split'] == s for r in rows) for s in ['train', 'val', 'test', 'real']}, flush=True)

    jobs = [(os.path.join(image_dirs[r['dataset']], r['name'] + '.png'), landmarks[f"{r['dataset']}/{r['name']}"][:, :2])
            for r in rows]
    if args.cache and os.path.exists(args.cache) and len(np.load(args.cache, mmap_mode='r')) == len(jobs):
        faces = np.load(args.cache)
        print('faces loaded from', args.cache, flush=True)
    else:
        started = time.time()
        with Pool(args.num_processes, initializer=init_worker) as pool:
            faces = np.stack(list(tqdm(pool.imap(load_face, jobs, chunksize=64), total=len(jobs), desc='loading faces', mininterval=10)))
        print(f'faces loaded in {(time.time() - started) / 60:.1f} min', flush=True)
        if args.cache:
            np.save(args.cache, faces)

    train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    evaluation = np.array([i for i, r in enumerate(rows) if r['split'] != 'train'])
    train_hash, train_faces, train_details = phash(faces[train]), normalized(faces[train]), details(faces[train])

    pairs = []
    print('comparing', flush=True)
    for mirrored in [False, True]:
        eval_faces = faces[evaluation][:, :, ::-1] if mirrored else faces[evaluation]
        eval_hash, eval_norm, eval_details = phash(eval_faces), normalized(eval_faces), details(eval_faces)
        for start in range(0, len(evaluation), 1024):
            distance = (64 - eval_hash[start:start + 1024] @ train_hash.T) / 2   # Hamming distance from the signs
            k = min(args.candidates_per_face, distance.shape[1])
            nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
            for row, columns in enumerate(nearest):
                e = start + row
                for t in columns[distance[row, columns] <= args.max_hash_distance]:
                    pairs.append((evaluation[e], train[t], int(distance[row, t]), float(eval_norm[e] @ train_faces[t]),
                                  float(eval_details[e] @ train_details[t]), mirrored))
    print(f'{len(pairs)} candidate pairs')

    # one row per (eval face, train face): the more similar orientation
    best = {}
    for e, t, distance, correlation, detail, mirrored in pairs:
        if (e, t) not in best or detail > best[(e, t)][2]:
            best[(e, t)] = (distance, correlation, detail, mirrored)

    os.makedirs(args.out_dir, exist_ok=True)
    # geometric verification of the similar candidates
    verify = [(e, t, mirrored) for (e, t), (_, correlation, detail, mirrored) in best.items()
              if detail >= args.verify_min_detail or correlation >= args.verify_min_correlation]
    keys = sorted({(e, mirrored) for e, _, mirrored in verify} | {(t, False) for _, t, _ in verify})
    features = {}
    sift_cache = args.cache + '.sift.pkl' if args.cache else None
    if sift_cache and os.path.exists(sift_cache):
        with open(sift_cache, 'rb') as f:
            features = pickle.load(f)
    missing = [k for k in keys if k not in features]
    print(f'{len(verify)} similar candidate pairs to verify ({len(missing)} faces to load)', flush=True)
    if missing:
        with Pool(args.num_processes, initializer=init_worker) as pool:
            loaded = tqdm(pool.imap(load_features, [jobs[i] + (mirrored,) for i, mirrored in missing], chunksize=32),
                          total=len(missing), desc='SIFT features', mininterval=10)
            features.update(zip(missing, loaded))
        if sift_cache:
            with open(sift_cache, 'wb') as f:
                pickle.dump(features, f)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    inliers = {(e, t): geometric_inliers(features[(e, mirrored)], features[(t, False)], matcher)
               for e, t, mirrored in tqdm(verify, desc='verifying', mininterval=10)}

    table, excluded = [], {}
    ordered = sorted(best.items(), key=lambda x: (-inliers.get(x[0], -1), -x[1][2]))
    is_duplicate = lambda key, detail: inliers.get(key, -1) >= args.min_inliers or detail >= args.auto_detail
    for (e, t), (distance, correlation, detail, mirrored) in ordered:
        count = inliers.get((e, t), -1)
        duplicate = is_duplicate((e, t), detail)
        a, b = rows[e], rows[t]
        table.append([a['dataset'], a['name'], a['split'], b['dataset'], b['name'], distance, round(correlation, 4),
                      round(detail, 4), count, mirrored, duplicate])
        if duplicate:
            excluded[(b['dataset'], b['name'])] = True
    with open(os.path.join(args.out_dir, 'duplicates.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eval_dataset', 'eval_name', 'eval_split', 'train_dataset', 'train_name', 'hash_distance',
                         'correlation', 'detail_correlation', 'inliers', 'mirrored', 'duplicate'])
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
    duplicates = [p for p in ordered if is_duplicate(p[0], p[1][2])]
    rejected = [p for p in ordered if not is_duplicate(p[0], p[1][2])]
    groups = [duplicates[:8], duplicates[-8:], rejected[:8]]
    tiles = []
    for group in groups:
        line = []
        for (e, t), (distance, correlation, detail, mirrored) in group:
            pair = np.concatenate([faces[e][:, ::-1] if mirrored else faces[e], faces[t]], axis=1)
            pair = cv2.resize(pair, (256, 128))
            cv2.putText(pair, f'{inliers.get((e, t), -1)}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
            line.append(np.pad(pair, 4))
        line += [np.zeros((136, 264), np.uint8)] * (8 - len(line))
        tiles.append(np.concatenate(line, axis=1))
    cv2.imwrite(os.path.join(args.out_dir, 'preview.jpg'), np.concatenate(tiles, axis=0))
    print('preview rows: most similar duplicates / least similar duplicates / most similar non-duplicates')


if __name__ == '__main__':
    main()
