"""
Pre-generate synthetic occlusions for the clean faces of one dataset (src/utils/occlusion.py), so that occlusion-aware
training only has to add the resulting Kaggle dataset as input.

The occluders are drawn on the 256x256 face crops with the MediaPipe landmarks of the clean image (clean_meta.csv /
landmarks.npz of the cleaning step), so every occluded image is pixel-aligned with its clean image: the same SMIRK
crop / augmentation can be applied to both, and the clean image and landmarks stay the training targets.

Besides single parts (mouth, eyes, brows, nose), large regions are drawn by default (lower_face: like a face mask or a
hand over the mouth; upper_face: like sunglasses), so that the occluded face area resembles real occlusions.

Train faces get --train_variants different occlusions, val / test faces --eval_variants. Seeds are fixed per image and
variant (and differ from the seeds of the evaluation benchmark), so the dataset can be regenerated identically.

Output (see src/utils/occluded_store.py): occluded_meta.csv, images.bin (JPEG), masks.bin (PNG), preview/*.jpg

    python datasets/preprocess_scripts/make_occluded_dataset.py --clean_dir CLEAN --image_dir IMAGES \
        --dataset expw --out_dir occluded_expw
"""
import os
import sys
import csv
import argparse
import numpy as np
import cv2
from multiprocessing import Pool
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.occlusion import FaceOccluder, occlusion_rng  # noqa: E402

PARTS = {'mouth': ['mouth'], 'eyes': ['left_eye', 'right_eye'], 'brows': ['left_brow', 'right_brow'], 'nose': ['nose']}
SEED_OFFSET = 1000  # the evaluation benchmark uses seeds 0..5
worker = {}


def init_worker(image_dir, texture_paths, jpeg_quality, parts, scale_range):
    textures = [cv2.imread(p)[..., ::-1] for p in texture_paths]
    worker.update(image_dir=image_dir, jpeg_quality=jpeg_quality,
                  occluder=FaceOccluder(parts=parts, scale_range=scale_range,
                                        texture_pool=[t for t in textures if t is not None]))


def occlude(job):
    dataset, name, variant, landmarks = job
    image = cv2.imread(os.path.join(worker['image_dir'], name + '.png'))
    if image is None:
        return job, None
    image = np.ascontiguousarray(image[..., ::-1])
    occluded, mask, info = worker['occluder'](image, landmarks, rng=occlusion_rng(f'{dataset}/{name}', seed=SEED_OFFSET + variant))
    ok_i, jpg = cv2.imencode('.jpg', occluded[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, worker['jpeg_quality']])
    ok_m, png = cv2.imencode('.png', np.round(mask * 255).astype(np.uint8))
    assert ok_i and ok_m
    return job, (jpg.tobytes(), png.tobytes(), info)


def main():
    parser = argparse.ArgumentParser(description='Pre-generate synthetic occlusions of the clean faces of one dataset.')
    parser.add_argument('--clean_dir', action='append', required=True, help='Folder with clean_meta.csv and landmarks.npz (repeatable)')
    parser.add_argument('--image_dir', required=True, help='Folder with the 256x256 crops of this dataset')
    parser.add_argument('--dataset', required=True, help='Dataset name as in clean_meta.csv (expw, affectnet, rafdb)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--train_variants', type=int, default=2, help='Different occlusions per train face')
    parser.add_argument('--eval_variants', type=int, default=1, help='Different occlusions per val / test face')
    parser.add_argument('--parts', default='mouth,eyes,brows,nose,lower_face,upper_face',
                        help='Occluded parts / regions to draw from (keys of PART_CHOICES in src/utils/occlusion.py)')
    parser.add_argument('--scale_min', type=float, default=0.8, help='Occluder size relative to the part, minimum')
    parser.add_argument('--scale_max', type=float, default=1.8, help='Occluder size relative to the part, maximum')
    parser.add_argument('--jpeg_quality', type=int, default=95)
    parser.add_argument('--textures', type=int, default=300, help='Clean faces used as source of image-texture occluders')
    parser.add_argument('--preview', type=int, default=48, help='Occluded images also written as plain files')
    parser.add_argument('--num_processes', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Only the first N faces (quick test); 0 = all')
    args = parser.parse_args()

    rows, landmarks = [], {}
    for folder in args.clean_dir:
        with open(os.path.join(folder, 'clean_meta.csv')) as f:
            rows += [r for r in csv.DictReader(f) if r['dataset'] == args.dataset]
        data = np.load(os.path.join(folder, 'landmarks.npz'))
        prefix = args.dataset + '/'
        landmarks.update({k: v for k, v in zip(data['keys'], data['landmarks']) if k.startswith(prefix)})
    assert rows, f'no clean faces of dataset {args.dataset}'
    if args.limit:
        rows = rows[:args.limit]

    # texture occluders are cut from other clean faces (fixed choice); test faces are never used as a texture source
    rng = np.random.default_rng(0)
    pool = [r for r in rows if r['split'] == 'train'] or rows
    picks = rng.choice(len(pool), min(args.textures, len(pool)), replace=False)
    texture_paths = [os.path.join(args.image_dir, pool[i]['name'] + '.png') for i in picks]

    jobs = []
    for r in rows:
        n = args.train_variants if r['split'] == 'train' else args.eval_variants
        jobs += [(args.dataset, r['name'], v, landmarks[f"{args.dataset}/{r['name']}"][:, :2]) for v in range(n)]
    meta = {r['name']: r for r in rows}

    os.makedirs(os.path.join(args.out_dir, 'preview'), exist_ok=True)
    columns = ['dataset', 'name', 'variant', 'split', 'label', 'expression', 'parts', 'shapes', 'fills'] + \
              [f'covered_{p}' for p in PARTS] + ['occluded_area', 'image_offset', 'image_length', 'mask_offset', 'mask_length']
    out_rows, missing = [], 0
    with open(os.path.join(args.out_dir, 'images.bin'), 'wb') as fi, open(os.path.join(args.out_dir, 'masks.bin'), 'wb') as fm, \
            Pool(args.num_processes, initializer=init_worker,
                 initargs=(args.image_dir, texture_paths, args.jpeg_quality, args.parts.split(','),
                           (args.scale_min, args.scale_max))) as workers:
        for job, result in tqdm(workers.imap(occlude, jobs, chunksize=32), total=len(jobs)):
            dataset, name, variant, _ = job
            if result is None:
                missing += 1
                continue
            jpg, png, info = result
            image_offset, mask_offset = fi.tell(), fm.tell()
            fi.write(jpg)
            fm.write(png)
            if len(out_rows) < args.preview:
                with open(os.path.join(args.out_dir, 'preview', f'{name}_v{variant}.jpg'), 'wb') as f:
                    f.write(jpg)
            m = meta[name]
            out_rows.append([dataset, name, variant, m['split'], m['label'], m['expression'],
                             '|'.join(info['parts']), '|'.join(info['shapes']), '|'.join(info['fills'])] +
                            [round(max(info['covered'][g] for g in groups), 4) for groups in PARTS.values()] +
                            [round(info['area'], 4), image_offset, len(jpg), mask_offset, len(png)])

    with open(os.path.join(args.out_dir, 'occluded_meta.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(out_rows)

    areas = np.array([r[columns.index('occluded_area')] for r in out_rows])
    print(f'{args.dataset}: {len(out_rows)} occluded images from {len(rows)} clean faces ({missing} unreadable)')
    print(f'occluded face area: mean {areas.mean():.3f}, percentiles 5/25/50/75/95 = {np.percentile(areas, [5, 25, 50, 75, 95]).round(3)}')
    for split in ['train', 'val', 'test']:
        print(f'  {split}: {sum(r[3] == split for r in out_rows)} images')


if __name__ == '__main__':
    main()
