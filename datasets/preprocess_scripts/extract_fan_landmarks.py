"""
FAN landmarks (68 points) of the clean face crops, with the same tools as the original SMIRK preprocessing
(apply_fan_to_dataset.py: ibug RetinaFace mobilenet0.25 detector + ibug FAN 2dfan2_alt, https://github.com/hhj1897).

SMIRK uses the 17 jaw-line FAN points in its landmark loss (MediaPipe has no jaw line) and 5 FAN points to align the
MICA input. Only the clean faces of clean_meta.csv are processed; when several faces are detected the one closest to
the MediaPipe face (or to the image center) is kept.

Output (.npz): names (N,), landmarks (N, 68, 2) in pixels of the 256x256 crop (NaN if no face), scores (N, 68),
detected (N,)
"""
import os
import csv
import argparse
import numpy as np
import cv2
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='FAN landmarks of the clean faces of one dataset.')
    parser.add_argument('--clean_dir', action='append', required=True, help='Folder with clean_meta.csv and landmarks.npz (repeatable)')
    parser.add_argument('--image_dir', required=True, help='Folder with the 256x256 crops of this dataset')
    parser.add_argument('--dataset', required=True, help='Dataset name as in clean_meta.csv')
    parser.add_argument('--out', required=True, help='Output .npz')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threshold', type=float, default=0.8, help='Face detection threshold (as apply_fan_to_dataset.py)')
    parser.add_argument('--limit', type=int, default=0, help='Only the first N faces (quick test); 0 = all')
    args = parser.parse_args()

    from ibug.face_detection import RetinaFacePredictor
    from ibug.face_alignment import FANPredictor
    detector = RetinaFacePredictor(threshold=args.threshold, device=args.device, model=RetinaFacePredictor.get_model('mobilenet0.25'))
    fan = FANPredictor(device=args.device, model=FANPredictor.get_model('2dfan2_alt'))

    names, mediapipe = [], {}
    for folder in args.clean_dir:
        with open(os.path.join(folder, 'clean_meta.csv')) as f:
            names += [r['name'] for r in csv.DictReader(f) if r['dataset'] == args.dataset]
        data = np.load(os.path.join(folder, 'landmarks.npz'))
        prefix = args.dataset + '/'
        mediapipe.update({k[len(prefix):]: v for k, v in zip(data['keys'], data['landmarks']) if k.startswith(prefix)})
    if args.limit:
        names = names[:args.limit]

    landmarks = np.full((len(names), 68, 2), np.nan, np.float32)
    scores = np.zeros((len(names), 68), np.float32)
    for i, name in enumerate(tqdm(names)):
        image = cv2.imread(os.path.join(args.image_dir, name + '.png'))
        if image is None:
            continue
        faces = detector(image, rgb=False)
        if len(faces) == 0:
            continue
        reference = mediapipe[name][:, :2].mean(0) if name in mediapipe else np.array(image.shape[1::-1]) / 2
        centers = (faces[:, :2] + faces[:, 2:4]) / 2
        face = faces[[np.argmin(np.linalg.norm(centers - reference, axis=1))]]
        points, point_scores = fan(image, face, rgb=False)
        landmarks[i], scores[i] = points[0], point_scores[0]

    detected = ~np.isnan(landmarks[:, 0, 0])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, names=np.array(names), landmarks=landmarks, scores=scores, detected=detected)
    print(f'{args.dataset}: FAN landmarks for {detected.sum()}/{len(names)} faces ({100 * detected.mean():.1f}%)')


if __name__ == '__main__':
    main()
