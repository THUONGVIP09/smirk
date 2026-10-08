"""
Run MediaPipe FaceLandmarker on a folder of face crops (e.g. the output of crop_expw.py / prepare_affectnet.py)
and store all landmarks in a single file, in the order of meta.csv.

The crops may contain neighbouring faces (they are enlarged around the labeled face), so up to --max_faces faces are
detected and the one closest to the image center is kept.

Output (.npz):
    names       (N,)        image names from meta.csv
    landmarks   (N, 478, 3) x, y in pixels and z as returned by MediaPipe (as in utils/mediapipe_utils.py); NaN if not detected
    detected    (N,)        whether a face was found
    num_faces   (N,)        number of detected faces
"""
import os
import argparse
import csv
import numpy as np
import cv2
from multiprocessing import Pool
from tqdm import tqdm

detector = None


def init_detector(model_path, max_faces):
    global detector
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path),
                                           num_faces=max_faces)
    detector = vision.FaceLandmarker.create_from_options(options)


def process(image_path):
    import mediapipe as mp

    image = cv2.imread(image_path)
    if image is None:
        return None, 0
    h, w = image.shape[:2]

    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    if not result.face_landmarks:
        return None, 0

    faces = [np.array([[l.x * w, l.y * h, l.z] for l in face], dtype=np.float32) for face in result.face_landmarks]
    center = np.array([w / 2, h / 2])
    best = min(faces, key=lambda f: np.linalg.norm(f[:, :2].mean(axis=0) - center))
    return best, len(faces)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract MediaPipe landmarks for a folder of face crops.')
    parser.add_argument('--image_dir', type=str, required=True, help='Folder with the images listed in meta.csv')
    parser.add_argument('--meta', type=str, required=True, help='meta.csv with a "name" column (image file name without .png)')
    parser.add_argument('--out', type=str, required=True, help='Output .npz file')
    parser.add_argument('--model', type=str, default='assets/face_landmarker.task', help='MediaPipe face_landmarker.task')
    parser.add_argument('--max_faces', type=int, default=3)
    parser.add_argument('--num_processes', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Only process the first N images (quick test run); 0 = all')
    args = parser.parse_args()

    with open(args.meta) as f:
        names = [row['name'] for row in csv.DictReader(f)]
    if args.limit:
        names = names[:args.limit]
    paths = [os.path.join(args.image_dir, name + '.png') for name in names]

    landmarks = np.full((len(names), 478, 3), np.nan, dtype=np.float32)
    num_faces = np.zeros(len(names), dtype=np.int32)

    with Pool(args.num_processes, initializer=init_detector, initargs=(args.model, args.max_faces)) as pool:
        for i, (lmk, n) in enumerate(tqdm(pool.imap(process, paths, chunksize=16), total=len(paths))):
            num_faces[i] = n
            if lmk is not None:
                landmarks[i] = lmk

    detected = num_faces > 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, names=np.array(names), landmarks=landmarks, detected=detected, num_faces=num_faces)
    print(f'Detected a face in {detected.sum()}/{len(names)} images ({100 * detected.mean():.1f}%), '
          f'{(num_faces > 1).sum()} images with several faces')
