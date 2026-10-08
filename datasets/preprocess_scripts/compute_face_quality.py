"""
Measure the quality of face crops, to filter them before SMIRK training. Needs the landmarks of
extract_mediapipe_landmarks.py.

For every image:
    black_fraction    fraction of pure black (0,0,0) pixels in the 224x224 window that SMIRK crops around the
                      MediaPipe landmarks (BaseDataset.crop_face with --scale); black pixels come from padding
    sharpness         variance of the Laplacian of that window (low = blurry)
    face_size         size of the MediaPipe landmark box in pixels

and, with face parsing (BiSeNet-ResNet18 trained on CelebAMask-HQ, https://github.com/yakhyo/face-parsing) of that
window, inside the convex hull of the 105 MediaPipe landmarks that SMIRK uses (brows, eyes, nose, mouth):
    occluded_fraction pixels that are not skin / brows / eyes / nose / mouth / lips (hands, objects, glasses, hair, ...)
    glasses_fraction  pixels labeled as eyeglasses / sunglasses
    hat_fraction      pixels labeled as hat
Parsing metrics are NaN when parsing is disabled.

Output: CSV with one row per image of meta.csv.
"""
import os
import csv
import argparse
import numpy as np
import cv2
from tqdm import tqdm

# indices of the MediaPipe landmarks that SMIRK uses (same as mediapipe_indices in datasets/base_dataset.py)
SMIRK_MEDIAPIPE_INDICES = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 46, 52, 53,
        55, 63, 65, 66, 70, 105, 107, 249, 263, 362, 373, 374, 380,
       381, 382, 384, 385, 386, 387, 388, 390, 398, 466, 7, 33, 133,
       144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
       168, 6, 197, 195, 5, 4, 129, 98, 97, 2, 326, 327, 358,
         0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84,
        87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
       308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
       415]

def smirk_crop_matrix(landmarks, scale, image_size=224):
    # same crop as BaseDataset.crop_face / demo.py
    left, top = landmarks.min(axis=0)
    right, bottom = landmarks.max(axis=0)
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)
    src = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2]], dtype=np.float32)
    dst = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]], dtype=np.float32)
    return cv2.getAffineTransform(src, dst)


def image_metrics(image, landmarks, scale):
    M = smirk_crop_matrix(landmarks[:, :2], scale)
    window = cv2.warpAffine(image, M, (224, 224), flags=cv2.INTER_NEAREST, borderValue=0)
    black_fraction = float((window == 0).all(axis=2).mean())

    gray = cv2.cvtColor(cv2.warpAffine(image, M, (224, 224), flags=cv2.INTER_LINEAR, borderValue=0), cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    extent = landmarks[:, :2].max(axis=0) - landmarks[:, :2].min(axis=0)
    return black_fraction, sharpness, float(extent.max())


class FaceParser:
    """BiSeNet-ResNet18 face parsing (CelebAMask-HQ classes) from https://github.com/yakhyo/face-parsing (MIT)."""
    LABELS = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
              'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    FACE_PARTS = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'nose', 'mouth', 'u_lip', 'l_lip']
    SIZE = 512  # BiSeNet input size

    def __init__(self, device, repo_dir, weights):
        import sys
        import torch
        sys.path.insert(0, repo_dir)
        from models.bisenet import BiSeNet

        self.torch, self.device = torch, device
        self.net = BiSeNet(len(self.LABELS), backbone_name='resnet18')
        self.net.load_state_dict(torch.load(weights, map_location=device))
        self.net.to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.face_ids = [self.LABELS.index(n) for n in self.FACE_PARTS]
        self.glasses_id, self.hat_id = self.LABELS.index('eye_g'), self.LABELS.index('hat')

    def __call__(self, images, landmarks, scale):
        """images: list of HxWx3 BGR uint8; landmarks: list of 478x3 arrays. Returns a list of 3-tuples."""
        torch = self.torch
        # parse the window that SMIRK crops (the face fills it, as in CelebAMask-HQ)
        matrices = [smirk_crop_matrix(lmk[:, :2], scale, self.SIZE) for lmk in landmarks]
        windows = np.stack([cv2.warpAffine(im, M, (self.SIZE, self.SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
                            for im, M in zip(images, matrices)])
        x = torch.from_numpy(windows[..., ::-1].copy()).to(self.device).permute(0, 3, 1, 2).float() / 255
        x = (x - self.mean) / self.std
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=self.device != 'cpu'):
            labels = self.net(x)[0].argmax(dim=1).cpu().numpy()

        results = []
        for label_map, lmk, M in zip(labels, landmarks, matrices):
            points = lmk[SMIRK_MEDIAPIPE_INDICES, :2] @ M[:, :2].T + M[:, 2]
            region = np.zeros(label_map.shape, dtype=np.uint8)
            cv2.fillConvexPoly(region, cv2.convexHull(points.astype(np.int32)), 1)
            inside = label_map[region.astype(bool)]
            if inside.size == 0:
                results.append((np.nan, np.nan, np.nan))
                continue
            results.append((float(1 - np.isin(inside, self.face_ids).mean()),
                            float((inside == self.glasses_id).mean()),
                            float((inside == self.hat_id).mean())))
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute quality / occlusion metrics of face crops.')
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--landmarks', type=str, required=True, help='.npz from extract_mediapipe_landmarks.py')
    parser.add_argument('--out', type=str, required=True, help='Output CSV')
    parser.add_argument('--scale', type=float, default=1.6, help='SMIRK crop scale (test scale of the configs)')
    parser.add_argument('--no_parsing', action='store_true', help='Skip face parsing (no occlusion metrics)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--parser_repo', type=str, default='/tmp/face_parsing', help='Clone of github.com/yakhyo/face-parsing')
    parser.add_argument('--parser_weights', type=str, default='/tmp/bisenet_resnet18.pt', help='resnet18.pt of its releases')
    args = parser.parse_args()

    data = np.load(args.landmarks)
    names, landmarks, detected = data['names'], data['landmarks'], data['detected']
    face_parser = None if args.no_parsing else FaceParser(args.device, args.parser_repo, args.parser_weights)

    columns = ['name', 'detected', 'face_size', 'black_fraction', 'sharpness', 'occluded_fraction', 'glasses_fraction', 'hat_fraction']
    rows = []
    batch = []

    def flush():
        parsed = face_parser([b[1] for b in batch], [b[2] for b in batch], args.scale) if face_parser else [(np.nan,) * 3] * len(batch)
        for (row, _, _), p in zip(batch, parsed):
            rows.append(row + list(p))
        batch.clear()

    for name, lmk, ok in tqdm(zip(names, landmarks, detected), total=len(names)):
        if not ok:
            rows.append([name, False] + [np.nan] * 6)
            continue
        image = cv2.imread(os.path.join(args.image_dir, f'{name}.png'))
        black_fraction, sharpness, face_size = image_metrics(image, lmk, args.scale)
        batch.append(([name, True, face_size, black_fraction, sharpness], image, lmk))
        if len(batch) == args.batch_size:
            flush()
    if batch:
        flush()

    # keep the order of the landmarks file
    order = {n: i for i, n in enumerate(names)}
    rows.sort(key=lambda r: order[r[0]])
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {args.out}')
