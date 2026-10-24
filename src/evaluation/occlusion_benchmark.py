"""
Occlusion benchmark for SMIRK encoders: how much do the predictions change when facial parts are occluded?

Every test face is evaluated clean and under synthetic occlusions (src/utils/occlusion.py, fixed per image):
    random              1-2 random parts
    mouth/eyes/brows/nose  exactly this part
The occluder is drawn on the full face crop with the landmarks of the clean image, then the same SMIRK crop
(BaseDataset.crop_face, --scale) is used for the clean and the occluded image.

Metrics (per test face and protocol):
    exp_dist   ||psi_occluded - psi_clean|| over the expression parameters
    exp_rel    exp_dist / ||psi_clean||
    exp_scale  ||psi_occluded|| / ||psi_clean||; > 1 means more intense expressions under occlusion (the SMIRK paper,
               Sec. 4.5, reports that the model "tends to assume more intense expressions" when parts are missing)
    expr_mm    mean per-vertex distance (mm) between the FLAME meshes built from the clean / occluded expression, jaw and
               eyelid parameters, with the clean shape and a neutral pose (so that pose / camera changes do not count)
    occluded_area  fraction of the face (convex hull of the 478 MediaPipe landmarks) covered by the occluders
    lmk_px     mean distance (pixels of the 224 crop) between the projected FLAME MediaPipe landmarks and the
               MediaPipe landmarks of the clean image (the training targets of SMIRK)
Emotion probe (as the EMOCA / SMIRK protocol: an MLP on the predicted parameters): MLP on [expression, jaw, eyelid]
of the clean train faces, all expression labels (8 with AffectNet's contempt); accuracy and mean class accuracy on the
clean / occluded test faces, computed over the classes present in each dataset (8 for AffectNet, 7 for ExpW / RAF-DB).
Valence / arousal are not evaluated (no such labels in the datasets used here).

Run from the repository root (FLAME assets are loaded with relative paths), e.g.
    python -m src.evaluation.occlusion_benchmark --clean_dir CLEAN_FACES --image_dir expw=.../images \
        --image_dir affectnet=.../images --checkpoint pretrained_models/SMIRK_em1.pt --out_dir results/baseline
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm

from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.util import batch_orth_proj
from src.utils.occlusion import FaceOccluder, occlusion_rng

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
PARTS = ['mouth', 'eyes', 'brows', 'nose']
COVERED_GROUPS = {'mouth': ['mouth'], 'eyes': ['left_eye', 'right_eye'], 'brows': ['left_brow', 'right_brow'], 'nose': ['nose']}
CROP_SIZE = 224
METRICS = ['occluded_area', 'exp_dist', 'exp_rel', 'exp_scale', 'expr_mm', 'lmk_px']


def crop_matrix(landmarks, scale, size=CROP_SIZE):
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


class FaceCrops(torch.utils.data.Dataset):
    """Clean (and occluded) SMIRK crops of the faces listed in a clean_meta table."""

    def __init__(self, rows, image_dirs, landmarks, protocols, scale):
        self.rows, self.image_dirs, self.landmarks = rows.reset_index(drop=True), image_dirs, landmarks
        self.protocols, self.scale = protocols, scale
        self.occluder = FaceOccluder()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        key = f"{r['dataset']}/{r['name']}"
        image = np.ascontiguousarray(cv2.imread(os.path.join(self.image_dirs[r['dataset']], r['name'] + '.png'))[..., ::-1])
        landmarks = self.landmarks[key][:, :2]
        M = crop_matrix(landmarks, self.scale)

        crops, covered, area = [cv2.warpAffine(image, M, (CROP_SIZE, CROP_SIZE), borderValue=0)], [], []
        for p, protocol in enumerate(self.protocols):
            occluded, _, info = self.occluder(image, landmarks, rng=occlusion_rng(key, seed=p),
                                              parts=None if protocol == 'random' else [protocol])
            crops.append(cv2.warpAffine(occluded, M, (CROP_SIZE, CROP_SIZE), borderValue=0))
            covered.append([max(info['covered'][g] for g in COVERED_GROUPS[part]) for part in PARTS])
            area.append(info['area'])

        target = landmarks[SMIRK_MEDIAPIPE_INDICES] @ M[:, :2].T + M[:, 2]
        return {'images': torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2),  # P x 3 x H x W, uint8
                'target': torch.from_numpy(target / CROP_SIZE * 2 - 1).float(),   # 105 x 2, in [-1, 1]
                'covered': torch.tensor(np.array(covered, dtype=np.float32).reshape(len(self.protocols), len(PARTS))),
                'area': torch.tensor(np.array(area, dtype=np.float32).reshape(len(self.protocols))),
                'index': i}


def load_encoder(checkpoint, device):
    encoder = SmirkEncoder().to(device).eval()
    state = torch.load(checkpoint, map_location=device)
    encoder.load_state_dict({k[len('smirk_encoder.'):]: v for k, v in state.items() if k.startswith('smirk_encoder.')})
    return encoder


@torch.no_grad()
def predict(encoder, flame, images):
    """images: N x 3 x H x W uint8. Returns the encoder outputs and the projected MediaPipe landmarks in [-1, 1]."""
    outputs = encoder(images.float().div(255))
    landmarks = batch_orth_proj(flame.forward(outputs)['landmarks_mp'], outputs['cam'])
    landmarks[:, :, 1:] = -landmarks[:, :, 1:]  # same convention as Renderer.forward / the training targets
    return outputs, landmarks[..., :2]


@torch.no_grad()
def expression_vertices(flame, shape, outputs):
    """FLAME vertices with a given shape, the predicted expression / jaw / eyelids and a neutral pose."""
    params = {'shape_params': shape, 'expression_params': outputs['expression_params'], 'jaw_params': outputs['jaw_params'],
              'eyelid_params': outputs['eyelid_params'], 'pose_params': torch.zeros_like(outputs['pose_params'])}
    return flame.forward(params)['vertices']


def features_of(outputs):
    return torch.cat([outputs['expression_params'], outputs['jaw_params'], outputs['eyelid_params']], dim=1).cpu().numpy()


def run_test(encoder, flame, loader, protocols, device):
    """Predictions and metrics of every test face for the clean image and every occlusion protocol."""
    names = ['clean'] + protocols
    n = len(loader.dataset)
    feats = {p: np.zeros((n, 55), np.float32) for p in names}
    metrics = {p: {m: np.full(n, np.nan, np.float32) for m in METRICS} for p in names}
    covered = np.zeros((n, len(protocols), len(PARTS)), np.float32)

    for batch in tqdm(loader, desc='test'):
        b, P = batch['images'].shape[:2]
        idx = batch['index'].numpy()
        images = batch['images'].flatten(0, 1).to(device)
        outputs, landmarks = predict(encoder, flame, images)
        outputs = {k: v.reshape(b, P, *v.shape[1:]) for k, v in outputs.items()}
        landmarks = landmarks.reshape(b, P, -1, 2)
        target = batch['target'].to(device)
        covered[idx] = batch['covered'].numpy()

        clean = {k: v[:, 0] for k, v in outputs.items()}
        clean_vertices = expression_vertices(flame, clean['shape_params'], clean)
        for j, protocol in enumerate(names):
            out = {k: v[:, j] for k, v in outputs.items()}
            feats[protocol][idx] = features_of(out)
            lmk_px = (landmarks[:, j] - target).norm(dim=-1).mean(-1) * CROP_SIZE / 2
            metrics[protocol]['lmk_px'][idx] = lmk_px.cpu().numpy()
            metrics[protocol]['occluded_area'][idx] = batch['area'][:, j - 1].numpy() if j > 0 else 0.0
            if j == 0:
                continue
            dist = (out['expression_params'] - clean['expression_params']).norm(dim=1)
            metrics[protocol]['exp_dist'][idx] = dist.cpu().numpy()
            clean_norm = clean['expression_params'].norm(dim=1).clamp(min=1e-6)
            metrics[protocol]['exp_rel'][idx] = (dist / clean_norm).cpu().numpy()
            metrics[protocol]['exp_scale'][idx] = (out['expression_params'].norm(dim=1) / clean_norm).cpu().numpy()
            vertices = expression_vertices(flame, clean['shape_params'], out)
            metrics[protocol]['expr_mm'][idx] = ((vertices - clean_vertices).norm(dim=-1).mean(-1) * 1000).cpu().numpy()
    return feats, metrics, covered


def run_features(encoder, flame, loader, device):
    """Clean-image features (for the emotion probe)."""
    feats = np.zeros((len(loader.dataset), 55), np.float32)
    for batch in tqdm(loader, desc='train features'):
        outputs, _ = predict(encoder, flame, batch['images'][:, 0].to(device))
        feats[batch['index'].numpy()] = features_of(outputs)
    return feats


def emotion_probe(train_feats, train_labels, test_feats):
    """MLP on clean train features (EMOCA / SMIRK protocol); returns predictions for every test feature set."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # balance the classes by resampling (MLPClassifier has no class weights)
    rng = np.random.default_rng(0)
    classes, counts = np.unique(train_labels, return_counts=True)
    index = np.concatenate([rng.choice(np.where(train_labels == c)[0], counts.max(), replace=True) for c in classes])
    train_feats, train_labels = train_feats[index], train_labels[index]

    probe = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=200,
                                                          early_stopping=True, random_state=0))
    probe.fit(train_feats, train_labels)
    return {p: probe.predict(f) for p, f in test_feats.items()}


def mean_class_accuracy(labels, predictions):
    classes = np.unique(labels)
    return float(np.mean([(predictions[labels == c] == c).mean() for c in classes]))


def main():
    parser = argparse.ArgumentParser(description='Occlusion benchmark of a SMIRK encoder.')
    parser.add_argument('--clean_dir', action='append', required=True, help='Folder with clean_meta.csv and landmarks.npz (repeatable)')
    parser.add_argument('--image_dir', action='append', required=True, help='dataset=folder with the crop images (repeatable)')
    parser.add_argument('--checkpoint', required=True, help='SMIRK checkpoint (keys smirk_encoder.*)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--protocols', default='random,mouth,eyes,brows,nose')
    parser.add_argument('--scale', type=float, default=1.6, help='SMIRK crop scale (test scale of the configs)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Use only N random test faces (quick check); 0 = all')
    parser.add_argument('--probe_train_max', type=int, default=30000, help='Max clean train faces for the emotion probe')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    protocols = [p for p in args.protocols.split(',') if p]
    assert all(p == 'random' or p in PARTS for p in protocols), f'protocols must be random or one of {PARTS}'
    image_dirs = dict(spec.split('=', 1) for spec in args.image_dir)

    metas, landmarks = [], {}
    for folder in args.clean_dir:
        metas.append(pd.read_csv(os.path.join(folder, 'clean_meta.csv'), dtype={'name': str}))
        data = np.load(os.path.join(folder, 'landmarks.npz'))
        landmarks.update(zip(data['keys'], data['landmarks']))
    meta = pd.concat(metas, ignore_index=True)
    meta = meta[meta['dataset'].isin(list(image_dirs))]
    print('faces per dataset / split:\n', pd.crosstab(meta['dataset'], meta['split']))

    test = meta[meta['split'] == 'test']
    if args.limit:
        test = test.sample(min(args.limit, len(test)), random_state=0)
    train = meta[meta['split'] == 'train']
    if len(train) > args.probe_train_max:
        train = train.sample(args.probe_train_max, random_state=0)

    encoder = load_encoder(args.checkpoint, args.device)
    flame = FLAME().to(args.device)
    loader = lambda rows, protocols: torch.utils.data.DataLoader(
        FaceCrops(rows, image_dirs, landmarks, protocols, args.scale), batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False)

    test_feats, metrics, covered = run_test(encoder, flame, loader(test, protocols), protocols, args.device)
    train_feats = run_features(encoder, flame, loader(train, []), args.device)
    predictions = emotion_probe(train_feats, train['label'].to_numpy(), test_feats)

    # per-image table
    test = test.reset_index(drop=True)
    rows = []
    for p, protocol in enumerate(['clean'] + protocols):
        table = test[['dataset', 'name', 'label', 'expression']].copy()
        table['protocol'] = protocol
        for m, values in metrics[protocol].items():
            table[m] = values
        table['pred_label'] = predictions[protocol]
        for k, part in enumerate(PARTS):
            table[f'covered_{part}'] = covered[:, p - 1, k] if p > 0 else 0.0
        rows.append(table)
    per_image = pd.concat(rows, ignore_index=True)

    # summary: metrics per protocol (all datasets and per dataset), emotion probe on the 7 basic expressions
    summary = []
    for (dataset, protocol), group in [(('all', p), g) for p, g in per_image.groupby('protocol', sort=False)] + \
                                       list(per_image.groupby(['dataset', 'protocol'], sort=False)):
        summary.append({'dataset': dataset, 'protocol': protocol, 'faces': len(group),
                        **{m: float(group[m].mean()) for m in METRICS},
                        'emotion_classes': group['label'].nunique(),
                        'emotion_acc': float((group['pred_label'] == group['label']).mean()),
                        'emotion_mean_class_acc': mean_class_accuracy(group['label'].to_numpy(), group['pred_label'].to_numpy())})
    summary = pd.DataFrame(summary)

    # change of every paper-style score with respect to the clean images of the same dataset
    clean = summary[summary['protocol'] == 'clean'].set_index('dataset')
    for column in ['emotion_acc', 'emotion_mean_class_acc', 'lmk_px']:
        reference = summary['dataset'].map(clean[column])
        summary[f'{column}_change'] = summary[column] - reference
        summary[f'{column}_change_pct'] = 100 * (summary[column] - reference) / reference

    os.makedirs(args.out_dir, exist_ok=True)
    per_image.to_csv(os.path.join(args.out_dir, 'per_image.csv'), index=False)
    summary.to_csv(os.path.join(args.out_dir, 'summary.csv'), index=False)
    np.savez_compressed(os.path.join(args.out_dir, 'features.npz'), train=train_feats, train_labels=train['label'].to_numpy(),
                        **{f'test_{p}': f for p, f in test_feats.items()})
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump({**vars(args), 'protocols': protocols, 'test_faces': len(test), 'probe_train_faces': len(train)}, f, indent=1)

    pd.set_option('display.width', 200)
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
