"""
Synthetic occlusions of facial parts, placed with the MediaPipe landmarks (478 points) of the clean image.

For every occluder the part (mouth, eye(s), eyebrow(s), nose), the shape, its size, rotation, position jitter and its
appearance (colour, gradient, noise, pattern, skin-like tone or a patch of another image) are drawn at random.
The position is jittered and the size varies, so that the occluder never marks the exact location of the part.

Usage:
    occluder = FaceOccluder()
    occluded, mask, info = occluder(image, landmarks, rng=np.random.default_rng(seed))
    # image: HxWx3 uint8 RGB, landmarks: 478x2 (or 478x3) pixel coordinates in the same image
    # mask: HxW float32 in [0, 1] (1 = occluded), info: parts / shapes / fills, the covered fraction of every part and
    # 'area': the occluded fraction of the face (convex hull of all the landmarks)

For a test set, use occlusion_rng(name) so that every image always gets the same occlusion.
"""
import zlib
import numpy as np
import cv2

# MediaPipe FaceMesh landmark indices of the facial parts ("left" / "right" are the subject's sides)
PART_LANDMARKS = {
    'left_eye': [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398],
    'right_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173],
    'left_brow': [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
    'right_brow': [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
    'nose': [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 129, 358, 49, 279, 64, 294],
    'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
}

# parts that can be chosen, and the landmark groups each choice covers
PART_CHOICES = {
    'mouth': [['mouth']],
    'eyes': [['left_eye'], ['right_eye'], ['left_eye', 'right_eye']],
    'brows': [['left_brow'], ['right_brow'], ['left_brow', 'right_brow']],
    'nose': [['nose']],
}

SHAPES = ['ellipse', 'rectangle', 'polygon', 'blob', 'bar']
FILLS = ['solid', 'gradient', 'noise', 'pattern', 'skin', 'texture']


def occlusion_rng(name, seed=0):
    """Deterministic random generator for one image (fixed occlusions of a test set)."""
    return np.random.default_rng([zlib.crc32(str(name).encode()), seed])


def _shape_polygon(shape, rng, n_points=64):
    """Outline of a shape in a unit frame (radius ~1, centred on 0)."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    if shape == 'ellipse':
        return np.stack([np.cos(t), np.sin(t)], 1)
    if shape == 'rectangle':
        return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    if shape == 'bar':  # long thin band (e.g. sunglasses band, strand of hair)
        return np.array([[-1, -0.3], [1, -0.3], [1, 0.3], [-1, 0.3]], dtype=np.float64)
    if shape == 'polygon':  # random polygon with 5-9 corners
        k = rng.integers(5, 10)
        angles = np.sort(rng.uniform(0, 2 * np.pi, k))
        radii = rng.uniform(0.6, 1.1, k)
        return np.stack([radii * np.cos(angles), radii * np.sin(angles)], 1)
    if shape == 'blob':  # smooth irregular outline: radius modulated by a few low-frequency harmonics
        radius = np.ones_like(t)
        for freq in range(2, 5):
            radius += rng.uniform(-0.18, 0.18) * np.cos(freq * t + rng.uniform(0, 2 * np.pi))
        return np.stack([radius * np.cos(t), radius * np.sin(t)], 1)
    raise ValueError(f'unknown shape {shape}')


def _fill(fill, height, width, rng, image, texture_pool):
    """HxWx3 float32 appearance of an occluder."""
    if fill == 'texture' and texture_pool:
        source = texture_pool[rng.integers(len(texture_pool))]
        sh, sw = source.shape[:2]
        ch, cw = max(8, int(sh * rng.uniform(0.2, 0.6))), max(8, int(sw * rng.uniform(0.2, 0.6)))
        y, x = rng.integers(0, sh - ch + 1), rng.integers(0, sw - cw + 1)
        patch = cv2.resize(source[y:y + ch, x:x + cw], (width, height), interpolation=cv2.INTER_LINEAR)
        return patch.astype(np.float32)
    if fill == 'texture':  # no texture pool: fall back to a random colour
        fill = 'solid'

    color = rng.uniform(0, 255, 3)
    if fill == 'solid':
        return np.broadcast_to(color, (height, width, 3)).astype(np.float32)
    if fill == 'skin':  # tone sampled around the image centre (the face), slightly changed: hand-like occluders
        cy, cx = image.shape[0] // 2, image.shape[1] // 2
        r = max(2, min(image.shape[:2]) // 16)
        tone = image[cy - r:cy + r, cx - r:cx + r].reshape(-1, 3).mean(0) * rng.uniform(0.8, 1.15)
        return np.broadcast_to(np.clip(tone, 0, 255), (height, width, 3)).astype(np.float32)
    if fill == 'gradient':
        other = rng.uniform(0, 255, 3)
        angle = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:height, 0:width]
        proj = np.cos(angle) * xx / max(width, 1) + np.sin(angle) * yy / max(height, 1)
        proj = (proj - proj.min()) / max(proj.max() - proj.min(), 1e-6)
        return (color * (1 - proj[..., None]) + other * proj[..., None]).astype(np.float32)
    if fill == 'noise':
        noise = rng.normal(0, rng.uniform(20, 70), (height, width, 3))
        return np.clip(color + noise, 0, 255).astype(np.float32)
    if fill == 'pattern':  # stripes or checkerboard of two colours
        other = rng.uniform(0, 255, 3)
        period = rng.integers(4, 16)
        yy, xx = np.mgrid[0:height, 0:width]
        if rng.random() < 0.5:
            angle = rng.uniform(0, np.pi)
            select = ((np.cos(angle) * xx + np.sin(angle) * yy) // period).astype(int) % 2
        else:
            select = (xx // period + yy // period) % 2
        return np.where(select[..., None] == 1, other, color).astype(np.float32)
    raise ValueError(f'unknown fill {fill}')


def occluded_face_area(mask, landmarks):
    """Fraction of the face (convex hull of the landmarks, i.e. the face oval for MediaPipe) covered by the mask."""
    face = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(face, cv2.convexHull(np.round(np.asarray(landmarks)[:, :2]).astype(np.int32)), 1)
    face = face.astype(bool)
    return float(mask[face].mean()) if face.any() else 0.0


class FaceOccluder:
    def __init__(self, parts=('mouth', 'eyes', 'brows', 'nose'), max_parts=2, shapes=SHAPES, fills=FILLS,
                 scale_range=(0.8, 1.8), jitter=0.25, feather=1.5, texture_pool=None):
        """
        parts:        parts that can be occluded (keys of PART_CHOICES)
        max_parts:    every call occludes between 1 and max_parts different parts
        scale_range:  occluder size relative to the part size (< 1: partial occlusion)
        jitter:       maximum shift of the occluder centre, as a fraction of the part size
        feather:      Gaussian blur (pixels) of the occluder edge, so that edges are not perfectly sharp
        texture_pool: optional list of HxWx3 uint8 RGB images to cut 'texture' occluders from
        """
        self.parts = list(parts)
        self.max_parts = max_parts
        self.shapes, self.fills = list(shapes), list(fills)
        self.scale_range, self.jitter, self.feather = scale_range, jitter, feather
        self.texture_pool = texture_pool

    def __call__(self, image, landmarks, rng=None, parts=None):
        """parts: force the occluded parts (e.g. ['mouth']) instead of drawing them at random."""
        rng = rng if rng is not None else np.random.default_rng()
        landmarks = np.asarray(landmarks)[:, :2]
        height, width = image.shape[:2]
        out = image.astype(np.float32)
        total_mask = np.zeros((height, width), np.float32)

        if parts is None:
            n = rng.integers(1, min(self.max_parts, len(self.parts)) + 1)
            parts = list(rng.choice(self.parts, n, replace=False))
        info = {'parts': [], 'shapes': [], 'fills': []}

        for part in parts:
            choices = PART_CHOICES[part]
            groups = choices[rng.integers(len(choices))]
            points = landmarks[sum((PART_LANDMARKS[g] for g in groups), [])]

            # occluder frame: centre / size of the part, random scale, rotation and shift
            lo, hi = points.min(0), points.max(0)
            center, size = (lo + hi) / 2, np.maximum(hi - lo, 4.0)
            shape = self.shapes[rng.integers(len(self.shapes))]
            scale = rng.uniform(*self.scale_range)
            half = size / 2 * scale
            if shape == 'bar':  # bars are longer than the part, and not thinner than ~15% of its width
                half = np.array([half[0] * 1.3, max(half[1], 0.5 * half[0])])
            center = center + rng.uniform(-self.jitter, self.jitter, 2) * size
            angle = rng.uniform(-np.pi / 6, np.pi / 6)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            outline = (_shape_polygon(shape, rng) * half) @ rotation.T + center

            mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask, [np.round(outline).astype(np.int32)], 255, lineType=cv2.LINE_AA)
            mask = mask.astype(np.float32) / 255
            if self.feather > 0:
                mask = cv2.GaussianBlur(mask, (0, 0), self.feather)
            if mask.max() == 0:
                continue

            fill = self.fills[rng.integers(len(self.fills))]
            appearance = _fill(fill, height, width, rng, image, self.texture_pool)
            out = out * (1 - mask[..., None]) + appearance * mask[..., None]
            total_mask = np.maximum(total_mask, mask)
            info['parts'].append('+'.join(groups))
            info['shapes'].append(shape)
            info['fills'].append(fill)

        # fraction of the landmarks of every part that ends up under the occluders
        inside = total_mask[np.clip(np.round(landmarks[:, 1]).astype(int), 0, height - 1),
                            np.clip(np.round(landmarks[:, 0]).astype(int), 0, width - 1)] > 0.5
        info['covered'] = {g: float(inside[idx].mean()) for g, idx in PART_LANDMARKS.items()}
        info['area'] = occluded_face_area(total_mask, landmarks)
        return np.clip(out, 0, 255).astype(np.uint8), total_mask, info
