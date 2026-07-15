"""Original copy-move forgery detection — block-based DCT feature
matching with shift-vector voting, following the spirit of Fridrich,
Soukal & Lukas (2003) "Detection of Copy-Move Forgery in Digital
Images." Detects and spatially localizes duplicated/cloned regions
within a single image: one of the most classic and well-regarded
techniques in image forensics, and one most "AI wrapper" tools skip
entirely because it has nothing to do with classification — it's pure
geometric/statistical detective work.

Method:
1. Slide overlapping blocks across the image, discard flat/low-texture
   blocks (they trivially "match" everywhere and produce noise).
2. Represent each block by its low-frequency DCT coefficients — a
   compact, illumination-robust feature vector.
3. Lexicographically sort blocks by feature vector so near-duplicate
   blocks land close together, then compare each block only to a small
   neighbourhood in sorted order (avoids an O(n^2) full comparison).
4. For matching block pairs, compute the spatial shift vector between
   their positions and vote in a shift-vector histogram. A cloned
   region produces many block pairs sharing the *same* shift vector —
   that consistency is the forgery signature, not any single match.

Fully self-implemented with numpy/opencv; no external models.
"""
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image, ImageDraw


def _resize_for_speed(gray: np.ndarray, max_dim: int = 640):
    h, w = gray.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return gray, scale


def _zigzag_indices(n: int):
    indices = [(i, j) for i in range(n) for j in range(n)]
    indices.sort(key=lambda p: (p[0] + p[1], p[0] if (p[0] + p[1]) % 2 == 0 else -p[0]))
    return indices


def _block_features(gray: np.ndarray, block: int, stride: int, n_coeffs: int, var_thresh: float):
    h, w = gray.shape
    zigzag = _zigzag_indices(block)[:n_coeffs]
    feats, positions = [], []
    for y in range(0, h - block + 1, stride):
        for x in range(0, w - block + 1, stride):
            patch = gray[y:y + block, x:x + block].astype(np.float32)
            if patch.var() < var_thresh:
                continue
            d = cv2.dct(patch)
            feats.append(np.array([d[i, j] for (i, j) in zigzag]))
            positions.append((x, y))
    return np.array(feats) if feats else np.empty((0, n_coeffs)), positions


def copy_move_analysis(image: Image.Image, block: int = 16, stride: int = 8,
                        max_dim: int = 640, match_thresh: float = 2.2,
                        min_shift: int = 24, neighbor_window: int = 14,
                        min_cluster: int = 6) -> dict:
    gray = np.array(image.convert("L")).astype(np.float32)
    gray, scale = _resize_for_speed(gray, max_dim)
    h, w = gray.shape

    if h < block * 3 or w < block * 3:
        return {"score": 0.0, "insufficient_data": True, "match_count": 0,
                "total_blocks": 0, "dominant_shift": None, "overlay_image": None}

    feats, positions = _block_features(gray, block, stride, n_coeffs=10, var_thresh=15.0)
    if len(feats) < 30:
        return {"score": 0.0, "insufficient_data": True, "match_count": 0,
                "total_blocks": 0, "dominant_shift": None, "overlay_image": None}

    order = np.lexsort(feats.T[::-1])
    sorted_feats = feats[order]
    sorted_pos = [positions[i] for i in order]

    shift_votes = defaultdict(list)
    n = len(sorted_feats)
    for i in range(n):
        vi = sorted_feats[i]
        for j in range(i + 1, min(i + 1 + neighbor_window, n)):
            dist = float(np.linalg.norm(vi - sorted_feats[j]))
            if dist > match_thresh:
                break  # sorted order -> distances trend upward; bounds runtime
            (x1, y1), (x2, y2) = sorted_pos[i], sorted_pos[j]
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < min_shift and abs(dy) < min_shift:
                continue
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
                p1, p2 = (x2, y2), (x1, y1)
            else:
                p1, p2 = (x1, y1), (x2, y2)
            shift_votes[(dx, dy)].append((p1, p2))

    if not shift_votes:
        return {"score": 0.0, "insufficient_data": False, "match_count": 0,
                "total_blocks": len(feats), "dominant_shift": None, "overlay_image": None}

    best_shift = max(shift_votes, key=lambda k: len(shift_votes[k]))
    matches = shift_votes[best_shift]
    match_count = len(matches)
    total_blocks = len(feats)

    # Saturating curve on absolute match count rather than a ratio to
    # total block count: a forged region's match count doesn't scale
    # with overall image detail/size, so a ratio-based score under-
    # weights forgeries in busy, high-detail images.
    score = float(np.clip(100 * (1 - np.exp(-match_count / 15.0)), 0, 100))

    overlay_image = None
    if match_count >= min_cluster:
        rgb = np.array(image.convert("RGB").resize((w, h)))
        overlay_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(overlay_img)
        for (p1, p2) in matches[:500]:
            draw.rectangle([p1[0], p1[1], p1[0] + block, p1[1] + block], outline=(255, 51, 102), width=1)
            draw.rectangle([p2[0], p2[1], p2[0] + block, p2[1] + block], outline=(0, 255, 179), width=1)
            draw.line([p1[0] + block // 2, p1[1] + block // 2,
                       p2[0] + block // 2, p2[1] + block // 2], fill=(0, 184, 255), width=1)
        overlay_image = overlay_img

    return {
        "score": score,
        "insufficient_data": False,
        "match_count": match_count,
        "total_blocks": total_blocks,
        "dominant_shift": best_shift,
        "overlay_image": overlay_image,
    }
