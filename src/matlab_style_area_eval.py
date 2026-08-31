import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_fill_holes
from sklearn.cluster import MiniBatchKMeans

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

# Configure aqui um caminho padrao para sua imagem base (opcional).
BASE_IMAGE_PATH = r""


def read_image(path: str) -> np.ndarray:
    resolved = Path(os.path.expandvars(path)).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Arquivo de imagem nao encontrado: {resolved}\n"
            "Informe um caminho valido com --base."
        )

    ext = resolved.suffix.lower()
    if ext in (".tif", ".tiff"):
        img = tiff.imread(str(resolved))
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[2] not in (3, 4):
            img = np.transpose(img, (1, 2, 0))
        return img

    # cv2.imread pode falhar em caminhos Unicode/OneDrive no Windows.
    raw_bytes = b""
    try:
        raw_bytes = resolved.read_bytes()
    except Exception:
        raw_bytes = b""

    if raw_bytes:
        img = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        try:
            buffer = np.fromfile(str(resolved), dtype=np.uint8)
        except Exception:
            buffer = np.empty((0,), dtype=np.uint8)
        img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED) if buffer.size > 0 else None

    if img is None:
        raise FileNotFoundError(f"Nao consegui ler: {resolved}")

    if img.ndim == 3 and img.shape[2] in (3, 4):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def mat2gray(img: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = img.astype(np.float32, copy=False)
    mn = float(np.min(x))
    mx = float(np.max(x))
    return (x - mn) / (mx - mn + eps)


def as_odd(value: int, min_value: int = 1) -> int:
    value = max(min_value, int(value))
    return value if (value % 2) == 1 else value + 1


def to_uint8_01(img01: np.ndarray) -> np.ndarray:
    return np.clip(mat2gray(img01) * 255.0, 0, 255).astype(np.uint8)


def ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] >= 3:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    raise ValueError("Formato de imagem nao suportado para conversao em cinza.")


def ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        rgb = cv2.cvtColor(to_uint8_01(img), cv2.COLOR_GRAY2RGB)
    else:
        rgb = img[:, :, :3].copy()
        if rgb.dtype != np.uint8:
            rgb = to_uint8_01(rgb)
    return rgb


def pregray_lab_enhance(
    img: np.ndarray,
    enabled: int = 1,
    clip_limit: float = 3.0,
    alpha: float = 1.12,
    beta: float = 8.0,
    use_local_sharpen: int = 1,
) -> np.ndarray:
    if int(enabled) <= 0:
        return img
    if img.ndim != 3 or img.shape[2] < 3:
        return img

    rgb_u8 = ensure_rgb_uint8(img)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=max(0.01, float(clip_limit)),
        tileGridSize=(8, 8),
    )
    l_eq = clahe.apply(l)
    l_eq = cv2.convertScaleAbs(
        l_eq,
        alpha=max(0.1, float(alpha)),
        beta=float(beta),
    )

    if int(use_local_sharpen) > 0:
        l32 = l_eq.astype(np.float32) / 255.0
        gx = cv2.Sobel(l32, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(l32, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        grad = mat2gray(grad)

        blur = cv2.GaussianBlur(
            l32, (0, 0), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REPLICATE
        )
        detail = l32 - blur

        l32 = l32 + 0.35 * detail + 0.18 * grad
        l_eq = np.clip(255.0 * mat2gray(l32), 0, 255).astype(np.uint8)

    lab_out = cv2.merge((l_eq, a, b))
    rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
    return rgb_out


def homomorphic_filter(
    img01: np.ndarray,
    sigma: float = 24.0,
    ksize: int = 121,
) -> np.ndarray:
    x = mat2gray(img01).astype(np.float32, copy=False)
    img_log = np.log1p(x)

    ksize = as_odd(ksize, min_value=3)

    illumination = cv2.GaussianBlur(
        img_log,
        (ksize, ksize),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REPLICATE,
    )
    reflectance = img_log - illumination
    img_homo = np.exp(reflectance)
    return mat2gray(img_homo)


def clahe_matlab_like(
    img01: np.ndarray,
    num_tiles: Tuple[int, int] = (24, 24),
    clip_limit: float = 0.08,
) -> np.ndarray:
    img_u8 = to_uint8_01(img01)
    tiles_y = max(2, int(num_tiles[0]))
    tiles_x = max(2, int(num_tiles[1]))
    cv_clip = max(0.01, float(clip_limit) * 25.0)
    clahe = cv2.createCLAHE(clipLimit=cv_clip, tileGridSize=(tiles_x, tiles_y))
    out_u8 = clahe.apply(img_u8)
    return out_u8.astype(np.float32) / 255.0


def edge_contrast_boost(
    img01: np.ndarray,
    unsharp_amount: float = 1.10,
    unsharp_sigma: float = 1.20,
    grad_weight: float = 0.45,
) -> np.ndarray:
    src = mat2gray(img01).astype(np.float32, copy=False)

    sigma = max(0.2, float(unsharp_sigma))
    blurred = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    detail = src - blurred
    sharp = src + (float(unsharp_amount) * detail)

    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    grad = mat2gray(cv2.magnitude(gx, gy))
    boosted = sharp + (float(grad_weight) * grad)
    return mat2gray(np.clip(boosted, 0.0, 1.0))


def sobel_edge_refine(
    img01: np.ndarray,
    sobel_weight: float = 0.34,
    unsharp_amount: float = 0.90,
    unsharp_sigma: float = 1.10,
    x_weight: float = 1.00,
    y_weight: float = 0.30,
) -> np.ndarray:
    src = mat2gray(img01).astype(np.float32, copy=False)

    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)

    grad = np.abs(float(x_weight) * gx) + np.abs(float(y_weight) * gy)
    grad = mat2gray(grad)

    blur = cv2.GaussianBlur(
        src,
        (0, 0),
        sigmaX=max(0.2, float(unsharp_sigma)),
        sigmaY=max(0.2, float(unsharp_sigma)),
        borderType=cv2.BORDER_REPLICATE,
    )
    detail = src - blur

    out = src + float(unsharp_amount) * detail + float(sobel_weight) * grad
    return mat2gray(np.clip(out, 0.0, 1.0))


@contextmanager
def limit_native_threads(max_cores: int):
    prev_cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        if threadpool_limits is None:
            yield
        else:
            with threadpool_limits(limits=int(max_cores)):
                yield
    finally:
        cv2.setNumThreads(prev_cv_threads)


def gabor_response(
    src01: np.ndarray,
    lam: float,
    ang_deg: float,
    ksize: int,
    smooth_sigma_factor: float,
    gamma: float,
    phase_offset: float = 0.0,
) -> np.ndarray:
    lam = float(lam)
    sigma = max(1.0, float(smooth_sigma_factor) * lam)
    theta = np.deg2rad(float(ang_deg))
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, float(gamma), float(phase_offset), ktype=cv2.CV_32F)
    resp = cv2.filter2D(src01, cv2.CV_32F, kernel, borderType=cv2.BORDER_REPLICATE)
    mag = np.abs(resp)
    mag = cv2.GaussianBlur(
        mag,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    return mag.astype(np.float32, copy=False)


def morph_matlab_like(
    img01: np.ndarray,
    radius: int = 30,
    a_top: float = 9.4,
    a_both: float = 0.72,
    return_hat_energy: bool = False,
):
    src_u8 = to_uint8_01(img01)
    r = max(1, int(radius))
    k = 2 * r + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    toph = cv2.morphologyEx(src_u8, cv2.MORPH_TOPHAT, se)
    both = cv2.morphologyEx(src_u8, cv2.MORPH_BLACKHAT, se)

    src01 = src_u8.astype(np.float32) / 255.0
    toph01 = toph.astype(np.float32) / 255.0
    both01 = both.astype(np.float32) / 255.0

    out = mat2gray(src01 + (float(a_top) * toph01) - (float(a_both) * both01))
    hat_energy = mat2gray((0.62 * toph01) + (0.38 * both01))

    if return_hat_energy:
        return out, hat_energy
    return out


def gabor_feature_stack(
    img01: np.ndarray,
    wavelengths: Sequence[float],
    orientations_deg: Sequence[float],
    ksize: int = 31,
    smooth_sigma_factor: float = 0.5,
    gamma: float = 0.5,
    phase_offset: float = 0.0,
    parallel_workers: int = 1,
) -> np.ndarray:
    src = mat2gray(img01).astype(np.float32, copy=False)
    jobs = [(float(lam), float(ang)) for lam in wavelengths for ang in orientations_deg]
    if len(jobs) == 0:
        raise ValueError("Banco de Gabor vazio: informe ao menos 1 wavelength e 1 orientacao.")

    ksize = as_odd(ksize, min_value=3)

    workers = max(1, int(parallel_workers))
    if workers == 1 or len(jobs) == 1:
        feats = [gabor_response(src, lam, ang, ksize, smooth_sigma_factor, gamma, phase_offset=phase_offset) for lam, ang in jobs]
    else:
        workers = min(workers, len(jobs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(gabor_response, src, lam, ang, ksize, smooth_sigma_factor, gamma, phase_offset)
                for lam, ang in jobs
            ]
            feats = [f.result() for f in futures]

    return np.stack(feats, axis=2)


def texture_score_map(img01: np.ndarray, local_win: int = 9) -> np.ndarray:
    src = mat2gray(img01).astype(np.float32, copy=False)

    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)

    grad = np.abs(1.0 * gx) + np.abs(0.35 * gy)
    grad = mat2gray(grad)

    win = as_odd(local_win, min_value=3)
    mean = cv2.blur(src, (win, win))
    mean2 = cv2.blur(src * src, (win, win))
    var = np.maximum(mean2 - (mean * mean), 0.0)
    local_std = mat2gray(np.sqrt(var).astype(np.float32))

    score = 0.68 * grad + 0.32 * local_std
    return mat2gray(score.astype(np.float32))


def local_variance_map(img01: np.ndarray, local_win: int = 9) -> np.ndarray:
    src = mat2gray(img01).astype(np.float32, copy=False)
    win = as_odd(local_win, min_value=3)
    mean = cv2.blur(src, (win, win))
    mean2 = cv2.blur(src * src, (win, win))
    var = np.maximum(mean2 - (mean * mean), 0.0)
    return mat2gray(var.astype(np.float32, copy=False))


def local_std_map(img01: np.ndarray, local_win: int) -> np.ndarray:
    src = mat2gray(img01).astype(np.float32, copy=False)
    win = as_odd(local_win, min_value=3)
    mean = cv2.blur(src, (win, win))
    mean2 = cv2.blur(src * src, (win, win))
    var = np.maximum(mean2 - (mean * mean), 0.0)
    return mat2gray(np.sqrt(var).astype(np.float32))


def pick_wound_label(
    labels: np.ndarray,
    dist_center: np.ndarray,
    texture_score: np.ndarray,
    center_window_frac: float = 0.14,
) -> int:
    h, w = labels.shape
    r = max(3, int(round(min(h, w) * float(center_window_frac) * 0.5)))
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    center_patch = labels[y0:y1, x0:x1]

    unique_labels = np.unique(labels)
    best_label = int(unique_labels[0])
    best_score = float("inf")

    for lb in unique_labels:
        m = labels == lb
        if not np.any(m):
            continue

        mean_dist = float(np.mean(dist_center[m]))
        mean_tex = float(np.mean(texture_score[m]))
        center_ratio = float(np.mean(center_patch == lb))
        score = (1.35 * mean_dist) + (2.10 * mean_tex) - (0.40 * center_ratio)

        if score < best_score:
            best_score = score
            best_label = int(lb)

    return best_label


def _otsu_center_mask(img01: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    src_u8 = to_uint8_01(img01)
    _, thr = cv2.threshold(src_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_bright = (thr > 0) & roi_mask
    mask_dark = (~mask_bright) & roi_mask

    h, w = src_u8.shape
    cy, cx = h // 2, w // 2
    r = max(3, int(round(min(h, w) * 0.07)))
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    center_roi = roi_mask[y0:y1, x0:x1]
    if not np.any(center_roi):
        return mask_dark

    center_dark = float(np.mean(mask_dark[y0:y1, x0:x1][center_roi]))
    center_bright = float(np.mean(mask_bright[y0:y1, x0:x1][center_roi]))
    return mask_dark if center_dark >= center_bright else mask_bright


def _mask_border_texture_support(mask: np.ndarray, texture_score: np.ndarray, ring_radius: int = 4) -> float:
    if not np.any(mask):
        return 0.0

    rk = as_odd((2 * max(1, int(ring_radius))) + 1, min_value=3)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rk, rk))
    mask_u8 = mask.astype(np.uint8) * 255
    dil = cv2.dilate(mask_u8, se)
    er = cv2.erode(mask_u8, se)
    ring = (cv2.subtract(dil, er) > 0) & (~mask.astype(bool))
    if not np.any(ring):
        return 0.0
    return float(np.mean(texture_score[ring]))


def _mask_compact_blob_penalty(mask: np.ndarray) -> float:
    if not np.any(mask):
        return 1.0

    ys, xs = np.where(mask)
    h = mask.shape[0]
    w = mask.shape[1]
    area = float(xs.size)
    bbox_h = float(ys.max() - ys.min() + 1)
    bbox_w = float(xs.max() - xs.min() + 1)
    bbox_area = max(bbox_h * bbox_w, 1.0)

    area_ratio = area / max(float(mask.size), 1.0)
    bbox_fill = area / bbox_area
    width_ratio = bbox_w / max(float(w), 1.0)
    height_ratio = bbox_h / max(float(h), 1.0)

    penalty = 0.0
    if area_ratio < 0.010:
        penalty += 0.55
    if area_ratio < 0.020:
        penalty += 0.25
    if width_ratio < 0.12:
        penalty += 0.40
    if height_ratio < 0.10:
        penalty += 0.25
    if bbox_fill > 0.72 and area_ratio < 0.030:
        penalty += 0.35
    return penalty


def _mask_lateral_edge_penalty(mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0

    h, w = mask.shape
    band_w = max(3, int(round(0.10 * w)))
    center_x0 = max(0, int(round(0.25 * w)))
    center_x1 = min(w, int(round(0.75 * w)))

    left_cov = float(np.mean(mask[:, :band_w]))
    right_cov = float(np.mean(mask[:, max(0, w - band_w):]))
    center_cov = float(np.mean(mask[:, center_x0:center_x1])) if center_x1 > center_x0 else 0.0
    side_cov = max(left_cov, right_cov)
    side_imbalance = abs(left_cov - right_cov)

    penalty = 0.0
    if side_cov > max(0.12, 1.8 * center_cov):
        penalty += min((side_cov - center_cov) * 2.2, 0.85)
    if side_imbalance > 0.10 and center_cov < 0.10:
        penalty += min(side_imbalance * 2.5, 0.65)
    return penalty


def _mask_quality_score(mask: np.ndarray, dist_center: np.ndarray, texture_score: np.ndarray) -> float:
    if not np.any(mask):
        return float("inf")
    mean_dist = float(np.mean(dist_center[mask]))
    mean_tex = float(np.mean(texture_score[mask]))
    border_tex = _mask_border_texture_support(mask, texture_score)
    compact_penalty = _mask_compact_blob_penalty(mask)
    lateral_edge_penalty = _mask_lateral_edge_penalty(mask)
    area = float(np.count_nonzero(mask))
    total = float(mask.size)
    area_ratio = area / max(total, 1.0)
    area_penalty = 0.0
    if area_ratio < 0.003:
        area_penalty += 0.35
    if area_ratio > 0.65:
        area_penalty += 0.55
    return (
        (1.15 * mean_dist)
        + (1.85 * mean_tex)
        - (1.35 * border_tex)
        + area_penalty
        + compact_penalty
        + lateral_edge_penalty
    )


def smooth_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    r = max(0, int(radius))
    if r == 0:
        return mask.astype(bool)

    k = 2 * r + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = (mask.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se)
    return m > 0


def imsegkmeans_like(
    feature_set: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 0,
    batch_size: int = 8192,
    max_iter: int = 200,
) -> np.ndarray:
    h, w, c = feature_set.shape
    x = feature_set.reshape(-1, c).astype(np.float32, copy=False)

    mu = x.mean(axis=0, keepdims=True, dtype=np.float32)
    sd = x.std(axis=0, keepdims=True, dtype=np.float32)
    sd = np.maximum(sd, 1e-6)
    xn = (x - mu) / sd

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init=1,
        random_state=random_state,
        batch_size=min(int(batch_size), xn.shape[0]),
        max_iter=int(max_iter),
    )
    labels = km.fit_predict(xn)
    return labels.reshape(h, w)


def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    bin255 = mask.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin255, connectivity=8, ltype=cv2.CV_32S)
    if n <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def remove_small_objects_compat(mask: np.ndarray, min_area: int) -> np.ndarray:
    bin255 = mask.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin255, connectivity=8, ltype=cv2.CV_32S)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)

    keep_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= int(min_area))[0] + 1
    if keep_labels.size == 0:
        return np.zeros_like(mask, dtype=bool)

    lut = np.zeros(n, dtype=np.uint8)
    lut[keep_labels] = 1
    return lut[labels].astype(bool)


def vertical_scratch_cleanup(mask: np.ndarray) -> np.ndarray:
    m = (mask.astype(np.uint8) * 255)

    se_close_v = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 31))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se_close_v)

    se_open_h = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se_open_h)

    se_close_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se_close_e)

    return m > 0


def postprocess_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    out = binary_fill_holes(mask.astype(bool))
    out = remove_small_objects_compat(out, min_area=int(min_area))
    out = keep_largest_cc(out)

    out = vertical_scratch_cleanup(out)
    # Fecha ilhas e recortes internos para evitar que o interior da lesao
    # seja segmentado como multiplas regioes separadas.
    se_close_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    out = cv2.morphologyEx((out.astype(np.uint8) * 255), cv2.MORPH_CLOSE, se_close_inner) > 0
    out = binary_fill_holes(out)
    se_smooth_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    out = cv2.morphologyEx((out.astype(np.uint8) * 255), cv2.MORPH_OPEN, se_smooth_inner) > 0
    out = binary_fill_holes(out)
    out = keep_largest_cc(out)
    out = smooth_mask(out, radius=2)
    out = binary_fill_holes(out)
    out = keep_largest_cc(out)

    return out.astype(bool)


def overlay_perimeter(img_rgb_u8: np.ndarray, perim_mask: np.ndarray, color_rgb_255: Tuple[int, int, int]) -> np.ndarray:
    out = img_rgb_u8.copy()
    out[perim_mask] = np.array(color_rgb_255, dtype=np.uint8)
    return out


def perimeter_mask(mask: np.ndarray) -> np.ndarray:
    bin255 = mask.astype(np.uint8) * 255
    er = cv2.erode(bin255, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    per = cv2.subtract(bin255, er)
    return per > 0


def labeloverlay_like(gray01: np.ndarray, mask: np.ndarray, color_rgb01=(1.0, 0.0, 0.0), alpha: float = 0.55) -> np.ndarray:
    base = np.dstack([gray01, gray01, gray01]).astype(np.float32)
    out = base.copy()
    color = np.array(color_rgb01, dtype=np.float32).reshape(1, 1, 3)
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * base[m] + alpha * color
    return np.clip(out, 0.0, 1.0)


@dataclass(frozen=True)
class AreaEvalConfig:
    r_auto: int = 2
    col_auto: Tuple[int, int, int] = (255, 0, 0)
    frac_min_area: float = 0.0005
    min_area_floor: int = 200
    min_area_fixed: int = 5000
    proc_scale: float = 0.75

    gabor_wavelengths: Tuple[float, ...] = (3, 5, 9, 17, 33, 65, 97)
    gabor_orientations_deg: Tuple[float, ...] = (0, 20, 40, 60, 90, 120, 140, 160)

    center_dist_weight: float = 2.6
    texture_weight: float = 3.45
    texture_reject_quantile: float = 0.64
    hat_texture_mix: float = 0.74
    texture_window: int = 15
    variance_weight: float = 2.90
    std_small_weight: float = 1.35
    std_large_weight: float = 1.80

    boundary_smooth_radius: int = 2
    max_cores: int = 4
    gabor_iterations: int = 2
    kmeans_max_iter: int = 150
    max_processing_time_s: float = 30.0

    homomorphic_sigma: float = 24.0
    homomorphic_ksize: int = 121
    clahe_num_tiles: Tuple[int, int] = (29, 29)
    clahe_clip_limit: float = 0.08

    edge_unsharp_amount: float = 1.10
    edge_unsharp_sigma: float = 1.20
    edge_grad_weight: float = 0.45
    sobel_refine_weight: float = 0.34

    morph_radius: int = 30
    morph_a_top: float = 11.0
    morph_a_both: float = 0.95

    gabor_ksize: int = 45
    gabor_smooth_sigma_factor: float = 0.72
    gabor_gamma: float = 0.38

    center_window_frac: float = 0.14
    kmeans_random_state: int = 0
    kmeans_batch_size: int = 8192
    kmeans_probe_batch_size: int = 4096
    kmeans_probe_max_samples: int = 100000
    pregray_lab_enhance_enabled: int = 1
    pregray_lab_clip_limit: float = 3.0
    pregray_lab_alpha: float = 1.12
    pregray_lab_beta: float = 8.0


@dataclass
class AreaResults:
    area_manual: int
    area_auto: int
    erro_abs: int
    erro_pct: float
    acerto_pct: float
    area_ratio: float
    area_diff_norm: float
    processing_time_s: float


@dataclass
class AreaEvalNoRefResults:
    area_auto: int
    processing_time_s: float


@dataclass
class AreaEvalArtifacts:
    base_rgb_u8: np.ndarray
    mask_auto: np.ndarray
    roi_mask: np.ndarray
    contour_mask: np.ndarray
    contour_overlay_rgb_u8: np.ndarray
    effective_gabor_iterations: int
    requested_gabor_iterations: int
    effective_kmeans_iter: int
    requested_kmeans_iter: int
    hat_ratio: float
    tex_ratio: float
    max_cores: int
    max_processing_time_s: float


@dataclass(frozen=True)
class ConfigParamSpec:
    field: str
    value_type: Any
    help_text: str


CONFIG_PARAM_SPECS: Tuple[ConfigParamSpec, ...] = (
    ConfigParamSpec("r_auto", int, "Espessura do contorno automatico."),
    ConfigParamSpec("min_area_fixed", int, "Equivalente ao bwareaopen(..., 5000)."),
    ConfigParamSpec("frac_min_area", float, "Fator minimo proporcional de area."),
    ConfigParamSpec("min_area_floor", int, "Piso absoluto de area minima."),
    ConfigParamSpec("proc_scale", float, "Escala de processamento [0.1, 1.0]. Menor = mais rapido."),
    ConfigParamSpec("gabor_iterations", int, "Numero de iteracoes/fases do banco de Gabor."),
    ConfigParamSpec("kmeans_max_iter", int, "Numero maximo de iteracoes do MiniBatchKMeans."),
    ConfigParamSpec("max_processing_time_s", float, "Orcamento maximo de tempo em segundos."),
    ConfigParamSpec("center_dist_weight", float, "Peso da distancia ao centro nas features."),
    ConfigParamSpec("texture_weight", float, "Peso da textura nas features."),
    ConfigParamSpec("texture_reject_quantile", float, "Rejeita topo de textura na mascara inicial."),
    ConfigParamSpec("hat_texture_mix", float, "Mistura da energia top+bottom-hat na textura [0..1]."),
    ConfigParamSpec("texture_window", int, "Janela para variancia local da textura."),
    ConfigParamSpec("variance_weight", float, "Peso da variancia local nas features."),
    ConfigParamSpec("std_small_weight", float, "Peso do desvio padrao local em janela menor."),
    ConfigParamSpec("std_large_weight", float, "Peso do desvio padrao local em janela maior."),
    ConfigParamSpec("boundary_smooth_radius", int, "Suavizacao morfologica da borda."),
    ConfigParamSpec("max_cores", int, "Limite de nucleos para processamento paralelo [1..4]."),
    ConfigParamSpec("homomorphic_sigma", float, "Sigma do filtro homomorfico."),
    ConfigParamSpec("homomorphic_ksize", int, "Kernel size do filtro homomorfico."),
    ConfigParamSpec("clahe_clip_limit", float, "Clip limit do CLAHE."),
    ConfigParamSpec("edge_unsharp_amount", float, "Ganho de nitidez (unsharp) para reforco de borda."),
    ConfigParamSpec("edge_unsharp_sigma", float, "Sigma do desfoque usado no unsharp."),
    ConfigParamSpec("edge_grad_weight", float, "Peso do gradiente local no reforco de borda."),
    ConfigParamSpec("sobel_refine_weight", float, "Peso do refinamento por Sobel."),
    ConfigParamSpec("morph_radius", int, "Raio da morfologia top/black-hat."),
    ConfigParamSpec("morph_a_top", float, "Peso do top-hat."),
    ConfigParamSpec("morph_a_both", float, "Peso do black-hat."),
    ConfigParamSpec("gabor_ksize", int, "Kernel size dos filtros de Gabor."),
    ConfigParamSpec("gabor_smooth_sigma_factor", float, "Fator de sigma da suavizacao Gabor."),
    ConfigParamSpec("gabor_gamma", float, "Gamma (aspect ratio) dos filtros de Gabor."),
    ConfigParamSpec("center_window_frac", float, "Fracao da janela central para escolher o cluster."),
    ConfigParamSpec("pregray_lab_enhance_enabled", int, "Ativa realce LAB antes da conversao para cinza (0/1)."),
    ConfigParamSpec("pregray_lab_clip_limit", float, "Clip limit do CLAHE em LAB antes do cinza."),
    ConfigParamSpec("pregray_lab_alpha", float, "Ganho de contraste no canal L (LAB) antes do cinza."),
    ConfigParamSpec("pregray_lab_beta", float, "Offset de brilho no canal L (LAB) antes do cinza."),
)


PIPELINE_OVERVIEW: Tuple[Tuple[str, str], ...] = (
    ("Leitura", "read_image -> pregray_lab_enhance -> ensure_gray -> _resize_gray_for_processing"),
    ("Pre-processamento", "_preprocess_maps (homomorphic, CLAHE, sobel_edge_refine, top-hat/bottom-hat)"),
    ("Features", "Gabor aumentado + std small/large + variance + texture + distCenter"),
    ("Clusterizacao", "imsegkmeans_like + candidatos (centro, score, Otsu)"),
    ("Pos-processamento", "postprocess_mask + vertical_scratch_cleanup"),
    ("Metricas", "calculo de area_auto, erro_abs, erro_pct e acerto_pct"),
)


def _apply_config_overrides(config: AreaEvalConfig, overrides: Mapping[str, Any]) -> AreaEvalConfig:
    if not overrides:
        return config

    valid_fields = {f.name for f in fields(AreaEvalConfig)}
    unknown = sorted(set(overrides) - valid_fields)
    if unknown:
        names = ", ".join(unknown)
        raise TypeError(f"Parametros desconhecidos para AreaEvalConfig: {names}")
    return replace(config, **overrides)


def _normalize_config(config: AreaEvalConfig) -> AreaEvalConfig:
    if not (0.1 <= float(config.proc_scale) <= 1.0):
        raise ValueError("proc_scale deve estar no intervalo [0.1, 1.0].")

    max_cores = int(config.max_cores)
    if not (1 <= max_cores <= 4):
        raise ValueError("max_cores deve estar no intervalo [1, 4].")

    gabor_wavelengths = tuple(float(v) for v in config.gabor_wavelengths)
    gabor_orientations = tuple(float(v) for v in config.gabor_orientations_deg)
    if not gabor_wavelengths or not gabor_orientations:
        raise ValueError("Informe ao menos 1 wavelength e 1 orientacao de Gabor.")

    col_auto = tuple(int(np.clip(c, 0, 255)) for c in config.col_auto)
    return replace(
        config,
        r_auto=max(0, int(config.r_auto)),
        col_auto=col_auto,
        proc_scale=float(config.proc_scale),
        gabor_wavelengths=gabor_wavelengths,
        gabor_orientations_deg=gabor_orientations,
        texture_window=max(3, int(config.texture_window)),
        variance_weight=max(0.0, float(config.variance_weight)),
        std_small_weight=max(0.0, float(config.std_small_weight)),
        std_large_weight=max(0.0, float(config.std_large_weight)),
        boundary_smooth_radius=max(0, int(config.boundary_smooth_radius)),
        max_cores=max_cores,
        gabor_iterations=max(1, int(config.gabor_iterations)),
        kmeans_max_iter=max(50, int(config.kmeans_max_iter)),
        max_processing_time_s=max(5.0, float(config.max_processing_time_s)),
        homomorphic_sigma=max(1.0, float(config.homomorphic_sigma)),
        homomorphic_ksize=as_odd(config.homomorphic_ksize, min_value=3),
        clahe_num_tiles=(max(2, int(config.clahe_num_tiles[0])), max(2, int(config.clahe_num_tiles[1]))),
        clahe_clip_limit=max(0.01, float(config.clahe_clip_limit)),
        edge_unsharp_amount=max(0.0, float(config.edge_unsharp_amount)),
        edge_unsharp_sigma=max(0.2, float(config.edge_unsharp_sigma)),
        edge_grad_weight=max(0.0, float(config.edge_grad_weight)),
        sobel_refine_weight=max(0.0, float(config.sobel_refine_weight)),
        morph_radius=max(1, int(config.morph_radius)),
        gabor_ksize=as_odd(config.gabor_ksize, min_value=3),
        center_window_frac=max(0.02, float(config.center_window_frac)),
        kmeans_batch_size=max(256, int(config.kmeans_batch_size)),
        kmeans_probe_batch_size=max(256, int(config.kmeans_probe_batch_size)),
        kmeans_probe_max_samples=max(1000, int(config.kmeans_probe_max_samples)),
        pregray_lab_enhance_enabled=1 if int(config.pregray_lab_enhance_enabled) > 0 else 0,
        pregray_lab_clip_limit=max(0.01, float(config.pregray_lab_clip_limit)),
        pregray_lab_alpha=max(0.1, float(config.pregray_lab_alpha)),
        pregray_lab_beta=float(config.pregray_lab_beta),
    )


def _resize_gray_for_processing(img_gray: np.ndarray, proc_scale: float) -> np.ndarray:
    if float(proc_scale) >= 0.999:
        return img_gray
    orig_h, orig_w = img_gray.shape
    work_w = max(8, int(round(orig_w * float(proc_scale))))
    work_h = max(8, int(round(orig_h * float(proc_scale))))
    return cv2.resize(img_gray, (work_w, work_h), interpolation=cv2.INTER_AREA)


def _preprocess_maps(img_gray_work: np.ndarray, config: AreaEvalConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat1 = mat2gray(img_gray_work)
    mat2 = homomorphic_filter(
        mat1,
        sigma=config.homomorphic_sigma,
        ksize=config.homomorphic_ksize,
    )

    img_eq = clahe_matlab_like(
        mat2,
        num_tiles=config.clahe_num_tiles,
        clip_limit=config.clahe_clip_limit,
    )

    img_edge = edge_contrast_boost(
        img_eq,
        unsharp_amount=config.edge_unsharp_amount,
        unsharp_sigma=config.edge_unsharp_sigma,
        grad_weight=config.edge_grad_weight,
    )

    img_refined = sobel_edge_refine(
        img_edge,
        sobel_weight=config.sobel_refine_weight,
        unsharp_amount=0.90,
        unsharp_sigma=1.10,
        x_weight=1.00,
        y_weight=0.30,
    )

    img_morph, hat_energy = morph_matlab_like(
        img_refined,
        radius=config.morph_radius,
        a_top=config.morph_a_top,
        a_both=config.morph_a_both,
        return_hat_energy=True,
    )
    return img_eq, img_morph, hat_energy


def _estimate_gabor_iterations(img_morph: np.ndarray, config: AreaEvalConfig, t0: float) -> int:
    gabor_jobs_per_iter = len(config.gabor_wavelengths) * len(config.gabor_orientations_deg)
    probe_t0 = time.perf_counter()
    _ = gabor_response(
        img_morph,
        config.gabor_wavelengths[0],
        config.gabor_orientations_deg[0],
        config.gabor_ksize,
        config.gabor_smooth_sigma_factor,
        config.gabor_gamma,
        phase_offset=0.0,
    )
    probe_s = max(time.perf_counter() - probe_t0, 1e-4)
    remaining_after_probe = config.max_processing_time_s - (time.perf_counter() - t0)
    reserve_for_kmeans = min(12.0, max(4.0, 0.38 * config.max_processing_time_s))
    gabor_budget = max(1.0, remaining_after_probe - reserve_for_kmeans)
    est_gabor_iter_s = 1.20 * probe_s * (gabor_jobs_per_iter / max(1.0, float(config.max_cores)))
    return min(
        config.gabor_iterations,
        max(1, int(np.floor(gabor_budget / max(est_gabor_iter_s, 1e-4)))),
    )


def _aggregate_gabor_features(
    img_morph: np.ndarray,
    config: AreaEvalConfig,
    gabor_iterations: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    phase_offsets = np.linspace(0.0, np.pi / 2.0, num=gabor_iterations, endpoint=True, dtype=np.float32)
    gabor_acc = None
    for idx, phase in enumerate(phase_offsets, start=1):
        feats_i = gabor_feature_stack(
            img_morph,
            wavelengths=config.gabor_wavelengths,
            orientations_deg=config.gabor_orientations_deg,
            ksize=config.gabor_ksize,
            smooth_sigma_factor=config.gabor_smooth_sigma_factor,
            gamma=config.gabor_gamma,
            phase_offset=float(phase),
            parallel_workers=config.max_cores,
        )
        gabor_acc = feats_i if gabor_acc is None else (gabor_acc + feats_i)
        if progress_callback is not None:
            try:
                progress_callback(idx, gabor_iterations)
            except Exception:
                pass
    return (gabor_acc / float(gabor_iterations)).astype(np.float32, copy=False)


def _build_auxiliary_maps(
    img_morph: np.ndarray,
    hat_energy: np.ndarray,
    config: AreaEvalConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img_morph.shape
    yy, xx = np.ogrid[:h, :w]
    dist_center = mat2gray(np.hypot(xx - (w / 2.0), yy - (h / 2.0)).astype(np.float32))
    texture_base = texture_score_map(img_morph, local_win=config.texture_window)
    variance_map = local_variance_map(img_morph, local_win=config.texture_window)
    hat_mix = float(np.clip(config.hat_texture_mix, 0.0, 1.0))
    texture_score = mat2gray(((1.0 - hat_mix) * texture_base) + (hat_mix * hat_energy))
    return dist_center, texture_score, variance_map


def _build_feature_set(
    gabor_feats: np.ndarray,
    dist_center: np.ndarray,
    texture_score: np.ndarray,
    variance_map: np.ndarray,
    std_small: np.ndarray,
    std_large: np.ndarray,
    config: AreaEvalConfig,
) -> np.ndarray:
    return np.concatenate(
        [
            gabor_feats,
            (float(config.std_small_weight) * std_small)[:, :, None],
            (float(config.std_large_weight) * std_large)[:, :, None],
            (float(config.variance_weight) * variance_map)[:, :, None],
            (float(config.texture_weight) * texture_score)[:, :, None],
            (float(config.center_dist_weight) * dist_center)[:, :, None],
        ],
        axis=2,
    ).astype(np.float32, copy=False)


def _estimate_kmeans_iterations(feature_set: np.ndarray, config: AreaEvalConfig, t0: float) -> int:
    remaining_for_kmeans = config.max_processing_time_s - (time.perf_counter() - t0)
    effective_kmeans_iter = config.kmeans_max_iter
    if remaining_for_kmeans <= 1.0:
        return min(effective_kmeans_iter, 30)
    if config.kmeans_max_iter <= 40:
        return effective_kmeans_iter

    probe_iters = min(20, max(8, config.kmeans_max_iter // 10))
    flat_feats = feature_set.reshape(-1, feature_set.shape[2])
    if flat_feats.shape[0] > config.kmeans_probe_max_samples:
        step = int(np.ceil(float(flat_feats.shape[0]) / float(config.kmeans_probe_max_samples)))
        probe_flat = flat_feats[::step]
    else:
        probe_flat = flat_feats

    probe_set = probe_flat.reshape(probe_flat.shape[0], 1, feature_set.shape[2])
    probe_t0 = time.perf_counter()
    _ = imsegkmeans_like(
        probe_set,
        n_clusters=2,
        random_state=config.kmeans_random_state,
        batch_size=config.kmeans_probe_batch_size,
        max_iter=probe_iters,
    )
    probe_s = max(time.perf_counter() - probe_t0, 1e-4)
    est_kmeans_s = probe_s * (
        float(flat_feats.shape[0]) / max(1.0, float(probe_flat.shape[0]))
    ) * (
        float(config.kmeans_max_iter) / max(1.0, float(probe_iters))
    )
    available_for_kmeans = max(1.0, config.max_processing_time_s - (time.perf_counter() - t0) - 1.0)
    if est_kmeans_s <= available_for_kmeans:
        return effective_kmeans_iter

    scale = available_for_kmeans / est_kmeans_s
    return max(20, int(np.floor(config.kmeans_max_iter * scale)))


def contour_mask_with_thickness(mask: np.ndarray, radius: int) -> np.ndarray:
    contour = perimeter_mask(mask)
    if int(radius) <= 0:
        return contour

    rk = as_odd((2 * int(radius)) + 1, min_value=3)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rk, rk))
    return cv2.dilate((contour.astype(np.uint8) * 255), se) > 0


def _show_results_plot(
    img_gray: np.ndarray,
    base_rgb_u8: np.ndarray,
    mask_auto: np.ndarray,
    area_auto: int,
    erro_pct: float,
    acerto_pct: float,
    config: AreaEvalConfig,
) -> None:
    auto_perim = contour_mask_with_thickness(mask_auto, radius=config.r_auto)
    img_contours = overlay_perimeter(base_rgb_u8, auto_perim, config.col_auto)
    img_gray01 = mat2gray(img_gray)
    filled_overlay = labeloverlay_like(img_gray01, mask_auto, color_rgb01=(1.0, 0.0, 0.0), alpha=0.55)

    fig = plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img_contours)
    ax1.set_title("Segmentacao automatica (contorno)")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(filled_overlay)
    ax2.set_title(
        f"Area auto = {area_auto:.0f} px | Erro = {erro_pct:.2f}% | Acerto = {acerto_pct:.2f}%"
    )
    ax2.axis("off")

    fig.suptitle("Avaliacao baseada exclusivamente em area (pixels)")
    plt.tight_layout()
    plt.show()


def run_area_eval(
    base_path: str,
    area_manual: int,
    config: Optional[AreaEvalConfig] = None,
    roi_mask: Optional[np.ndarray] = None,
    show: bool = True,
    verbose: bool = True,
    return_artifacts: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    **overrides: Any,
) -> Union[AreaResults, Tuple[AreaResults, AreaEvalArtifacts]]:
    t0 = time.perf_counter()

    def emit_progress(stage: str, value01: float) -> None:
        if progress_callback is None:
            return
        try:
            clamped = float(np.clip(value01, 0.0, 1.0))
            progress_callback(stage, clamped)
        except Exception:
            pass

    emit_progress("Carregando imagem", 0.02)
    if isinstance(config, dict):
        merged_overrides = {**config, **overrides}
        base_config = AreaEvalConfig()
    elif isinstance(config, AreaEvalConfig) or config is None:
        merged_overrides = overrides
        base_config = config or AreaEvalConfig()
    else:
        merged_overrides = {"r_auto": config, **overrides}
        base_config = AreaEvalConfig()

    cfg = _apply_config_overrides(base_config, merged_overrides)
    cfg = _normalize_config(cfg)
    emit_progress("Configurando parametros", 0.06)

    gabor_iterations_requested = cfg.gabor_iterations
    kmeans_max_iter_requested = cfg.kmeans_max_iter

    img_base = read_image(base_path)
    img_base_for_gray = pregray_lab_enhance(
        img_base,
        enabled=cfg.pregray_lab_enhance_enabled,
        clip_limit=cfg.pregray_lab_clip_limit,
        alpha=cfg.pregray_lab_alpha,
        beta=cfg.pregray_lab_beta,
    )
    img_gray = ensure_gray(img_base_for_gray)
    base_rgb_u8 = ensure_rgb_uint8(img_base)
    emit_progress("Preparando ROI", 0.12)

    orig_h, orig_w = img_gray.shape
    if roi_mask is None:
        roi_full = np.ones((orig_h, orig_w), dtype=bool)
    else:
        roi_arr = np.asarray(roi_mask).astype(bool)
        if roi_arr.shape != (orig_h, orig_w):
            raise ValueError(
                "roi_mask deve ter o mesmo tamanho da imagem base: "
                f"esperado {(orig_h, orig_w)}, recebido {roi_arr.shape}."
            )
        if not np.any(roi_arr):
            raise ValueError("roi_mask nao pode ser vazio.")
        roi_full = roi_arr

    img_gray_work = _resize_gray_for_processing(img_gray, cfg.proc_scale)
    if img_gray_work.shape != img_gray.shape:
        roi_work = cv2.resize(
            roi_full.astype(np.uint8),
            (img_gray_work.shape[1], img_gray_work.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    else:
        roi_work = roi_full.copy()

    with limit_native_threads(cfg.max_cores):
        emit_progress("Pre-processando imagem", 0.22)
        img_eq, img_morph, hat_energy = _preprocess_maps(img_gray_work, cfg)

        effective_gabor_iterations = max(1, int(cfg.gabor_iterations))
        emit_progress("Executando filtros de Gabor", 0.36)
        gabor_feats = _aggregate_gabor_features(
            img_morph,
            cfg,
            effective_gabor_iterations,
            progress_callback=lambda idx, total: emit_progress(
                f"Executando filtros de Gabor ({idx}/{total})",
                0.36 + (0.30 * (float(idx) / float(max(total, 1)))),
            ),
        )

        emit_progress("Montando mapas auxiliares", 0.68)
        dist_center, texture_score, variance_map = _build_auxiliary_maps(
            img_morph,
            hat_energy,
            cfg,
        )

        std_small = local_std_map(img_eq, local_win=21)
        std_large = local_std_map(img_eq, local_win=45)

        emit_progress("Montando feature set", 0.75)
        feature_set = _build_feature_set(
            gabor_feats=gabor_feats,
            dist_center=dist_center,
            texture_score=texture_score,
            variance_map=variance_map,
            std_small=std_small,
            std_large=std_large,
            config=cfg,
        )

        emit_progress("Estimando iteracoes do k-means", 0.82)
        effective_kmeans_iter = _estimate_kmeans_iterations(feature_set, cfg, t0)

        emit_progress("Separando regioes (k-means)", 0.88)
        labels = imsegkmeans_like(
            feature_set,
            n_clusters=2,
            random_state=cfg.kmeans_random_state,
            batch_size=cfg.kmeans_batch_size,
            max_iter=effective_kmeans_iter,
        )

    emit_progress("Pos-processando mascara", 0.93)
    center_label = int(labels[labels.shape[0] // 2, labels.shape[1] // 2])
    mask_center = (labels == center_label) & roi_work

    score_label = pick_wound_label(
        labels,
        dist_center=dist_center,
        texture_score=texture_score,
        center_window_frac=cfg.center_window_frac,
    )
    mask_score = (labels == score_label) & roi_work

    mask_otsu = _otsu_center_mask(img_morph, roi_work)

    tex_thr = np.quantile(texture_score[roi_work], cfg.texture_reject_quantile) if np.any(roi_work) else 1.0
    low_tex_region = texture_score <= tex_thr

    area_scale = float(cfg.proc_scale) * float(cfg.proc_scale)
    min_area_floor_work = max(1, int(round(int(cfg.min_area_floor) * area_scale)))
    min_area_fixed_work = max(1, int(round(int(cfg.min_area_fixed) * area_scale)))
    roi_work_area = int(np.count_nonzero(roi_work))
    min_area_auto = max(min_area_floor_work, int(round(cfg.frac_min_area * max(roi_work_area, 1))), min_area_fixed_work)

    candidates: list[np.ndarray] = []
    for cand in (mask_center, mask_score, mask_otsu):
        cand = cand & low_tex_region & roi_work
        cand_pp = postprocess_mask(cand, min_area=min_area_auto) & roi_work
        if np.any(cand_pp):
            candidates.append(cand_pp)

    if not candidates:
        raise RuntimeError("Todas as mascaras candidatas ficaram vazias apos pos-processamento.")

    scored = [(_mask_quality_score(c, dist_center, texture_score), c) for c in candidates]
    scored.sort(key=lambda x: x[0])
    mask_auto = scored[0][1]
    mask_auto = smooth_mask(mask_auto, radius=cfg.boundary_smooth_radius) & roi_work

    if img_gray_work.shape != img_gray.shape:
        mask_auto = cv2.resize(mask_auto.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 0
        resize_smooth_radius = max(1, cfg.boundary_smooth_radius - 1)
        mask_auto = smooth_mask(mask_auto, radius=resize_smooth_radius)
        mask_auto = keep_largest_cc(mask_auto)

    mask_auto = mask_auto & roi_full
    if np.any(mask_auto):
        mask_auto = keep_largest_cc(mask_auto)

    if np.count_nonzero(mask_auto) == 0:
        raise RuntimeError("Mascara automatica ficou vazia apos o pos-processamento.")

    if mask_auto.shape != hat_energy.shape:
        mask_sep = cv2.resize(
            mask_auto.astype(np.uint8),
            (hat_energy.shape[1], hat_energy.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    else:
        mask_sep = mask_auto

    if np.any(mask_sep) and np.any(~mask_sep):
        hat_w = float(np.mean(hat_energy[mask_sep]))
        hat_c = float(np.mean(hat_energy[~mask_sep]))
        tex_w = float(np.mean(texture_score[mask_sep]))
        tex_c = float(np.mean(texture_score[~mask_sep]))
        hat_ratio = hat_c / max(hat_w, 1e-8)
        tex_ratio = tex_c / max(tex_w, 1e-8)
    else:
        hat_ratio = float("nan")
        tex_ratio = float("nan")

    area_auto = int(np.count_nonzero(mask_auto))
    erro_abs = int(abs(area_auto - int(area_manual)))
    erro_pct = 100.0 * float(erro_abs) / max(int(area_manual), 1)
    acerto_pct = max(0.0, 100.0 - float(erro_pct))
    area_ratio = float(area_auto) / max(float(area_manual), 1.0)
    area_diff_norm = float(area_auto - int(area_manual)) / max(float(area_manual), 1.0)
    processing_time_s = time.perf_counter() - t0
    emit_progress("Calculando metricas", 0.98)

    results = AreaResults(
        area_manual=int(area_manual),
        area_auto=area_auto,
        erro_abs=erro_abs,
        erro_pct=float(erro_pct),
        acerto_pct=float(acerto_pct),
        area_ratio=float(area_ratio),
        area_diff_norm=float(area_diff_norm),
        processing_time_s=float(processing_time_s),
    )

    contour_mask = contour_mask_with_thickness(mask_auto, radius=cfg.r_auto)
    contour_overlay = overlay_perimeter(base_rgb_u8, contour_mask, cfg.col_auto)
    artifacts = AreaEvalArtifacts(
        base_rgb_u8=base_rgb_u8,
        mask_auto=mask_auto,
        roi_mask=roi_full,
        contour_mask=contour_mask,
        contour_overlay_rgb_u8=contour_overlay,
        effective_gabor_iterations=effective_gabor_iterations,
        requested_gabor_iterations=gabor_iterations_requested,
        effective_kmeans_iter=effective_kmeans_iter,
        requested_kmeans_iter=kmeans_max_iter_requested,
        hat_ratio=hat_ratio,
        tex_ratio=tex_ratio,
        max_cores=cfg.max_cores,
        max_processing_time_s=cfg.max_processing_time_s,
    )

    if verbose:
        print("\n=========== RESULTADOS (AREA) ===========")
        print(f"Area manual (ref) : {results.area_manual:.0f} px")
        print(f"Area automatica   : {results.area_auto:.0f} px")
        print(f"Erro absoluto     : {results.erro_abs:.0f} px")
        print(f"Erro relativo     : {results.erro_pct:.2f} %")
        print(f"Acerto relativo   : {results.acerto_pct:.2f} %")
        print(f"Razao de area     : {results.area_ratio:.3f}")
        print(f"Limite de nucleos : {cfg.max_cores}")
        print(f"Gabor iteracoes   : {effective_gabor_iterations}/{gabor_iterations_requested}")
        print(f"K-means max_iter  : {effective_kmeans_iter}/{kmeans_max_iter_requested}")
        print(f"Orcamento tempo   : {cfg.max_processing_time_s:.1f} s")
        print(f"Tempo processamento: {results.processing_time_s:.3f} s")
        if results.processing_time_s > cfg.max_processing_time_s:
            print("Aviso             : Tempo excedeu o limite configurado; reduza iters ou escala.")
        print(f"Separacao hat (cel/cic): {hat_ratio:.3f} | Separacao textura (cel/cic): {tex_ratio:.3f}")
        print("Tendencia         : Supersegmentacao" if results.area_ratio > 1 else "Tendencia         : Subsegmentacao")

    if show:
        _show_results_plot(img_gray, base_rgb_u8, mask_auto, area_auto, erro_pct, acerto_pct, cfg)

    emit_progress("Concluido", 1.0)
    if return_artifacts:
        return results, artifacts
    return results


def run_area_eval_no_reference(
    base_path: str,
    config: Optional[AreaEvalConfig] = None,
    roi_mask: Optional[np.ndarray] = None,
    show: bool = True,
    verbose: bool = True,
    return_artifacts: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    **overrides: Any,
) -> Union[AreaEvalNoRefResults, Tuple[AreaEvalNoRefResults, AreaEvalArtifacts]]:
    def emit_progress(stage: str, value01: float) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(stage, float(np.clip(value01, 0.0, 1.0)))
        except Exception:
            pass

    emit_progress("Avaliando brilho da imagem", 0.01)
    probe_rgb = read_image(base_path)
    probe_gray = ensure_gray(
        pregray_lab_enhance(
            probe_rgb,
            enabled=1,
            clip_limit=2.2,
            alpha=1.0,
            beta=0.0,
            use_local_sharpen=0,
        )
    )
    brightness01 = float(np.mean(mat2gray(probe_gray)))
    bright_mode = brightness01 >= 0.58

    merged_overrides = dict(overrides)
    emit_progress("Configurando modo adaptativo", 0.03)
    results_with_dummy, artifacts = run_area_eval(
        base_path=base_path,
        area_manual=1,
        config=config,
        roi_mask=roi_mask,
        show=False,
        verbose=False,
        return_artifacts=True,
        progress_callback=progress_callback,
        **merged_overrides,
    )

    results = AreaEvalNoRefResults(
        area_auto=results_with_dummy.area_auto,
        processing_time_s=results_with_dummy.processing_time_s,
    )

    if verbose:
        print("\n=========== RESULTADOS (SEM GABARITO) ===========")
        print(
            f"Modo adaptativo   : {'claro' if bright_mode else 'escuro'} "
            f"(brilho medio={brightness01:.3f})"
        )
        print(f"Area automatica   : {results.area_auto:.0f} px")
        print(f"Tempo processamento: {results.processing_time_s:.3f} s")
        print(f"Limite de nucleos : {artifacts.max_cores}")
        print(
            "Gabor iteracoes   : "
            f"{artifacts.effective_gabor_iterations}/{artifacts.requested_gabor_iterations}"
        )
        print(
            "K-means max_iter  : "
            f"{artifacts.effective_kmeans_iter}/{artifacts.requested_kmeans_iter}"
        )
        print(f"Orcamento tempo   : {artifacts.max_processing_time_s:.1f} s")

    if show:
        img_gray = ensure_gray(pregray_lab_enhance(read_image(base_path)))
        filled_overlay = labeloverlay_like(
            mat2gray(img_gray),
            artifacts.mask_auto,
            color_rgb01=(1.0, 0.0, 0.0),
            alpha=0.55,
        )

        fig = plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(artifacts.contour_overlay_rgb_u8)
        ax1.set_title("Segmentacao automatica (contorno)")
        ax1.axis("off")

        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(filled_overlay)
        ax2.set_title(f"Area auto = {results.area_auto:.0f} px")
        ax2.axis("off")

        fig.suptitle("Processamento sem gabarito")
        plt.tight_layout()
        plt.show()

    if return_artifacts:
        return results, artifacts
    return results


def build_parser() -> argparse.ArgumentParser:
    defaults = AreaEvalConfig()
    parser = argparse.ArgumentParser(description="Conversao MATLAB -> Python para avaliacao de area de scratch.")
    parser.add_argument(
        "--base",
        type=str,
        default=BASE_IMAGE_PATH,
        help="Caminho da imagem base. Se vazio, usa BASE_IMAGE_PATH definido no codigo.",
    )
    parser.add_argument("--area_manual", type=int, default=403281, help="Area manual em pixels (medicao externa).")
    parser.add_argument(
        "--gabor_wavelengths",
        type=float,
        nargs="+",
        default=list(defaults.gabor_wavelengths),
        help="Comprimentos de onda do banco de Gabor.",
    )
    parser.add_argument(
        "--gabor_orientations",
        type=float,
        nargs="+",
        default=list(defaults.gabor_orientations_deg),
        help="Orientacoes (graus) do banco de Gabor.",
    )
    parser.add_argument(
        "--clahe_num_tiles",
        type=int,
        nargs=2,
        metavar=("TILES_Y", "TILES_X"),
        default=list(defaults.clahe_num_tiles),
        help="Numero de tiles CLAHE em Y e X.",
    )

    for spec in CONFIG_PARAM_SPECS:
        parser.add_argument(
            f"--{spec.field}",
            type=spec.value_type,
            default=getattr(defaults, spec.field),
            help=spec.help_text,
        )

    parser.add_argument("--no_show", action="store_true", help="Nao exibe figura final.")
    parser.add_argument(
        "--show_pipeline",
        action="store_true",
        help="Exibe resumo das etapas/funcoes do pipeline e encerra.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AreaEvalConfig:
    cfg_kwargs = {spec.field: getattr(args, spec.field) for spec in CONFIG_PARAM_SPECS}
    cfg_kwargs["gabor_wavelengths"] = tuple(args.gabor_wavelengths)
    cfg_kwargs["gabor_orientations_deg"] = tuple(args.gabor_orientations)
    cfg_kwargs["clahe_num_tiles"] = tuple(args.clahe_num_tiles)
    return AreaEvalConfig(**cfg_kwargs)


def print_pipeline_overview() -> None:
    print("\n=== PIPELINE ===")
    for stage, chain in PIPELINE_OVERVIEW:
        print(f"- {stage}: {chain}")
    print("\n=== PARAMETROS AJUSTAVEIS ===")
    for spec in CONFIG_PARAM_SPECS:
        print(f"- --{spec.field}: {spec.help_text}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.show_pipeline:
        print_pipeline_overview()
        return
    base_path = str(args.base).strip()
    if not base_path:
        raise ValueError(
            "Defina o caminho da imagem em BASE_IMAGE_PATH no codigo "
            "ou passe --base no terminal."
        )
    config = config_from_args(args)
    run_area_eval(
        base_path=base_path,
        area_manual=args.area_manual,
        config=config,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
