#!/usr/bin/env python3
"""
3a画像検索CGI課題: 特徴量抽出スクリプト

抽出する特徴量:
  color histogram: {RGB, HSV, LUV} x {1x1, 2x2, 3x3}    -> 9種
  dcnn           : VGG16 fc2 (4096-d, L2正規化)          -> 1種
合計 13種。

使い方:
  python3 extract_features.py --img-dir /path/to/images --out-dir features --features all
  python3 extract_features.py --img-dir imgdata --features rgb_3x3 hsv_2x2
  python3 extract_features.py --img-dir imgdata --features dcnn --device cuda
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np


IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp")

COLOR_SPACES = ("rgb", "hsv", "luv")
GRIDS = (1, 2, 3)


# ------------------------------------------------------------------
# 画像列挙
# ------------------------------------------------------------------
def list_images(img_dir):
    paths = []
    for ext in IMG_EXT:
        paths.extend(glob.glob(os.path.join(img_dir, "**", ext), recursive=True))
    paths.sort()
    return paths


def read_image(path, resize=None):
    img = cv2.imread(path)
    if img is None:
        return None
    if resize is not None:
        img = cv2.resize(img, resize)
    return img


# ------------------------------------------------------------------
# カラーヒストグラム (自作: cv2.calcHist を使う)
# ------------------------------------------------------------------
def color_hist(img_bgr, color_space, bins=8):
    """1画像 (または1パッチ) の color histogram. bins*3 次元, L1正規化."""
    cs = color_space.lower()
    if cs == "rgb":
        im = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ranges = [(0, 256), (0, 256), (0, 256)]
    elif cs == "hsv":
        im = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        ranges = [(0, 180), (0, 256), (0, 256)]  # OpenCVのHは0-179
    elif cs == "luv":
        im = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LUV)
        ranges = [(0, 256), (0, 256), (0, 256)]
    else:
        raise ValueError(f"unknown color space: {color_space}")

    hs = []
    for c in range(3):
        h = cv2.calcHist([im], [c], None, [bins], [ranges[c][0], ranges[c][1]])
        hs.append(h.flatten())
    hist = np.concatenate(hs).astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def grid_color_hist(img_bgr, color_space, grid, bins=8):
    """grid x grid に分割してヒストを並べる. 次元 = 3 * bins * grid * grid."""
    h, w = img_bgr.shape[:2]
    feats = []
    for gy in range(grid):
        for gx in range(grid):
            y0 = h * gy // grid
            y1 = h * (gy + 1) // grid
            x0 = w * gx // grid
            x1 = w * (gx + 1) // grid
            patch = img_bgr[y0:y1, x0:x1]
            feats.append(color_hist(patch, color_space, bins))
    return np.concatenate(feats).astype(np.float32)

# ------------------------------------------------------------------
# DCNN (VGG16 fc2, 4096d, L2正規化)
# ------------------------------------------------------------------
def build_vgg16(device):
    import torch
    import torchvision.models as models

    weights = models.VGG16_Weights.IMAGENET1K_V1
    m = models.vgg16(weights=weights)
    # classifier = [Linear, ReLU, Dropout, Linear(fc2), ReLU, Dropout, Linear(1000)]
    # fc2の直後 (ReLU後) を取る: 4096次元
    m.classifier = _Sequential_upto(m.classifier, 5)  # index 0..4 を残す
    m.eval()
    m.to(device)
    return m, weights.transforms()


def _Sequential_upto(seq, n):
    import torch.nn as nn

    return nn.Sequential(*list(seq.children())[:n])


def dcnn_batch(model, transform, imgs_bgr, device, batch_size=32):
    import torch

    feats = []
    n = len(imgs_bgr)
    for i in range(0, n, batch_size):
        batch = []
        for img in imgs_bgr[i : i + batch_size]:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # transformはPIL想定なので変換
            from PIL import Image

            pil = Image.fromarray(rgb)
            batch.append(transform(pil))
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            y = model(x).cpu().numpy()
        feats.append(y)
    feats = np.concatenate(feats, axis=0).astype(np.float32)
    # L2正規化
    norm = np.linalg.norm(feats, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    feats = feats / norm
    return feats


# ------------------------------------------------------------------
# 特徴リスト解決
# ------------------------------------------------------------------
def resolve_feature_names(names):
    """'all' と個別指定を展開."""
    out = []
    for n in names:
        n = n.lower()
        if n == "all":
            for cs in COLOR_SPACES:
                for g in GRIDS:
                    out.append(f"{cs}_{g}x{g}")
            out.append("dcnn")
        else:
            out.append(n)
    # 重複除去, 順序保持
    seen = set()
    uniq = []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def parse_feature_name(name):
    """'rgb_2x2' -> ('rgb', 2), 'gabor_3x3' -> ('gabor', 3), 'dcnn' -> ('dcnn', 0)."""
    name = name.lower()
    if name == "dcnn":
        return "dcnn", 0
    if "_" in name:
        kind, grid_str = name.split("_", 1)
        # '2x2' -> 2
        g = int(grid_str.split("x")[0])
        return kind, g
    raise ValueError(f"unknown feature name: {name}")


# ------------------------------------------------------------------
# メイン
# ------------------------------------------------------------------
def extract_all(img_dir, out_dir, feature_names, bins=8, resize=(256, 256),
                device="cpu", batch_size=32, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    paths = list_images(img_dir)
    if not paths:
        print(f"[error] no images in {img_dir}", file=sys.stderr)
        sys.exit(1)
    if verbose:
        print(f"[info] {len(paths)} images found in {img_dir}")

    # 最初に全画像を読んでメモリに載せる (1000枚*256*256*3 = 約200MB なので許容)
    imgs = []
    good_paths = []
    t0 = time.time()
    for p in paths:
        img = read_image(p, resize=resize)
        if img is None:
            if verbose:
                print(f"[warn] cannot read: {p}", file=sys.stderr)
            continue
        imgs.append(img)
        good_paths.append(p)
    if verbose:
        print(f"[info] loaded {len(imgs)} images in {time.time()-t0:.1f}s")

    # 画像パスはbasename相対に保存 (CGIで扱いやすく)
    rel_paths = [os.path.relpath(p, img_dir) for p in good_paths]

    # DCNN モデルは一度だけロード
    vgg = None
    vgg_tf = None

    for fname in feature_names:
        kind, grid = parse_feature_name(fname)
        out_path = os.path.join(out_dir, f"{fname}.npz")
        t0 = time.time()

        if kind in COLOR_SPACES:
            feats = np.stack([
                grid_color_hist(im, kind, grid, bins=bins) for im in imgs
            ]).astype(np.float32)

        elif kind == "dcnn":
            if vgg is None:
                if verbose:
                    print("[info] loading VGG16 ...")
                vgg, vgg_tf = build_vgg16(device)
            feats = dcnn_batch(vgg, vgg_tf, imgs, device=device, batch_size=batch_size)

        else:
            print(f"[error] unknown feature kind: {kind}", file=sys.stderr)
            continue

        np.savez(
            out_path,
            features=feats,
            paths=np.array(rel_paths),
            name=np.array(fname),
        )
        if verbose:
            print(f"[ok] {fname}: shape={feats.shape} "
                  f"saved to {out_path} ({time.time()-t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True, help="画像ルートディレクトリ")
    ap.add_argument("--out-dir", default="features", help="特徴量出力ディレクトリ")
    ap.add_argument("--features", nargs="+", default=["all"],
                    help="抽出する特徴名 (例: rgb_1x1 hsv_2x2 dcnn / all)")
    ap.add_argument("--bins", type=int, default=8, help="color histogram の bin 数")
    ap.add_argument("--resize", type=int, nargs=2, default=[256, 256],
                    help="リサイズ後 w h (0 0 でリサイズ無し)")
    ap.add_argument("--device", default="cpu", help="DCNN 用 device (cpu/cuda)")
    ap.add_argument("--batch-size", type=int, default=32, help="DCNN バッチサイズ")
    args = ap.parse_args()

    resize = tuple(args.resize) if args.resize[0] > 0 else None
    names = resolve_feature_names(args.features)
    print(f"[info] features to extract: {names}")

    extract_all(
        img_dir=args.img_dir,
        out_dir=args.out_dir,
        feature_names=names,
        bins=args.bins,
        resize=resize,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
