#!/usr/bin/env python3
"""
画像検索CGI課題: 類似度計算モジュール

- L2 (ユークリッド) 距離: 小さい順にソート
- Histogram Intersection: sum(min(a,b))、大きい順にソート (類似度)

CGI や他スクリプトから以下のように使う:

    from search import load_feature, search_by_index
    data = load_feature('features/rgb_3x3.npz')
    results = search_by_index(data, query_idx=0, metric='l2', topk=20)
    # results: [(idx, score), ...]  paths は data['paths'][idx]
"""

import argparse
import os
import sys
import numpy as np


# ------------------------------------------------------------------
# ロード
# ------------------------------------------------------------------
def load_feature(npz_path):
    """extract_features.py が出した .npz を読む."""
    d = np.load(npz_path, allow_pickle=True)
    return {
        "features": d["features"].astype(np.float32),
        "paths": [str(p) for p in d["paths"]],
        "name": str(d["name"]),
    }


# ------------------------------------------------------------------
# 距離 / 類似度
# ------------------------------------------------------------------
def l2_distance(query, db):
    """
    query: (D,) または (1, D)
    db   : (N, D)
    返り値: (N,) 距離 (小さいほど類似)
    """
    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    diff = db.astype(np.float32) - q
    return np.sqrt((diff * diff).sum(axis=1))


def histogram_intersection(query, db):
    """
    sum_i min(q_i, d_i). 値域は両者が L1 正規化なら [0, 1].
    返り値: (N,) 類似度 (大きいほど類似)
    """
    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    return np.minimum(db.astype(np.float32), q).sum(axis=1)


# ------------------------------------------------------------------
# ランキング
# ------------------------------------------------------------------
def rank(query, db, metric="l2", topk=None):
    """
    metric='l2'  -> 距離 (昇順)
    metric='hist' or 'intersection' -> 類似度 (降順)
    返り値: [(idx, score), ...]
    """
    metric = metric.lower()
    if metric == "l2":
        scores = l2_distance(query, db)
        order = np.argsort(scores)
    elif metric in ("hist", "intersection", "hi"):
        scores = histogram_intersection(query, db)
        order = np.argsort(-scores)
    else:
        raise ValueError(f"unknown metric: {metric}")

    if topk is not None:
        order = order[: topk]
    return [(int(i), float(scores[i])) for i in order]


def search_by_index(feature_data, query_idx, metric="l2", topk=None,
                    exclude_query=False):
    """
    feature_data: load_feature() の戻り値
    query_idx   : クエリ画像の index
    exclude_query: True ならクエリ自身を結果から除外
    """
    feats = feature_data["features"]
    results = rank(feats[query_idx], feats, metric=metric, topk=None)
    if exclude_query:
        results = [(i, s) for (i, s) in results if i != query_idx]
    if topk is not None:
        results = results[: topk]
    return results


def search_by_vector(feature_data, query_vec, metric="l2", topk=None):
    return rank(query_vec, feature_data["features"], metric=metric, topk=topk)


# ------------------------------------------------------------------
# コマンドラインから単体動作確認
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-file", required=True, help=".npz 特徴ファイル")
    ap.add_argument("--query-idx", type=int, default=0)
    ap.add_argument("--metric", choices=["l2", "hist"], default="l2")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--exclude-query", action="store_true",
                    help="結果からクエリ自身を除外")
    args = ap.parse_args()

    data = load_feature(args.feature_file)
    results = search_by_index(
        data, args.query_idx, metric=args.metric, topk=args.topk,
        exclude_query=args.exclude_query,
    )
    n_total = len(data["paths"])
    qpath = data["paths"][args.query_idx]
    print(f"feature = {data['name']}  (N={n_total}, D={data['features'].shape[1]})")
    print(f"metric  = {args.metric}")
    print(f"query   = [{args.query_idx}] {qpath}")
    print(f"top {len(results)}:")
    for r, (idx, score) in enumerate(results):
        print(f"  {r:3d}: [{idx:5d}] {data['paths'][idx]}  score={score:.4f}")


if __name__ == "__main__":
    main()
