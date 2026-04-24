#!/usr/bin/env python3
"""
icrawler で複数カテゴリの画像を収集.
出力: <out_dir>/<keyword>/000001.jpg ...

例:
    python3 crawler.py                       # デフォルト 8カテゴリ x 80枚
    python3 crawler.py --per-keyword 50
    python3 crawler.py --keywords sushi ramen --per-keyword 100
    python3 crawler.py --out-dir /export/space0/lan-y/imgdata
"""
import argparse
import os

os.environ["http_proxy"] = "http://proxy.uec.ac.jp:8080/"
os.environ["https_proxy"] = "http://proxy.uec.ac.jp:8080/"

from icrawler.builtin import BingImageCrawler  # noqa: E402


# 8カテゴリ (食べ物/動物/乗り物/自然) で差が出やすい構成
DEFAULT_KEYWORDS = [
    "sushi",
    "ramen",
    "cat",
    "dog",
    "car",
    "bicycle",
    "sunflower",
    "mountain",
]

DEFAULT_OUT = "/export/space0/lan-y/imgdata"


def crawl_one(keyword, out_root, max_num, threads=5):
    storage_dir = os.path.join(out_root, keyword.replace(" ", "_"))
    os.makedirs(storage_dir, exist_ok=True)
    crawler = BingImageCrawler(
        storage={"root_dir": storage_dir},
        downloader_threads=threads,
    )
    crawler.crawl(keyword=keyword, max_num=max_num)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--keywords", nargs="+", default=DEFAULT_KEYWORDS)
    ap.add_argument("--per-keyword", type=int, default=80)
    ap.add_argument("--threads", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[info] out_dir = {args.out_dir}")
    print(f"[info] keywords ({len(args.keywords)}): {args.keywords}")
    print(f"[info] per-keyword = {args.per_keyword}")

    for k in args.keywords:
        print(f"\n===== crawling: {k} =====")
        crawl_one(k, args.out_dir, args.per_keyword, threads=args.threads)

    # 結果サマリ
    print("\n===== summary =====")
    total = 0
    for k in args.keywords:
        d = os.path.join(args.out_dir, k.replace(" ", "_"))
        n = len([f for f in os.listdir(d)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
                ) if os.path.isdir(d) else 0
        print(f"  {k:15s}: {n} images")
        total += n
    print(f"  {'TOTAL':15s}: {total} images")


if __name__ == "__main__":
    main()
