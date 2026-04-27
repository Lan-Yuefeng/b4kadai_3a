#!/usr/local/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
画像検索CGI (mm.cs.uec.ac.jp デプロイ想定).
ローカル動作確認: QUERY_STRING= python3 index.cgi
  (shebangは mm の /usr/local/anaconda3/bin/python3 を指す)
"""
import cgi
import cgitb
import glob
import html
import os
import sys
from urllib.parse import quote, urlencode

cgitb.enable()  # エラー時にブラウザへ traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from search import load_feature, search_by_index  # noqa: E402

# 設定 ----------------------------------------------------------------
FEATURES_DIR = os.path.join(HERE, "features")
IMGDATA_DIR = os.path.join(HERE, "imgdata")  # /export/space0/... へのsymlink想定
IMGDATA_URL = "imgdata"                       # ブラウザ向け相対URL
DEFAULT_TOPK = 24
GALLERY_MAX = 600
COLS = 6
IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
METRICS = [("l2", "L2距離"), ("hist", "ヒストグラムインターセクション")]


# --------------------------------------------------------------------
def available_features():
    if not os.path.isdir(FEATURES_DIR):
        return []
    names = [
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(FEATURES_DIR, "*.npz"))
    ]
    # 見やすい順: color -> dcnn
    order_key = lambda n: (0 if n != "dcnn" else 1, n)
    return sorted(names, key=order_key)


def list_images():
    paths = []
    for ext in IMG_EXT:
        paths.extend(glob.glob(os.path.join(IMGDATA_DIR, "**", ext),
                               recursive=True))
    paths.sort()
    return paths


def img_url(rel_path):
    parts = rel_path.replace(os.sep, "/").split("/")
    return f"{IMGDATA_URL}/" + "/".join(quote(p) for p in parts)


def search_link(path, feature, metric):
    return "?" + urlencode({"q": path, "feature": feature, "metric": metric})


# --------------------------------------------------------------------
def print_header(title):
    print("Content-Type: text/html; charset=utf-8")
    print()
    print(f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; color: #222; }}
h1 {{ font-size: 18px; margin: 0 0 8px; }}
h2 {{ font-size: 15px; margin: 14px 0 6px; }}
form.top {{ padding: 10px; background: #eef; border: 1px solid #ccd; margin-bottom: 12px; }}
form.top select {{ margin: 0 6px; }}
img.thumb {{ width: 120px; height: 120px; object-fit: cover; border: 1px solid #ccc; background:#fafafa; }}
img.query {{ max-width: 240px; max-height: 240px; border: 3px solid #c33; }}
table.results {{ border-collapse: collapse; }}
table.results td {{ padding: 4px; text-align: center; vertical-align: top; }}
table.results a {{ text-decoration: none; color: #333; }}
.rank {{ font-size: 10px; color: #666; }}
.score {{ font-size: 10px; color: #999; font-family: monospace; }}
.meta  {{ font-size: 12px; color: #444; }}
.err   {{ color: #c33; }}
</style></head><body>
<h1>画像検索 (3a) </h1>""")


def print_footer():
    print("</body></html>")


def render_form(feature, metric, query):
    feats = available_features()
    print('<form class="top" method="get" action="">')
    print(f'<input type="hidden" name="q" value="{html.escape(query)}">')
    print('特徴量: <select name="feature">')
    for f in feats:
        sel = " selected" if f == feature else ""
        print(f'<option value="{html.escape(f)}"{sel}>{html.escape(f)}</option>')
    print("</select>距離: <select name=\"metric\">")
    for k, label in METRICS:
        sel = " selected" if k == metric else ""
        print(f'<option value="{k}"{sel}>{html.escape(label)}</option>')
    print('</select><input type="submit" value="検索">')
    print(' <a href="?">ギャラリーに戻る</a>')
    print("</form>")


def render_gallery(feature, metric):
    paths = list_images()[:GALLERY_MAX]
    print(f'<h2>画像一覧 (計 {len(list_images())} 枚、先頭 {len(paths)} 枚表示) '
          f'— クリックで類似検索</h2>')
    if not paths:
        print('<p class="err">imgdata/ に画像がありません</p>')
        return
    print('<table class="results"><tr>')
    for i, p in enumerate(paths):
        if i and i % COLS == 0:
            print("</tr><tr>")
        rel = os.path.relpath(p, IMGDATA_DIR)
        link = search_link(rel, feature, metric)
        print(f'<td><a href="{html.escape(link)}">'
              f'<img class="thumb" src="{html.escape(img_url(rel))}" alt=""></a></td>')
    print("</tr></table>")


def render_results(query_path, feature, metric):
    npz = os.path.join(FEATURES_DIR, f"{feature}.npz")
    if not os.path.exists(npz):
        print(f'<p class="err">特徴ファイルが見つかりません: {html.escape(feature)}.npz</p>')
        return
    data = load_feature(npz)
    if query_path not in data["paths"]:
        print(f'<p class="err">クエリ画像が特徴DBに存在しません: '
              f'{html.escape(query_path)}</p>')
        return
    qidx = data["paths"].index(query_path)
    results = search_by_index(data, qidx, metric=metric,
                              topk=DEFAULT_TOPK, exclude_query=False)

    print("<h2>クエリ画像</h2>")
    print(f'<img class="query" src="{html.escape(img_url(query_path))}" alt="">')
    print(f'<p class="meta">{html.escape(query_path)}<br>'
          f'feature={html.escape(feature)} / metric={html.escape(metric)} / '
          f'N={len(data["paths"])} / D={data["features"].shape[1]}</p>')

    print(f"<h2>類似画像 Top {len(results)}</h2>")
    print('<table class="results"><tr>')
    for rank, (idx, score) in enumerate(results):
        if rank and rank % COLS == 0:
            print("</tr><tr>")
        path = data["paths"][idx]
        link = search_link(path, feature, metric)
        print(f'<td>'
              f'<div class="rank">#{rank+1}</div>'
              f'<a href="{html.escape(link)}">'
              f'<img class="thumb" src="{html.escape(img_url(path))}" alt=""></a>'
              f'<div class="score">{score:.4f}</div>'
              f'</td>')
    print("</tr></table>")


# --------------------------------------------------------------------
def main():
    form = cgi.FieldStorage()
    query = (form.getvalue("q") or "").strip()
    feature = (form.getvalue("feature") or "").strip()
    metric = (form.getvalue("metric") or "").strip()

    feats = available_features()
    if not feats:
        print_header("画像検索")
        print('<p class="err">features/ に .npz がありません</p>')
        print_footer()
        return

    if feature not in feats:
        # デフォルトは dcnn があればそれ、なければ先頭
        feature = "dcnn" if "dcnn" in feats else feats[0]
    if metric not in {k for k, _ in METRICS}:
        metric = "l2"

    # パストラバーサル対策: query は imgdata 配下の既存ファイルのみ許可
    if query:
        safe = os.path.normpath(query)
        if safe.startswith("..") or os.path.isabs(safe):
            query = ""
        else:
            if not os.path.exists(os.path.join(IMGDATA_DIR, safe)):
                query = ""
            else:
                query = safe.replace(os.sep, "/")

    print_header("画像検索")
    render_form(feature, metric, query)
    if query:
        render_results(query, feature, metric)
    else:
        render_gallery(feature, metric)
    print_footer()


if __name__ == "__main__":
    main()
