# b4kadai 3a — 画像検索CGI課題

OpenCV + PyTorch で特徴量を事前計算し、Webから類似画像検索できるCGIシステム。

## 特徴量 (計10種)

| 種類 | 次元 | 備考 |
| --- | ---: | --- |
| RGB / HSV / LUV ヒストグラム (1x1) | 24 | 8bin × 3ch |
| RGB / HSV / LUV ヒストグラム (2x2) | 96 | 4分割 |
| RGB / HSV / LUV ヒストグラム (3x3) | 216 | 9分割 |
| VGG16 fc2 (DCNN) | 4096 | L2正規化 |

## 距離 / 類似度

- L2 (ユークリッド) 距離 — 小さい順
- ヒストグラムインターセクション `Σmin(q,d)` — 大きい順

## ディレクトリ構成

```
3a/
  crawler.py            # icrawler で画像収集 (Bing)
  extract_features.py   # 10種の特徴量抽出 → .npz
  search.py             # 距離計算 + ランキング
  index.cgi             # CGI本体
www/
  .htaccess             # UEC学内限定 + Basic認証
  imsearch/             # (デプロイ時に作成。中身は全て3a/へのsymlink)
```

## 使い方

```sh
# 1. 画像収集 (~600枚)
python3 3a/crawler.py --out-dir /export/space0/<user>/imgdata

# 2. 特徴量抽出 (全10種)
python3 3a/extract_features.py \
    --img-dir /export/space0/<user>/imgdata \
    --out-dir 3a/features --features all --device cuda

# 3. CGIから検索
# (www/imsearch/ にシンボリックリンクを張ってブラウザからアクセス)
```

## デプロイ (mm.cs.uec.ac.jp)

`~/3a/` をソースの単一実体として、`~/www/imsearch/` には全てsymlinkを張る:

```sh
mkdir -p ~/www/imsearch && cd ~/www/imsearch
ln -sfn ../../3a/index.cgi      index.cgi
ln -sfn ../../3a/search.py      search.py
ln -sfn ../../3a/features       features
ln -sfn /export/space0/<user>/imgdata imgdata
```

`.htaccess` に `FollowSymLinks` があるのでCGIはsymlink経由で実行される。

アクセス: `http://mm.cs.uec.ac.jp/<user>/imsearch/`
