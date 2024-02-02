# reddit-summarizer

## 開発のセットアップ

### 要件

* Python 3.7以上
* gcloud

### Python仮想環境の作成とアクティベート

```bash
python -m venv venv
source venv/bin/activate  # Linux/Macの場合
venv\Scripts\activate  # Windowsの場合
```

### 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 環境変数の設定

`.env.example` を `.env` にコピーして内容を修正します。

### ローカルでのテスト

ローカルで関数をテストするには、 `local_test.py` ファイルを実行します。

```bash
python local_test.py
```

## デプロイ

```bash
gcloud functions deploy [関数名] --runtime python39 --trigger-http --allow-unauthenticated
```
