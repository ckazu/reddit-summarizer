# reddit-summarizer

## 開発のセットアップ

### 要件

* Python 3.7以上
* (option) github actions

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

ローカルで関数をテストするには、 `main.py` ファイルを実行します。

```bash
python main.py <subreddit名>
```
