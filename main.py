import os
import praw
import requests
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

# AI エンジンサポート
import cohere
from openai import OpenAI
import google.generativeai as genai

# dotenv サポート
from dotenv import load_dotenv

load_dotenv()


class RedditSummary(BaseModel):
    """Reddit要約のレスポンス構造"""
    digest: List[str] = Field(
        description="3行の要点まとめ",
        min_items=3,
        max_items=3
    )
    details: str = Field(
        description="キャラクター会話形式の詳細内容"
    )


class RedditClient:
    """Reddit からデータを取得するクライアントクラス"""

    def __init__(self):
        """Reddit API クライアントの初期化"""
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
        )

    def get_hot_posts_with_comments(
        self, subreddit_name: str, limit: int = 3, time_filter: str = "week"
    ) -> str:
        """指定されたサブレディットからホットな投稿とコメントを取得する

        Args:
            subreddit_name: サブレディット名
            limit: 取得する投稿数
            time_filter: 時間フィルター（例: "day", "week", "month"）

        Returns:
            全ての投稿とコメントをテキスト形式で連結した文字列
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        best_posts = list(subreddit.top(limit=limit, time_filter=time_filter))

        all_posts_text = []
        for post in best_posts:
            post_date = datetime.utcfromtimestamp(post.created_utc)
            post_text = [
                f"タイトル: {post.title}",
                f"URL: {post.url}",
                f"投稿日時: {post_date:%Y/%m/%d %H:%M:%S}",
                f"スコア: {post.score}",
                f"コメント数: {post.num_comments}",
                f"本文:\n{post.selftext}",
            ]

            # コメント処理
            comments = post.comments
            comment_list = [
                comment.body
                for comment in comments
                if isinstance(comment, praw.models.Comment)
            ]

            post_text.append("コメントリスト:\n" + "\n".join(comment_list))
            all_posts_text.append("\n".join(post_text))

        return "\n\n".join(all_posts_text)


class AIClient(ABC):
    """AI サービスとのインターフェースを提供する抽象基底クラス"""

    def build_common_messages(self, subreddit: str, text: str) -> List[Dict[str, str]]:
        """共通のメッセージ構造を構築する

        Args:
            subreddit: サブレディット名
            text: 要約するテキスト

        Returns:
            AI サービスに送信するメッセージの構造
        """
        conversation_length = int(os.getenv("CONVERSATION_LENGTH", "15"))

        return [
            {
                "role": "system",
                "content": f"""# Redditトピック要約タスク
    Reddit のホットトピックをキャラクター会話形式で要約します。

    ## 出力フォーマット
    1. 最初に「=== ダイジェスト ===」セクションで、3行の要点をまとめてください
    2. その後「=== 詳細 ===」セクションで、会話形式の詳細を記載してください
    3. 会話は、ずんだもん・四国めたん・東北きりたんによる会話形式で構成します
    4. 地の文は使用せず、会話のみで構成します
    5. 一つのトピックにつき、{conversation_length}回以上の発言を含めてください
    6. トピック間は「---」で区切り、各区切りにはトピックの「タイトル」と「RedditのURL」を含めます

    ## レスポンス構造
    === ダイジェスト ===
    • [1つ目の重要ポイントや話題を1行で簡潔に]
    • [2つ目の重要ポイントや話題を1行で簡潔に]
    • [3つ目の重要ポイントや話題を1行で簡潔に]
    
    === 詳細 ===
    めたん: 今週の r/{subreddit} (https://www.reddit.com/r/{subreddit}/) で話題になっているトピックを紹介していくわ。
    ---
    タイトル: 「Redditのタイトル」
    URL: https://...
    [キャラクター名]: [発言内容]
    [キャラクター名]: [発言内容]
    ...（合計{conversation_length}回以上の発言）
    ---
    （以降、各トピックについて同様の形式で続ける）
    ---
    ずんだもん: （最後のオチ）
    """,
            },
            {
                "role": "system",
                "content": f"""# キャラクター設定

    ## ずんだもん
    * ずんだ餅の妖精
    * 一人称は「ボク」
    * 必ず語尾に「〜のだ」「〜なのだ」をつける（例:「わかったのだ」「大好きなのだ」）
    * フレンドリーかつ優しい言葉遣い（たまに毒舌になることもある）
    * 禁止表現: 「だよ。」「なのだよ。」「！」（多用しない）、「かな？」（代わりに「のだ？」）
    * 特定表現: 「ごめん」→「ごめんなのだ」

    ## 四国めたん
    * 良家のお嬢様の設定
    * タメ口基調だが「〜でしょう」などの言い回しを使用
    * 特徴的な語尾: 「〜かしら。」「〜わね。」「〜わよ。」「〜なのよ。」
    * たまに厨二病的な発言をする

    ## 東北きりたん
    * 11歳の女性だが、しっかり者
    * 丁寧な言葉遣いを使用

    ## 会話の進行方法
    1. 最初の発言は必ず: 「めたん: 今週の r/{subreddit} (https://www.reddit.com/r/{subreddit}/) で話題になっているトピックを紹介していくわ」
    2. 各トピックでは3人全員が会話に参加すること
    3. キャラクターの発言順序はトピックごとにランダムに変更
    4. 最後はずんだもんがオチをつけて終了
    """,
            },
            {"role": "user", "content": text},
        ]

    @abstractmethod
    def summarize_text(self, subreddit: str, text: str) -> Tuple[RedditSummary, str]:
        """テキストを要約する

        Args:
            subreddit: サブレディット名
            text: 要約するテキスト

        Returns:
            (RedditSummary, モデル名)のタプル
        """
        pass


class OpenAIChatClient(AIClient):
    """OpenAI API クライアント"""

    def __init__(self):
        """OpenAI API クライアントの初期化"""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("AI_MODEL")

    def summarize_text(self, subreddit: str, text: str) -> Tuple[RedditSummary, str]:
        """OpenAI API を使用してテキストを要約する

        Args:
            subreddit: サブレディット名
            text: 要約するテキスト

        Returns:
            (RedditSummary, モデル名)のタプル
        """
        conversation_length = int(os.getenv("CONVERSATION_LENGTH", "15"))

        # Structured Outputs用のシステムメッセージ
        system_message = f"""あなたはRedditのトピックを要約するアシスタントです。

以下のキャラクターを使った会話形式で要約してください：
- ずんだもん: 語尾に「〜のだ」「〜なのだ」をつける
- 四国めたん: 「〜かしら」「〜わね」「〜わよ」の語尾を使う
- 東北きりたん: 丁寧な言葉遣い

詳細のフォーマット:
めたん: 今週の r/{subreddit} (https://www.reddit.com/r/{subreddit}/) で話題になっているトピックを紹介していくわ。
---
タイトル: 「[Redditのタイトル]」
URL: [URL]
[キャラクター名]: [発言内容]
（{conversation_length}回以上の発言）
---
（各トピックを同様に）
---
ずんだもん: [オチ]"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]

        # Structured Outputsを使用
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=RedditSummary,
            )
            return response.choices[0].message.parsed, self.model
        except Exception as e:
            # Structured Outputsがサポートされていない場合のフォールバック
            print(f"Structured Output failed, falling back to JSON mode: {e}")

            # JSONモードで再試行
            json_messages = [
                {"role": "system", "content": system_message + "\n\nJSON形式で返答してください: {\"digest\": [3つの要点], \"details\": \"詳細\"}"},
                {"role": "user", "content": text}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=json_messages,
                response_format={"type": "json_object"}
            )

            import json
            content = response.choices[0].message.content
            data = json.loads(content)
            return RedditSummary(**data), self.model


class CohereChatClient(AIClient):
    """Cohere API クライアント"""

    def __init__(self):
        """Cohere API クライアントの初期化"""
        self.api_key = os.getenv("COHERE_API_KEY")
        self.client = cohere.Client(self.api_key)
        self.model = os.getenv("AI_MODEL")

    def _convert_messages_format(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """メッセージ形式を Cohere API 形式に変換する

        Args:
            messages: 標準形式のメッセージリスト

        Returns:
            Cohere API 形式のメッセージリスト
        """
        return [{"role": msg["role"], "text": msg["content"]} for msg in messages]

    def summarize_text(self, subreddit: str, text: str) -> Tuple[RedditSummary, str]:
        """Cohere API を使用してテキストを要約する

        Args:
            subreddit: サブレディット名
            text: 要約するテキスト

        Returns:
            (RedditSummary, モデル名)のタプル
        """
        import json

        common_messages = self.build_common_messages(subreddit, text)
        # JSONレスポンスを要求するメッセージを追加
        common_messages[-1]["content"] += "\n\nJSON形式で返答してください: {\"digest\": [「要点1」, 「要点2」, 「要点3」], \"details\": \"詳細内容\"}"
        messages = self._convert_messages_format(common_messages)

        response = self.client.chat(
            model=self.model,
            chat_history=messages,
            message="指示に従ってJSON形式で要約してください",
            temperature=1.0,
        )

        try:
            data = json.loads(response.text)
            return RedditSummary(**data), self.model
        except:
            # JSONパースに失敗した場合のフォールバック
            return self._parse_text_response(response.text), self.model

    def _parse_text_response(self, content: str) -> RedditSummary:
        """テキストレスポンスをRedditSummaryにパース"""
        digest = ["要約を生成中...", "要約を生成中...", "要約を生成中..."]
        details = content
        return RedditSummary(digest=digest, details=details)


class GeminiChatClient(AIClient):
    """Google Gemini API クライアント"""

    def __init__(self):
        """Gemini API クライアントの初期化"""
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(os.getenv("AI_MODEL"))
        self.model_name = os.getenv("AI_MODEL")

    def summarize_text(self, subreddit: str, text: str) -> Tuple[RedditSummary, str]:
        """Gemini API を使用してテキストを要約する

        Args:
            subreddit: サブレディット名
            text: 要約するテキスト

        Returns:
            (RedditSummary, モデル名)のタプル
        """
        import json

        messages = self.build_common_messages(subreddit, text)
        # JSONレスポンスを要求するプロンプトを追加
        prompt_with_json = " ".join([msg["content"] for msg in messages])
        prompt_with_json += "\n\nJSON形式で返答してください: {\"digest\": [「要点1」, 「要点2」, 「要点3」], \"details\": \"詳細内容\"}"

        response = self.model.generate_content(prompt_with_json)

        try:
            # JSON部分を抽出してパース
            text = response.text
            # JSONブロックを探す
            if '{' in text and '}' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                json_str = text[start:end]
                data = json.loads(json_str)
                return RedditSummary(**data), self.model_name
        except:
            pass

        # JSONパースに失敗した場合のフォールバック
        return self._parse_text_response(response.text), self.model_name

    def _parse_text_response(self, content: str) -> RedditSummary:
        """テキストレスポンスをRedditSummaryにパース"""
        digest = ["要約を生成中...", "要約を生成中...", "要約を生成中..."]
        details = content
        return RedditSummary(digest=digest, details=details)


class SlackNotifier:
    """Slack 通知クライアント"""

    def __init__(self):
        """Slack API クライアントの初期化"""
        self.token = os.getenv("SLACK_BOT_TOKEN")
        self.channel = os.getenv("SLACK_CHANNEL")
        self.url = "https://slack.com/api/chat.postMessage"

    def send_message(self, text: str, thread_ts: Optional[str] = None) -> Optional[str]:
        """Slack にメッセージを送信する

        Args:
            text: 送信するテキスト
            thread_ts: スレッド ID (オプション)

        Returns:
            送信成功時はメッセージ ID、失敗時は None
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "channel": self.channel,
            "text": text,
        }

        if thread_ts:
            payload["thread_ts"] = thread_ts

        response = requests.post(self.url, headers=headers, json=payload)

        if response.status_code == 200 and response.json().get("ok"):
            return response.json().get("ts")
        else:
            print(f"エラーが発生しました: {response.status_code}: {response.text}")
            return None


def create_ai_client(ai_engine: str) -> AIClient:
    """AI エンジン名に基づいて適切な AI クライアントを作成する

    Args:
        ai_engine: AI エンジン名 ("openai", "cohere", "gemini")

    Returns:
        AIClient インスタンス

    Raises:
        ValueError: サポートされていない AI エンジンが指定された場合
    """
    if ai_engine == "openai":
        return OpenAIChatClient()
    elif ai_engine == "cohere":
        return CohereChatClient()
    elif ai_engine == "gemini":
        return GeminiChatClient()
    else:
        raise ValueError(f"サポートされていない AI エンジン: {ai_engine}")


class Application:
    """メインアプリケーションクラス"""

    def __init__(self):
        """アプリケーションの初期化"""
        ai_engine = os.getenv("AI_ENGINE", "openai")
        self.ai_client = create_ai_client(ai_engine)
        self.reddit_client = RedditClient()
        self.slack_notifier = SlackNotifier()

    def run(self, subreddit_name: str, limit: int) -> None:
        """アプリケーションを実行する

        Args:
            subreddit_name: サブレディット名
            limit: 取得する投稿数
        """
        try:
            # Reddit からデータを取得
            all_posts_text = self.reddit_client.get_hot_posts_with_comments(
                subreddit_name, limit
            )

            # AI による要約 (structured output)
            summary_response, model_name = self.ai_client.summarize_text(subreddit_name, all_posts_text)

            # ダイジェストを整形
            digest_formatted = "\n".join([f"• {line}" for line in summary_response.digest])

            # Slack に通知
            # 最初のメッセージにダイジェストを含める
            first_message = f"📊 今週の r/{subreddit_name}\n\n{digest_formatted}"
            thread_ts = self.slack_notifier.send_message(first_message)
            if thread_ts:
                # 詳細とモデル名を追加
                details_with_model = f"{summary_response.details}\n\n使用モデル: {model_name}"
                self.slack_notifier.send_message(details_with_model, thread_ts)
            else:
                print("Slack への通知に失敗しました。")

        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            sys.exit(1)


def main():
    """メイン関数"""
    if len(sys.argv) > 1:
        subreddit_name = sys.argv[1]
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3

        app = Application()
        app.run(subreddit_name, limit)
    else:
        print("使用法: python script.py <subreddit_name> [limit]")
        sys.exit(1)


if __name__ == "__main__":
    main()
