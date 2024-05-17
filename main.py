import os
import praw
import requests
import sys
from abc import ABC, abstractmethod
from datetime import datetime

# ai engine support
import cohere
from openai import OpenAI
import google.generativeai as genai

# dotenv support
from dotenv import load_dotenv

load_dotenv()


class RedditClient:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
        )

    def get_hot_posts_with_comments(self, subreddit_name, limit=3, time_filter="week"):
        subreddit = self.reddit.subreddit(subreddit_name)
        best_posts = list(subreddit.top(limit=limit, time_filter=time_filter))
        all_posts_text = ""
        for post in best_posts:
            post_date = datetime.utcfromtimestamp(post.created_utc)
            post_text = f"タイトル: {post.title}\nURL: {post.url}\n投稿日時: {post_date:%Y/%m/%d %H:%M:%S}\n"
            post_text += f"スコア: {post.score}\nコメント数: {post.num_comments}\n本文:\n{post.selftext}\n"
            comments = post.comments
            comment_list = [
                comment.body
                for comment in comments
                if isinstance(comment, praw.models.Comment)
            ]
            post_text += "コメントリスト:\n" + "\n".join(comment_list)
            all_posts_text += post_text + "\n\n"
        return all_posts_text


class AIClient(ABC):
    def build_common_messages(self, subreddit, text):
        conversation_length = 15
        if os.getenv("CONVERSATION_LENGTH"):
            conversation_length = int(os.getenv("CONVERSATION_LENGTH"))
        return [
            # {"role": "system", "content": "次の情報からホットトピックを最大 10 件選定し、それぞれに対してどういう内容でどういうコメントがあるか日本語で要約してください。要約は3文程度にまとめますが、短すぎないようにしてください。また、タイトルとURLの情報も付与してください。この要約は、Slack に投稿されます。"},
            {
                "role": "system",
                "content": """Reddit のホットトピックの情報をこれから提示します。トピック本文とコメントから、全体の概要を要約し、どのような議論が行われているかを日本語で会話形式で紹介します。
                これから提示する会話ログの全件を一件ずつ処理してください。
                会話は、ずんだもん（ずんだもん）と四国めたん（めたん）、東北きりたん（きりたん）による会話形式で紹介します。会話の順番は同じにならないようにトピックごとにランダムに変更してください。
                地の文は必要ありません。会話のみで構成してください。
                ひとつのトピックにつき、{conversation_length} 回以上の発言をしてください。
                トピックとトピックの間には、区切りを入れます。区切りの行には、トピックの「タイトル」とトピックの「Reddit URL」を付与してください。
                """,
            },
            {
                "role": "system",
                "content": f"""## キャラクターの特徴

                ### ずんだもんの特徴を指示します。

                **これは最も重要な指示です。**
                次の特徴を必ず守ってください。
                * ずんだもんは、ずんだ餅の妖精です。
                * 一人称は「ボク」です。
                * 必ず語尾に「〜のだ」や「〜なのだ」とつけて話します。例:「わかったのだ」「大好きなのだ」。
                * フレンドリーな性格で敬語は使用しません。たまにやさぐれて毒舌になりますが、基本的には優しい言葉遣いを使います。
                * 「だよ。」「なのだよ。」という表現は禁止します。
                * みだりに！はつけません。
                * 「ごめん」は「ごめんなのだ」とします。
                * 「かな？」は使用しません。例:「質問はあるかな？」は「質問はあるのだ？」とします。

                ### めたんの特徴を指示します。めたんは良家のお嬢様でした。

                * 基本的にはタメ口ですが、元お嬢様らしく「〜でしょう」などといった言い回しをします。
                * 「〜かしら。」「〜わね。」「〜わよ。」「〜なのよ。」などの語尾を使います。
                * たまに厨二病な発言をします。

                ### きりたんの特徴を指示します。

                * きりたんは11歳の女性ですが、しっかり者です。
                * 丁寧な言葉遣いをします。
                """,
            },
            {
                "role": "system",
                "content": f"""## 会話に関する指示
                * 最初の発言は、「めたん: 今週の r/{subreddit} (https://www.reddit.com/r/{subreddit}/) で話題になっているトピックを紹介していくわ」で始めます。",
                * 各 Reddit の紹介では、必ず3人のキャラクターを使って会話を進めてください。
                * 最後は、ずんだもんがオチをつけて終わります。

                ### レスポンスの形式

                めたん: 今週の r/{subreddit} (https://www.reddit.com/r/{subreddit}/) で話題になっているトピックを紹介していくわ。
                ---
                タイトル: 「reddit のタイトル」
                URL: https://...
                発言者: 発話1
                発言者: 発話2
                発言者: 発話3
                ...
                発言者: 発話{conversation_length}
                ---
                タイトル: 「reddit のタイトル」
                URL: https://...
                発言者: 発話1
                発言者: 発話2
                発言者: 発話3
                ...
                発言者: 発話{conversation_length}
                ---
                <すべてのトピックについての会話形式で紹介を続ける。>
                ---
                ずんだもん: <最後のオチをつける。>
                """,
            },
            {"role": "user", "content": text},
        ]

    @abstractmethod
    def summarize_text(self, subreddit, text):
        pass


class OpenAIChatClient(AIClient):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("AI_MODEL")

    def build_messages(self, subreddit, text):
        return self.build_common_messages(subreddit, text)

    def summarize_text(self, subreddit, text):
        messages = self.build_messages(subreddit, text)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return response.choices[0].message.content


class CohereChatClient(AIClient):
    def __init__(self):
        self.api_key = os.getenv("COHERE_API_KEY")
        self.client = cohere.Client(self.api_key)
        self.model = os.getenv("AI_MODEL")

    def build_messages(self, subreddit, text):
        common_messages = self.build_common_messages(subreddit, text)
        modified_messages = [
            {"role": msg["role"], "text": msg["content"]} for msg in common_messages
        ]
        return modified_messages

    def summarize_text(self, subreddit, text):
        messages = self.build_messages(subreddit, text)
        response = self.client.chat(
            model=self.model,
            chat_history=messages,
            message="指示に従って要約してください",
            temperature=1.0,
        )
        return response.text


class GeminiChatClient(AIClient):
    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(os.getenv("AI_MODEL"))

    def build_messages(self, subreddit, text):
        return self.build_common_messages(subreddit, text)

    def summarize_text(self, subreddit, text):
        messages = self.build_messages(subreddit, text)
        plain_prompt = " ".join([msg["content"] for msg in messages])
        response = self.model.generate_content(plain_prompt)
        return response.text


class SlackNotifier:
    def __init__(self):
        self.token = os.getenv("SLACK_BOT_TOKEN")
        self.channel = os.getenv("SLACK_CHANNEL")
        self.url = "https://slack.com/api/chat.postMessage"

    def send_message(self, text, thread_ts=None):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "channel": self.channel,
            "text": text,
            "thread_ts": thread_ts,
        }
        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code == 200 and response.json()["ok"]:
            return response.json()["ts"]
        else:
            print(f"エラーが発生しました: {response.status_code}: {response.text}")
            return None


class Application:
    def __init__(self):
        ai_engine = os.getenv("AI_ENGINE", "openai")
        if ai_engine == "openai":
            self.ai_client = OpenAIChatClient()
        elif ai_engine == "cohere":
            self.ai_client = CohereChatClient()
        elif ai_engine == "gemini":
            self.ai_client = GeminiChatClient()
        else:
            raise ValueError(f"Unsupported AI engine: {ai_engine}")
        self.reddit_client = RedditClient()
        self.slack_notifier = SlackNotifier()

    def run(self, subreddit_name, limit):
        all_posts_text = self.reddit_client.get_hot_posts_with_comments(
            subreddit_name, limit
        )
        print(all_posts_text)

        summary = self.ai_client.summarize_text(subreddit_name, all_posts_text)
        print(summary)

        thread_ts = self.slack_notifier.send_message(f"今週の r/{subreddit_name}", None)
        self.slack_notifier.send_message(summary, thread_ts)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        subreddit_name = sys.argv[1]
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3

        app = Application()
        app.run(subreddit_name, limit)
    else:
        print("Subreddit名をコマンドライン引数として指定してください。")
        sys.exit(1)
