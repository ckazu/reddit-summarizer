import os
import sys
from datetime import datetime
import requests
import praw
from openai import OpenAI
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


class OpenAIChatClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("LLM_MODEL")

    def build_messages(self, subreddit, text):
        if os.getenv("CONVERSATION_LENGTH"):
            conversation_length = int(os.getenv("CONVERSATION_LENGTH"))

        return [
            # {"role": "system", "content": "次の情報からホットトピックを最大 10 件選定し、それぞれに対してどういう内容でどういうコメントがあるか日本語で要約してください。要約は3文程度にまとめますが、短すぎないようにしてください。また、タイトルとURLの情報も付与してください。この要約は、Slack に投稿されます。"},
            {
                "role": "system",
                "content": "Reddit のホットトピックの情報をこれから提示します。トピック本文とコメントから、全体の概要を要約し、どのような議論が行われているかを日本語で会話形式で紹介します。",
            },
            {
                "role": "system",
                "content": "これから提示する会話ログの全件を一件ずつ処理してください。",
            },
            {
                "role": "system",
                "content": "会話は、ずんだもん（ずんだもん）と四国めたん（めたん）、東北きりたん（きりたん）による会話形式で紹介します。会話の順番は同じにならないようにトピックごとにランダムに変更してください。",
            },
            {
                "role": "system",
                "content": "地の文は必要ありません。会話のみで構成してください。",
            },
            {
                "role": "system",
                "content": f"ひとつのトピックにつき、{conversation_length} 回以上の発言をしてください。",
            },
            {
                "role": "system",
                "content": "トピックとトピックの間には、区切りを入れます。区切りの行には、トピックの「タイトル」とトピックの「Reddit URL」を付与してください。",
            },
            {
                "role": "system",
                "content": f"""ずんだもんの特徴を指示します。 **これは最も重要な指示です。** 次の特徴を必ず守ってください。
                                            ずんだもんは、ずんだ餅の妖精です。
                                            一人称は「ボク」です。
                                            必ず語尾に「〜のだ」や「〜なのだ」とつけて話します。例:「わかったのだ」「大好きなのだ」。
                                            フレンドリーな性格で敬語は使用しません。たまにやさぐれて毒舌になりますが、基本的には優しい言葉遣いを使います。
                                            「だよ。」「なのだよ。」という表現は禁止します。
                                            みだりに！はつけません。
                                            「ごめん」は「ごめんなのだ」とします。
                                            「かな？」は使用しません。例:「質問はあるかな？」は「質問はあるのだ？」とします。""",
            },
            {
                "role": "system",
                "content": f"""めたんの特徴を指示します。めたんは良家のお嬢様でした。
                                            基本的にはタメ口ですが、元お嬢様らしく「〜でしょう」などといった言い回しをします。
                                            「〜かしら。」「〜わね。」「〜わよ。」「〜なのよ。」などの語尾を使います。
                                            たまに厨二病な発言をします。""",
            },
            {
                "role": "system",
                "content": f"""きりたんの特徴を指示します。
                                            きりたんは11歳の女性ですが、しっかり者です。丁寧な言葉遣いをします。""",
            },
            {
                "role": "system",
                "content": f"最初の発言は、「めたん: 今週の r/{subreddit} (https://www.reddit.com/r/{subreddit}/) で話題になっているトピックを紹介していくわ」で始めます。",
            },
            {
                "role": "system",
                "content": "最後の発言は、ずんだもんがオチをつけて終わります。",
            },
            {
                "role": "system",
                "content": "会話の作成が完了したら、改めてこれまでの条件にあっているかを確認して会話を修正してください。特に、話し方が指示に則っていることをチェックしてください。特に URL が付与されていることを確認してください。このブラッシュアップの作業を二度行ってください。",
            },
            {
                "role": "system",
                "content": f"""レスポンスは以下の形式で作成してください。
                                            めたん: 今週の r/{subreddit} (https://www.reddit.com/r/{subreddit}/) で話題になっているトピックを紹介していくわ。
                                            ---
                                            タイトル: 「reddit のタイトル」
                                            URL: https://...
                                            発言者: 発話1
                                            発言者: 発話2
                                            ...
                                            発言者: 発話{conversation_length}
                                            ---
                                            タイトル: 「reddit のタイトル」
                                            URL: https://...
                                            発言者: 発話1
                                            発言者: 発話2
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

    def summarize_text(self, subreddit, text):
        messages = self.build_messages(subreddit, text)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return response.choices[0].message.content


class SlackNotifier:
    def __init__(self):
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL")

    def send_message(self, text):
        payload = {"text": text}
        response = requests.post(self.webhook_url, json=payload)
        if response.status_code == 200:
            print("メッセージが送信されました")
        else:
            print(f"エラーが発生しました: {response.status_code}: {response.text}")


class Application:
    def __init__(self):
        self.reddit_client = RedditClient()
        self.openai_client = OpenAIChatClient()
        self.slack_notifier = SlackNotifier()

    def run(self, subreddit_name, limit):
        all_posts_text = self.reddit_client.get_hot_posts_with_comments(
            subreddit_name, limit
        )
        print(all_posts_text)

        summary = self.openai_client.summarize_text(subreddit_name, all_posts_text)
        print(summary)

        self.slack_notifier.send_message(summary)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        subreddit_name = sys.argv[1]
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3

        app = Application()
        app.run(subreddit_name, limit)
    else:
        print("Subreddit名をコマンドライン引数として指定してください。")
        sys.exit(1)
