import os
import sys
import requests
import praw
from openai import OpenAI
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

def summarize_text(subreddit, text):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)

    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            # {"role": "system", "content": "次の情報からホットトピックを最大 10 件選定し、それぞれに対してどういう内容でどういうコメントがあるか日本語で要約してください。要約は3文程度にまとめますが、短すぎないようにしてください。また、タイトルとURLの情報も付与してください。この要約は、Slack に投稿されます。"},
            {"role": "system", "content": "次の情報からホットトピックを選定し、それぞれに対してどういう内容かを日本語で要約します。"},
            {"role": "system", "content": "ピックアップするトピックはスコアとコメント数を基準に選んでください。"},
            {"role": "system", "content": "ピックアップするトピックは全部で 5 つ選びます。候補が 5 つに満たない場合は、コメント数が多いものから選んでください。"},
            {"role": "system", "content": "要約は、ずんだもん（ずんだもん）と四国めたん（めたん）、東北きりたん（きりたん）による会話形式で紹介します。会話の順番は同じにならないようにトピックごとにランダムに変更してください。"},
            {"role": "system", "content": "地の文は必要ありません。会話のみで構成してください。"},
            {"role": "system", "content": "ひとつのトピックにつき、6 回以上の発言をしてください。"},
            {"role": "system", "content": "トピックとトピックの間には、区切りを入れます。区切りの行には、トピックの「タイトル」とトピックの「Reddit URL」を付与してください。"},
            {"role": "system", "content": f"""ずんだもんの特徴を指示します。これは最も重要な指示です。以下の特徴を必ず守ってください。
                                            ずんだもんは、ずんだ餅の妖精です。
                                            一人称は「ボク」です。
                                            必ず語尾に「〜のだ」や「〜なのだ」とつけて話します。例:「わかったのだ」「大好きなのだ」。
                                            フレンドリーな性格で敬語は使用しません。たまにやさぐれて毒舌になりますが、基本的には優しい言葉遣いを使います。
                                            「だよ。」「なのだよ。」という表現は禁止します。
                                            みだりに！はつけません。
                                            「ごめん」は「ごめんなのだ」とします。
                                            「かな？」は使用しません。例:「質問はあるかな？」は「質問はあるのだ？」とします。"""},
            {"role": "system", "content": f"""めたんの特徴を指示します。めたんは良家のお嬢様でした。
                                            基本的にはタメ口ですが、元お嬢様らしく「〜でしょう」などといった言い回しをします。
                                            「〜かしら。」「〜わね。」「〜わよ。」「〜なのよ。」などの語尾を使います。
                                            たまに厨二病な発言をします。"""},
            {"role": "system", "content": f"""きりたんの特徴を指示します。
                                            きりたんは11歳の女性ですが、しっかり者です。丁寧な言葉遣いをします。"""},
            {"role": "system", "content": f"最初の発言は、「めたん: 今週の r/{subreddit} (https://www.reddit.com/r/{subreddit}/) で話題になっているトピックを紹介していくわ」で始めます。"},
            {"role": "system", "content": "最後の発言は、ずんだもんがオチをつけて終わります。"},
            {"role": "system", "content": "会話の作成が完了したら、改めてこれまでの条件にあっているかを確認して会話を修正してください。特に、話し方が指示に則っていることをチェックしてください。特に URL が付与されていることを確認してください。このブラッシュアップの作業を二度行ってください。"},
            {"role": "system", "content": f"""レスポンスは以下の形式で作成してください。
                                            めたん: 今週の r/{subreddit} (https://www.reddit.com/r/{subreddit}/) で話題になっているトピックを紹介していくわ。
                                            ---
                                            タイトル: 「reddit のタイトル」
                                            URL: https://...
                                            発言者: 発話1
                                            発言者: 発話2
                                            発言者: 発話3
                                            発言者: 発話4
                                            発言者: 発話5
                                            ---
                                            タイトル: 「reddit のタイトル」
                                            URL: https://...
                                            発言者: 発話1
                                            発言者: 発話2
                                            発言者: 発話3
                                            発言者: 発話4
                                            発言者: 発話5
                                            ---
                                            <最大５件のトピックについての会話を続ける。>
                                            ---
                                            ずんだもん: <最後のオチをつける。>
                                            """},
            {"role": "user", "content": text}
        ]
    )

    return response.choices[0].message.content

def get_hot_posts_with_comments(subreddit_name, limit=10):
    reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
    reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    reddit_user_agent = os.getenv("REDDIT_USER_AGENT")

    reddit = praw.Reddit(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        user_agent=reddit_user_agent,
    )
    subreddit = reddit.subreddit(subreddit_name)
    best_posts = list(subreddit.top(limit=limit, time_filter="week"))
    # hot_posts = list(subreddit.hot(limit=limit))

    all_posts = best_posts # + hot_posts

    all_posts_text = ""
    for post in all_posts:
        # 各投稿の情報を結合
        post_text = f"タイトル: {post.title}\nURL: {post.url}\nスコア: {post.score}\nコメント数: {post.num_comments}\n本文: {post.selftext}\n"

        # コメントリストを取得して結合
        comments = post.comments
        comment_list = []
        for comment in comments:
            if isinstance(comment, praw.models.Comment):
                comment_list.append(comment.body)

        post_text += "コメントリスト:\n" + "\n".join(comment_list)
        all_posts_text += post_text + "\n\n"

    return all_posts_text

def send_message(text):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    payload = {
        "text": text
    }
    response = requests.post(webhook_url, json=payload)
    if response.status_code == 200:
        print("メッセージが送信されました")
    else:
        print(f"エラーが発生しました: {response.status_code}: {response.text}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        subreddit_name = sys.argv[1]
        all_posts_text = get_hot_posts_with_comments(subreddit_name)
        print(all_posts_text)
        summary = summarize_text(subreddit_name, all_posts_text)
        print(summary)
        send_message(summary)
    else:
        print("Subreddit名をコマンドライン引数として指定してください。")
        sys.exit(1)
