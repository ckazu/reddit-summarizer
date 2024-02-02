import praw
import os
import sys
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

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

    # ホットなトピックを取得
    hot_posts = subreddit.hot(limit=limit)

    for post in hot_posts:
        # タイトル、スコア、コメント数、URLを表示
        print(f"タイトル: {post.title}")
        print(f"スコア: {post.score}")
        print(f"コメント数: {post.num_comments}")
        print(f"URL: {post.url}")

        # 本文を表示
        print(f"本文: {post.selftext}")

        # コメントリストを取得
        comments = post.comments
        comment_list = []
        for comment in comments:
            if isinstance(comment, praw.models.Comment):
                comment_list.append(comment.body)

        # コメントリストを表示
        print("コメントリスト:")
        for i, comment_text in enumerate(comment_list):
            print(f"コメント {i + 1}: {comment_text}\n")

if __name__ == "__main__":
    # コマンドライン引数からSubreddit名を取得
    if len(sys.argv) > 1:
        subreddit_name = sys.argv[1]
        get_hot_posts_with_comments(subreddit_name)
    else:
        print("Subreddit名をコマンドライン引数として指定してください。")
        sys.exit(1)
