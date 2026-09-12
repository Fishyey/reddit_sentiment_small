from config import BASE_URL
from comments import (
    DEFAULT_COMMENT_COLUMNS,
    fetch_comments_for_post,
    fetch_comments_for_posts,
    get_top_comments,
)
from posts import (
    DEFAULT_POST_COLUMNS,
    clean_posts_dataframe,
    fetch_posts,
)

# Backward-compatibility alias
DEFAULT_COLUMNS = DEFAULT_POST_COLUMNS

__all__ = [
    "BASE_URL",
    "DEFAULT_POST_COLUMNS",
    "DEFAULT_COLUMNS",
    "DEFAULT_COMMENT_COLUMNS",
    "clean_posts_dataframe",
    "fetch_posts",
    "fetch_comments_for_post",
    "fetch_comments_for_posts",
    "get_top_comments",
]


if __name__ == "__main__":
    # 1. Fetch posts (top 3 posts per calendar day)
    posts_df = fetch_posts(
        subreddit="politics",
        start_time="2025-10-01",
        end_time="2025-10-05",
        top_posts_per_day=3,       # Get top 3 posts per day (e.g. 4 days = 12 posts)
        sort_posts_by="num_comments",
    )
    posts_df.to_csv("politics_posts.csv", index=False)
    print(f"Saved {len(posts_df)} posts to politics_posts.csv")

    # 2. Fetch comments for those posts (with post_id feature)
    comments_df = fetch_comments_for_posts(
        post_ids=posts_df["id"],
        top_n_per_post=3,          # Top 3 most replied comments per post (or None for all)
        metric="replies",
    )
    comments_df.to_csv("politics_comments.csv", index=False)
    print(f"Saved {len(comments_df)} comments to politics_comments.csv")

