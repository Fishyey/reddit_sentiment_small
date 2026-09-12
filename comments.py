import time
import requests
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from collections import Counter
from config import BASE_URL

# Reusable HTTP session for connection pooling
_session = requests.Session()

# Essential fields for comments
DEFAULT_COMMENT_COLUMNS = [
    "post_id",
    "id",
    "created_utc",
    "created_datetime",
    "author",
    "score",
    "num_replies",
    "body",
    "parent_id",
    "link_id",
    "permalink",
    "subreddit",
]

# Fields supported by Arctic Shift search API for comments
VALID_COMMENT_API_FIELDS = "id,created_utc,author,score,body,parent_id,link_id,subreddit"


def fetch_comments_for_post(
    post_id: str,
    limit: Union[int, str] = "auto",
    max_comments: Optional[int] = 300,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fetches comments for a given post ID, computes reply counts, and attaches post_id."""
    if columns is None:
        columns = DEFAULT_COMMENT_COLUMNS

    clean_id = post_id.replace("t3_", "")
    url = f"{BASE_URL}/api/comments/search"
    comments: List[Dict[str, Any]] = []
    current_after = 0

    while True:
        params: Dict[str, Any] = {
            "link_id": clean_id,
            "limit": limit,
            "sort": "asc",
            "fields": VALID_COMMENT_API_FIELDS,
        }
        if current_after:
            params["after"] = current_after

        try:
            response = _session.get(url, params=params)
            response.raise_for_status()
            data = response.json().get("data", [])
        except requests.RequestException as e:
            print(f"Comment request failed for post {clean_id}: {e}")
            break

        if not data:
            break

        comments.extend(data)

        # Stop early if max_comments reached
        if max_comments is not None and len(comments) >= max_comments:
            comments = comments[:max_comments]
            break

        last_created_utc = data[-1].get("created_utc")
        if last_created_utc is None or last_created_utc <= current_after:
            current_after += 1
        else:
            current_after = last_created_utc

        time.sleep(0.2)

    if not comments:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(comments)

    # Attach post_id foreign key
    df["post_id"] = clean_id

    # Compute direct reply counts by counting references in parent_id
    parent_counts = Counter()
    for c in comments:
        pid = str(c.get("parent_id", ""))
        if pid.startswith("t1_"):
            parent_counts[pid[3:]] += 1
        elif pid and not pid.startswith("t3_"):
            parent_counts[pid] += 1

    df["num_replies"] = df["id"].map(parent_counts).fillna(0).astype(int)

    # Sanitize newlines and whitespace in text fields for clean single-line CSV rows
    if "body" in df.columns:
        df["body"] = (
            df["body"]
            .astype(str)
            .str.replace(r"[\r\n]+", " ", regex=True)
            .str.strip()
        )

    if "created_utc" in df.columns:
        df["created_datetime"] = pd.to_datetime(
            df["created_utc"], unit="s", utc=True
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

    existing_cols = [c for c in columns if c in df.columns]
    return df[existing_cols]


def get_top_comments(
    comments_df: pd.DataFrame,
    n: int = 3,
    metric: str = "replies",
) -> pd.DataFrame:
    """Returns top N comments by most replies, upvotes, or downvotes (lowest score).

    Args:
        comments_df: DataFrame of comments from fetch_comments_for_post.
        n: Number of comments to return.
        metric: 'replies', 'upvoted', or 'downvoted'.
    """
    if comments_df.empty:
        return comments_df

    if metric == "replies":
        return comments_df.nlargest(n, "num_replies")
    elif metric == "upvoted":
        return comments_df.nlargest(n, "score")
    elif metric == "downvoted":
        return comments_df.nsmallest(n, "score")
    else:
        raise ValueError(
            f"Unknown metric: {metric}. Choose from 'replies', 'upvoted', or 'downvoted'."
        )


def fetch_comments_for_posts(
    post_ids: Union[List[str], pd.Series],
    top_n_per_post: Optional[int] = 3,
    metric: str = "replies",
    max_comments_per_post: Optional[int] = 300,
) -> pd.DataFrame:
    """Fetches comments for multiple posts and returns a combined DataFrame with post_id.

    Args:
        post_ids: List or Series of post IDs.
        top_n_per_post: If set (e.g. 3), keeps only top N comments per post. None fetches all comments.
        metric: Metric for top comments selection ('replies', 'upvoted', 'downvoted').
        max_comments_per_post: Maximum comments to scan per post before ranking.
    """
    all_comments: List[pd.DataFrame] = []
    total_posts = len(post_ids)
    print(f"Fetching comments for {total_posts} posts (top {top_n_per_post} per post)...")

    for i, post_id in enumerate(post_ids, start=1):
        clean_id = str(post_id).replace("t3_", "")
        df_c = fetch_comments_for_post(
            clean_id, limit="auto", max_comments=max_comments_per_post
        )
        if not df_c.empty:
            if top_n_per_post is not None:
                df_c = get_top_comments(df_c, n=top_n_per_post, metric=metric)
            all_comments.append(df_c)

        if i % 10 == 0 or i == total_posts:
            print(f"Fetched comments for {i}/{total_posts} posts...")

    if not all_comments:
        return pd.DataFrame(columns=DEFAULT_COMMENT_COLUMNS)

    return pd.concat(all_comments, ignore_index=True)
