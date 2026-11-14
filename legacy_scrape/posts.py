import time
import requests
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from fetch_cli.config import BASE_URL

# Reusable HTTP session for connection pooling
_session = requests.Session()

# Essential fields for posts (single row per post)
DEFAULT_POST_COLUMNS = [
    "id",
    "created_utc",
    "created_datetime",
    "title",
    "selftext",
    "score",
    "upvote_ratio",
    "num_comments",
    "author",
    "subreddit",
    "url",
    "permalink",
    "link_flair_text",
    "is_self",
    "over_18",
]

# Fields accepted by Arctic Shift search API for posts (excludes non-database fields)
VALID_POST_API_FIELDS = (
    "id,created_utc,title,selftext,score,num_comments,author,subreddit,url,link_flair_text,over_18"
)


def clean_posts_dataframe(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Filters dataframe to only essential post columns and adds human-readable UTC datetime."""
    if columns is None:
        columns = DEFAULT_POST_COLUMNS

    # Sanitize newlines in text columns for clean single-line CSV rows
    for col in ["title", "selftext"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.replace(r"[\r\n]+", " ", regex=True)
                .str.strip()
            )

    # Add human-readable datetime if created_utc is present
    if "created_utc" in df.columns and "created_datetime" in columns:
        df["created_datetime"] = pd.to_datetime(
            df["created_utc"], unit="s", utc=True
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Select available columns matching the desired list
    existing_cols = [c for c in columns if c in df.columns]
    return df[existing_cols]


def fetch_posts(
    subreddit: str,
    start_time: str,
    end_time: str,
    limit: Union[int, str] = "auto",
    columns: Optional[List[str]] = None,
    top_posts_per_day: Optional[int] = None,
    filter_top_posts: Optional[int] = None,
    sort_posts_by: str = "num_comments",
) -> pd.DataFrame:
    """Fetches posts for a subreddit within a date range and constructs a clean DataFrame (1 row per post).

    Args:
        subreddit: Subreddit name (e.g. 'politics').
        start_time: Start date string 'YYYY-MM-DD'.
        end_time: End date string 'YYYY-MM-DD'.
        limit: Max batch size per request (integer 1-100 or 'auto').
        columns: Specific columns to return in final DataFrame.
        top_posts_per_day: If set (e.g. 3), selects top N most popular posts for each calendar day.
        filter_top_posts: If set (e.g. 3), selects top N most popular posts across the entire period.
        sort_posts_by: Metric for ranking posts ('num_comments' or 'score').
    """
    posts = []

    # Convert date strings to UTC Unix timestamps (integer seconds)
    start_ts = int(
        datetime.strptime(start_time, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    end_ts = int(
        datetime.strptime(end_time, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )

    current_after = start_ts
    url = f"{BASE_URL}/api/posts/search"

    while current_after < end_ts:
        params: Dict[str, Any] = {
            "subreddit": subreddit,
            "after": current_after,
            "before": end_ts,
            "sort": "asc",
            "limit": limit,
            "fields": VALID_POST_API_FIELDS,
        }

        try:
            response = _session.get(url, params=params)
            response.raise_for_status()
            data = response.json().get("data", [])
        except requests.RequestException as e:
            print(f"Post request failed: {e}")
            break

        if not data:
            break

        posts.extend(data)

        # Get timestamp of last post to paginate forward
        last_created_utc = data[-1].get("created_utc")
        if last_created_utc is None or last_created_utc <= current_after:
            current_after += 1
        else:
            current_after = last_created_utc

        print(
            f"Fetched {len(posts)} posts so far (up to {datetime.fromtimestamp(current_after, tz=timezone.utc)})..."
        )
        time.sleep(0.2)  # Respect rate limits

    df = pd.DataFrame(posts)
    if df.empty:
        return clean_posts_dataframe(df, columns=columns)

    # 1. Filter to top N posts per day if requested (e.g. top 3 per day)
    if top_posts_per_day is not None and top_posts_per_day > 0:
        sort_col = sort_posts_by if sort_posts_by in df.columns else "num_comments"
        df["_date"] = pd.to_datetime(
            df["created_utc"], unit="s", utc=True
        ).dt.strftime("%Y-%m-%d")
        print(f"Selecting top {top_posts_per_day} posts per day sorted by {sort_col}...")
        df = (
            df.sort_values(by=["_date", sort_col], ascending=[True, False])
            .groupby("_date", as_index=False)
            .head(top_posts_per_day)
            .drop(columns=["_date"])
            .reset_index(drop=True)
        )

    # 2. Filter to overall top N posts if requested (across whole period)
    elif filter_top_posts is not None and filter_top_posts > 0:
        sort_col = sort_posts_by if sort_posts_by in df.columns else "num_comments"
        print(f"Selecting top {filter_top_posts} posts across period sorted by {sort_col}...")
        df = df.nlargest(filter_top_posts, sort_col).copy().reset_index(drop=True)

    return clean_posts_dataframe(df, columns=columns)
