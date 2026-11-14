from config import BASE_URL, DEFAULT_POST_COLUMNS
import argparse
import asyncio

# from legacy_scrape.comments import (
#     DEFAULT_COMMENT_COLUMNS,
#     fetch_comments_for_post,
#     fetch_comments_for_posts,
#     get_top_comments,
# )
# from legacy_scrape.posts import (
#     DEFAULT_POST_COLUMNS,
#     clean_posts_dataframe,
#     fetch_posts,
# )

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
    parser = argparse.ArgumentParser(description="Resumable, compact Reddit bulk collector")
    parser.add_argument("--subreddits", required=True, help="Comma list or one-subreddit-per-line file")
    parser.add_argument("--start-date", required=True, help="UTC YYYY-MM-DD, inclusive")
    parser.add_argument("--end-date", required=True, help="UTC YYYY-MM-DD, exclusive")
    parser.add_argument("--database", default="data/reddit.sqlite")
    parser.add_argument("--run-id", default="default")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--requests-per-second", type=float, default=1.0)
    parser.add_argument("--no-comments", action="store_true")
    parser.add_argument("--max-comments-per-post", type=int, default=300,
                        help="Use 0 for no cap (exact but potentially much slower)")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.requests_per_second <= 0:
        parser.error("--requests-per-second must be positive")
    if args.max_comments_per_post < 0:
        parser.error("--max-comments-per-post cannot be negative")
    from fetcher import run_bulk_fetch
    from storage import BulkStore
    from subreddits import load_subreddits
    store = BulkStore(args.database)
    try:
        result = asyncio.run(run_bulk_fetch(
            store=store, run_id=args.run_id, subreddits=load_subreddits(args.subreddits),
            start_date=args.start_date, end_date=args.end_date, max_concurrent=args.workers,
            requests_per_second=args.requests_per_second, fetch_comments=not args.no_comments,
            max_comments_per_post=None if args.max_comments_per_post == 0 else args.max_comments_per_post,
            retry_failed=args.retry_failed,
        ))
        print("Run progress:", result)
    finally:
        store.close()
