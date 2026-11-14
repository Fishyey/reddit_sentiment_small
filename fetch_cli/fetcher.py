"""Async, low-disk bulk collection from the Arctic Shift API."""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any

# pyrefly: ignore [missing-import]
import aiohttp

from config import BASE_URL, VALID_COMMENT_API_FIELDS, VALID_POST_API_FIELDS
from storage import BulkStore


class ApiError(RuntimeError):
    pass


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.interval = 1 / requests_per_second
        self.next_time = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        loop = asyncio.get_running_loop()
        async with self.lock:
            now = loop.time()
            scheduled = max(now, self.next_time)
            self.next_time = scheduled + self.interval
        await asyncio.sleep(max(0, scheduled - loop.time()))

    async def pause(self, seconds: float) -> None:
        """Push back every worker after the upstream service asks us to slow down."""
        async with self.lock:
            self.next_time = max(self.next_time, asyncio.get_running_loop().time() + seconds)


class ArcticClient:
    def __init__(self, session: aiohttp.ClientSession, limiter: RateLimiter, retries: int = 5) -> None:
        self.session, self.limiter, self.retries = session, limiter, retries

    async def get_data(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        url = f"{BASE_URL}{path}"
        for attempt in range(self.retries + 1):
            await self.limiter.wait()
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        payload = await response.json(content_type=None)
                        return payload.get("data", [])
                    retry_after = response.headers.get("Retry-After")
                    # Arctic Shift uses 422 with this message when a search times
                    # out under load; it is a transient overload signal, not a bad query.
                    if response.status not in (422, 429, 500, 502, 503, 504) or attempt == self.retries:
                        raise ApiError(f"HTTP {response.status}: {(await response.text())[:300]}")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60, 2 ** attempt)
                    if response.status == 422:
                        delay = max(delay, min(60, 5 * (2 ** attempt)))
                    await self.limiter.pause(delay)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == self.retries:
                    raise ApiError(str(exc)) from exc
                delay = min(60, 2 ** attempt)
            await asyncio.sleep(delay + random.uniform(0, 0.5))
        raise AssertionError("unreachable")

    async def all_pages(self, path: str, params: dict[str, Any], max_items: int | None = None) -> list[dict[str, Any]]:
        """Page forward by timestamp, retaining pages only while ranking one unit."""
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        # The comments endpoint rejects ``after=0``.  Omit the cursor for its
        # first page, then add it only after a page supplies a real timestamp.
        after = int(params["after"]) if "after" in params else None
        while True:
            page_params = dict(params)
            if after is None:
                page_params.pop("after", None)
            else:
                page_params["after"] = after
            page = await self.get_data(path, page_params)
            if not page:
                return result
            new = [item for item in page if str(item.get("id", "")) not in seen]
            for item in new:
                seen.add(str(item.get("id", "")))
            result.extend(new)
            if max_items is not None and len(result) >= max_items:
                return result[:max_items]
            last = max((int(item.get("created_utc", 0)) for item in page), default=after or 0)
            # We explicitly request 100 records, so a short page is a reliable end signal.
            if len(page) < 100:
                return result
            if after is not None and last <= after:
                raise ApiError("pagination stalled at a shared created_utc timestamp")
            after = last


def _day_bounds(day: str) -> tuple[int, int]:
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(start.timestamp()), int((start + timedelta(days=1)).timestamp())


def _rank(items: list[dict[str, Any]], n: int, metric: str, reverse: bool = True) -> list[dict[str, Any]]:
    def value(item: dict[str, Any]) -> int:
        try:
            return int(item.get(metric) or 0)
        except (TypeError, ValueError):
            return 0
    return sorted(items, key=lambda item: (value(item), str(item.get("id", ""))), reverse=reverse)[:n]


async def fetch_posts_for_day(client: ArcticClient, subreddit: str, day: str, n: int, metric: str) -> list[dict[str, Any]]:
    after, before = _day_bounds(day)
    items = await client.all_pages("/api/posts/search", {
        "subreddit": subreddit, "after": after, "before": before,
        "sort": "asc", "limit": 100, "fields": VALID_POST_API_FIELDS,
    })
    return _rank(items, n, metric)


async def fetch_top_comments(client: ArcticClient, post_id: str, n: int, metric: str, max_comments: int | None) -> list[dict[str, Any]]:
    clean_id = post_id.removeprefix("t3_")
    comments = await client.all_pages("/api/comments/search", {
        "link_id": clean_id, "sort": "asc", "limit": 100, "fields": VALID_COMMENT_API_FIELDS,
    }, max_items=max_comments)
    replies: Counter[str] = Counter()
    for comment in comments:
        parent = str(comment.get("parent_id", "")).removeprefix("t1_")
        if parent:
            replies[parent] += 1
    for comment in comments:
        comment["num_replies"] = replies[str(comment.get("id", "")).removeprefix("t1_")]
        comment["post_id"] = clean_id
    if metric == "replies":
        return _rank(comments, n, "num_replies")
    return _rank(comments, n, "score", reverse=metric != "downvoted")


async def run_bulk_fetch(
    *, store: BulkStore, run_id: str, subreddits: list[str], start_date: str, end_date: str,
    top_posts_per_day: int = 3, sort_posts_by: str = "num_comments", fetch_comments: bool = True,
    top_comments_per_post: int = 3, comment_metric: str = "replies", max_comments_per_post: int | None = 300,
    max_concurrent: int = 5, requests_per_second: float = 2, retry_failed: bool = False,
) -> dict[str, int]:
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    if end <= start:
        raise ValueError("end_date must be after start_date (end is exclusive)")
    config = {"subreddits": subreddits, "start_date": start_date, "end_date": end_date,
              "top_posts_per_day": top_posts_per_day, "sort_posts_by": sort_posts_by,
              "fetch_comments": fetch_comments, "top_comments_per_post": top_comments_per_post,
              "comment_metric": comment_metric, "max_comments_per_post": max_comments_per_post}
    store.create_run(run_id, config)
    recovered = store.recover_interrupted_work(run_id)
    if recovered:
        print(f"Recovered {recovered} interrupted work item(s).")
    days = [(start + timedelta(days=i)).isoformat() for i in range((end - start).days)]
    store.add_work_items(run_id, subreddits, days)
    limiter = RateLimiter(requests_per_second)
    timeout = aiohttp.ClientTimeout(total=90, connect=20, sock_read=60)
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers={"User-Agent": "sentiment-analysis-corr/1.0"}) as session:
        client = ArcticClient(session, limiter)
        async def worker() -> None:
            while item := store.claim_next(run_id, retry_failed):
                subreddit, day = item
                try:
                    posts = await fetch_posts_for_day(client, subreddit, day, top_posts_per_day, sort_posts_by)
                    comments: list[dict[str, Any]] = []
                    if fetch_comments:
                        for post in posts:
                            comments.extend(await fetch_top_comments(client, str(post["id"]), top_comments_per_post, comment_metric, max_comments_per_post))
                    store.save_day(run_id, subreddit, day, posts, comments)
                except Exception as exc:  # retain the failure for a deliberate future retry
                    store.mark_failed(run_id, subreddit, day, f"{type(exc).__name__}: {exc}")
                    print(f"FAILED r/{subreddit} {day}: {exc}")
        await asyncio.gather(*(worker() for _ in range(max_concurrent)))
    return store.progress(run_id)
