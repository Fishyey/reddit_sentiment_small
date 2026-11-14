"""Input handling for reproducible subreddit collections."""

from pathlib import Path


def load_subreddits(value: str) -> list[str]:
    """Accept a comma-separated list or a UTF-8 text file (one name per line)."""
    source = Path(value)
    raw = source.read_text().splitlines() if source.is_file() else value.split(",")
    seen: set[str] = set()
    result: list[str] = []
    for item in raw:
        name = item.strip().removeprefix("r/")
        if name and not name.startswith("#") and name.lower() not in seen:
            seen.add(name.lower())
            result.append(name)
    if not result:
        raise ValueError("No subreddits supplied")
    return result
