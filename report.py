"""Show the progress and compact output stored by the bulk collector."""

import argparse
import json

from fetch_cli.storage import BulkStore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a Reddit bulk-collection SQLite database")
    parser.add_argument("--database", default="data/reddit.sqlite")
    parser.add_argument("--run-id", default="default")
    parser.add_argument("--limit", type=int, default=1, help="Rows to preview from each output table")
    args = parser.parse_args()
    store = BulkStore(args.database)
    try:
        print("Progress:", store.progress(args.run_id))
        for table in ("posts", "comments"):
            rows = [dict(row) for row in store.preview(args.run_id, table, args.limit)]
            print(f"\nLatest {table} ({len(rows)} rows):")
            print(json.dumps(rows, indent=2, ensure_ascii=False))
    finally:
        store.close()
