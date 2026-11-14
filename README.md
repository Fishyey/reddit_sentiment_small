# Reddit sentiment analysis

This project collects a small, ranked sample of Reddit posts and comments through the [Arctic Shift API](https://arctic-shift.photon-reddit.com/), stores it in SQLite, and provides command-line scripts for sentiment scoring, exploratory clustering, and supervised multi-label training. Collection and analysis are separate steps: the SQLite database contains text and metadata, not ground-truth sentiment labels.

## Setup

Run commands from the repository root. Use a Python version supported by your installed PyTorch and Transformers packages (the code requires Python 3.10 or newer).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first analysis run downloads the relevant Hugging Face model files and can take time and disk space. The scripts use CUDA, Apple MPS, or CPU, in that order of availability. No Reddit credentials are required for the collector.

## Collect posts and comments

`fetch_cli/main.py` accepts either a comma-separated subreddit list or a UTF-8 file with one subreddit per line. Blank lines, comment lines starting with `#`, and an optional `r/` prefix are accepted. The repository includes an example `subreddits.txt`.

```bash
python fetch_cli/main.py \
  --subreddits subreddits.txt \
  --start-date 2025-01-01 --end-date 2025-01-08 \
  --database data/reddit.sqlite --run-id pilot-week \
  --workers 2 --requests-per-second 0.5
```

Dates are UTC. The start date is inclusive and the end date is exclusive, so this example covers seven days. Each subreddit-day is one checkpoint. By default, the collector retains the three posts with the most comments per day and the three comments with the most direct replies per retained post. It scans at most 300 comments per post before ranking them; `--max-comments-per-post 0` removes that cap and can be much slower. `--no-comments` skips comments entirely. API pages and posts that were not selected are not saved.

The SQLite database has `runs` (collection settings), `work_items` (checkpoints and errors), `posts`, and `comments` tables. Inspect progress and sample rows with:

```bash
python report.py --database data/reddit.sqlite --run-id pilot-week --limit 3
```

Rerun the same collection command to resume pending or interrupted work. Add `--retry-failed` to revisit failed subreddit-days. A run ID is bound to its original subreddit list, date range, and selection settings; use a new run ID when those change. Worker count and request pace may change between resumptions. Avoid running two collectors against the same run ID at once. The collector retries transient API failures and applies a shared request rate limit across workers.

Use `python fetch_cli/main.py --help` for all collection options. Collection time depends heavily on the date range, number of subreddits, comments, and API response time.

## Analyze a collected run

All three analysis commands below use the collector's SQLite database and the same `--run-id`. Their code defaults to `data/socialtest.sqlite`, so pass `--database` when using the collection example above. They score post title plus self-text; empty posts are omitted. CSV outputs are ignored by Git.

### Daily sentiment

```bash
python analysis/daily_sentiment.py \
  --database data/reddit.sqlite --run-id pilot-week \
  --output-dir artifacts/daily-sentiment
```

This uses `cardiffnlp/twitter-roberta-base-sentiment-latest` to estimate negative, neutral, and positive probabilities. Each post receives `sentiment_score = P(positive) - P(negative)`. It writes `post_sentiment.csv` and `daily_sentiment.csv`, including daily means and sample variance. This is model-inferred sentiment, not a human label.

### Explore post clusters

```bash
python analysis/cluster_posts.py \
  --database data/reddit.sqlite --run-id pilot-week \
  --clusters 5 --pca-components 5 \
  --output-dir artifacts/post-clusters
```

The script embeds posts with Twitter-RoBERTa, reduces the vectors with PCA, and groups them with cosine k-means. It writes `post_clusters.csv`, `cluster_summary.json` (terms and representative posts), and `pca_transform.pt`. The number of clusters and PCA components cannot exceed the number of usable posts. Cluster IDs have no inherent sentiment meaning; inspect the representative posts before interpreting them.

### Train a 20-label classifier

The supervised script requires your own labels CSV with `run_id`, `id`, and **exactly 20** additional columns containing only `0` or `1`. Each `(run_id, id)` must be unique and match a collected post. It needs at least 10 matching, non-empty posts spanning at least three distinct days for its chronological train, validation, and test splits.

```bash
python analysis/train_multilabel.py \
  --database data/reddit.sqlite --run-id pilot-week \
  --labels-csv data/post_labels.csv \
  --output-dir artifacts/multilabel-roberta
```

It fine-tunes Twitter-RoBERTa with a multi-label classification head, saves the best validation model and tokenizer, and writes `training_metadata.json` with label names, split dates, and test metrics. Optional `--mask-locations --masked-text-cache data/masked_posts.csv` runs a separate named-entity model and caches location-masked text. Review `python analysis/train_multilabel.py --help` for training settings.

## Project layout and local files

| Path | Purpose |
| --- | --- |
| `fetch_cli/` | Arctic Shift bulk collector, input parsing, and SQLite storage |
| `report.py` | Collection progress and row preview |
| `analysis/daily_sentiment.py` | Per-post and per-day sentiment scores |
| `analysis/cluster_posts.py` | Exploratory PCA and clustering |
| `analysis/train_multilabel.py` | Supervised classifier training |
| `analysis/data_init.py`, `analysis/tokenizer.py` | Older exploratory scripts with local database assumptions |
| `legacy_scrape/` | Earlier scraping helpers |
| `artifacts/` | Analysis output; some example summaries and an HTML visualization are present |

Local databases in `data/`, CSV files, the virtual environment, logs, `.env` files, and the personal `guideForMe` notes are ignored by Git. Keep a separate backup of a database or trained model that you need to retain. The API and pretrained model can change over time, so record run settings and model versions when comparing results across runs.
