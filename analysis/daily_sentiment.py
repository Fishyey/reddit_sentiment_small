"""Score Reddit posts with Twitter-RoBERTa and aggregate sentiment by UTC day.

The model's negative, neutral, and positive probabilities are converted to one
continuous sentiment score: ``P(positive) - P(negative)``.  For each post day,
the script saves the mean and sample variance of that score, as well as the
mean probability of each original model class.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class TextDataset(Dataset):
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        return self.texts[index]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/socialtest.sqlite")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="artifacts/daily-sentiment")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.max_length < 4 or args.batch_size < 1:
        parser.error("max length and batch size must be positive")
    return args


def load_posts(database: str, run_id: str) -> pd.DataFrame:
    with sqlite3.connect(database) as connection:
        posts = pd.read_sql_query(
            "SELECT run_id, id, post_day, title, selftext FROM posts WHERE run_id = ?",
            connection,
            params=(run_id,),
            dtype={"run_id": str, "id": str},
        )
    posts["text"] = (posts["title"].fillna("").astype(str) + " " + posts["selftext"].fillna("").astype(str)).str.strip()
    posts = posts[posts["text"].ne("")].reset_index(drop=True)
    if posts.empty:
        raise ValueError("this run has no non-empty posts")
    return posts


def model_indices(model: AutoModelForSequenceClassification) -> dict[str, int]:
    labels = {str(index): str(label).lower() for index, label in model.config.id2label.items()}
    result = {}
    for wanted in ("negative", "neutral", "positive"):
        matching = [int(index) for index, label in labels.items() if wanted in label]
        if len(matching) != 1:
            raise ValueError(f"model must define exactly one {wanted!r} class; found {labels}")
        result[wanted] = matching[0]
    return result


def score_posts(posts: pd.DataFrame, max_length: int, batch_size: int) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    class_index = model_indices(model)

    def collate(texts: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    probabilities = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(TextDataset(posts["text"].tolist()), batch_size=batch_size, collate_fn=collate):
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            probabilities.append(torch.softmax(logits, dim=1).cpu())
    values = torch.cat(probabilities).numpy()
    scored = posts.drop(columns="text").copy()
    for label, index in class_index.items():
        scored[f"{label}_probability"] = values[:, index]
    scored["sentiment_score"] = scored["positive_probability"] - scored["negative_probability"]
    return scored


def aggregate_by_day(scored: pd.DataFrame) -> pd.DataFrame:
    daily = scored.groupby("post_day", as_index=False).agg(
        posts=("id", "size"),
        negative_mean=("negative_probability", "mean"),
        neutral_mean=("neutral_probability", "mean"),
        positive_mean=("positive_probability", "mean"),
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_variance=("sentiment_score", "var"),
    )
    # A day with one retained post has no sample variance; use 0 rather than a blank chart point.
    daily["sentiment_variance"] = daily["sentiment_variance"].fillna(0.0)
    return daily


def main() -> None:
    args = arguments()
    posts = load_posts(args.database, args.run_id)
    scored = score_posts(posts, args.max_length, args.batch_size)
    daily = aggregate_by_day(scored)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output / "post_sentiment.csv", index=False)
    daily.to_csv(output / "daily_sentiment.csv", index=False)
    print(f"Scored {len(scored)} posts across {len(daily)} days; saved to {output}")


if __name__ == "__main__":
    main()
