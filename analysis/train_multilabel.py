"""Fine-tune Twitter-RoBERTa on explicitly supplied multi-label post targets.

The posts database contains text and collection metadata, not ground-truth
sentiment labels.  Targets therefore come from a CSV with ``run_id``, ``id``,
and one binary column per label.  For example::

    run_id,id,joy,anger,neutral
    test-month-no-askreddit,abc123,1,0,0

This module intentionally uses a small PyTorch loop rather than a notebook so
that tokenisation, chronological splitting, metrics, and saved artifacts are
reproducible from the command line.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LOCATION_TOKEN = "[LOCATION]"
IDENTIFIER_COLUMNS = {"run_id", "id"}


@dataclass(frozen=True)
class TrainingConfig:
    database: str
    run_id: str
    labels_csv: str
    output_dir: str
    max_length: int = 256
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 2e-5
    valid_fraction: float = 0.15
    test_fraction: float = 0.15
    threshold: float = 0.5
    seed: int = 42
    mask_locations: bool = False
    masked_text_cache: str | None = None
    ner_batch_size: int = 16


class EncodedPosts(Dataset):
    def __init__(self, encodings: dict[str, Tensor], labels: Tensor):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {key: value[index] for key, value in self.encodings.items()} | {"labels": self.labels[index]}


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/socialtest.sqlite")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--labels-csv", required=True, help="CSV with run_id, id, and binary label columns.")
    parser.add_argument("--output-dir", default="artifacts/multilabel-roberta")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-locations", action="store_true", help="Replace detected places with [LOCATION] before tokenization.")
    parser.add_argument("--masked-text-cache", help="Optional CSV cache for location-masked text.")
    parser.add_argument("--ner-batch-size", type=int, default=16)
    values = vars(parser.parse_args())
    config = TrainingConfig(**values)
    if not 0 < config.valid_fraction < 1 or not 0 < config.test_fraction < 1:
        parser.error("validation and test fractions must be between zero and one")
    if config.valid_fraction + config.test_fraction >= 1:
        parser.error("validation and test fractions must sum to less than one")
    if config.max_length < 4 or config.batch_size < 1 or config.epochs < 1 or config.ner_batch_size < 1:
        parser.error("max length, batch size, epochs, and NER batch size must be positive")
    return config


def load_examples(config: TrainingConfig) -> tuple[pd.DataFrame, list[str]]:
    """Join post text to validated, multi-hot ground-truth labels."""
    labels = pd.read_csv(config.labels_csv, dtype={"run_id": str, "id": str})
    missing_ids = IDENTIFIER_COLUMNS - set(labels.columns)
    if missing_ids:
        raise ValueError(f"labels CSV is missing identifier columns: {sorted(missing_ids)}")
    label_columns = [column for column in labels.columns if column not in IDENTIFIER_COLUMNS]
    if len(label_columns) != 20:
        raise ValueError(f"expected exactly 20 label columns, found {len(label_columns)}: {label_columns}")
    if labels.duplicated(["run_id", "id"]).any():
        raise ValueError("labels CSV has duplicate (run_id, id) rows")
    for column in label_columns:
        values = pd.to_numeric(labels[column], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"label column {column!r} must contain only 0 or 1")
        labels[column] = values.astype("float32")

    query = """
        SELECT run_id, id, post_day, title, selftext
        FROM posts WHERE run_id = ?
    """
    with sqlite3.connect(config.database) as connection:
        posts = pd.read_sql_query(query, connection, params=(config.run_id,), dtype={"run_id": str, "id": str})
    examples = posts.merge(labels, on=["run_id", "id"], how="inner", validate="one_to_one")
    if examples.empty:
        raise ValueError("no labels matched posts for this run ID")
    examples["text"] = examples["title"].fillna("").astype(str) + " " + examples["selftext"].fillna("").astype(str)
    examples["text"] = examples["text"].str.strip()
    examples = examples[examples["text"].ne("")].copy()
    if len(examples) < 10:
        raise ValueError("need at least 10 non-empty labelled posts to make chronological splits")
    return examples, label_columns


def chronological_split(examples: pd.DataFrame, config: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep later posts fully held out, avoiding leakage from future language."""
    ordered = examples.assign(_day=pd.to_datetime(examples["post_day"], errors="raise")).sort_values(["_day", "id"])
    days = ordered["_day"].drop_duplicates().tolist()
    if len(days) < 3:
        raise ValueError("need posts from at least three distinct days for chronological splitting")
    test_start = int(len(days) * (1 - config.test_fraction))
    valid_start = int(len(days) * (1 - config.test_fraction - config.valid_fraction))
    if not 0 < valid_start < test_start < len(days):
        raise ValueError("split fractions leave an empty day-based partition")
    train_days, valid_days, test_days = set(days[:valid_start]), set(days[valid_start:test_start]), set(days[test_start:])
    train = ordered[ordered["_day"].isin(train_days)]
    valid = ordered[ordered["_day"].isin(valid_days)]
    test = ordered[ordered["_day"].isin(test_days)]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("split fractions leave an empty partition")
    return train, valid, test


def mask_locations(texts: list[str], batch_size: int) -> list[str]:
    """Mask location/GPE spans while preserving all non-location text.

    This model is deliberately loaded only when requested: it is substantially
    larger than the tokenizer and may need to be downloaded on first use.
    """
    from transformers import pipeline

    ner = pipeline(
        "token-classification",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
    )
    results = ner(texts, batch_size=batch_size, truncation=True)
    masked = []
    for text, entities in zip(texts, results):
        characters = list(text)
        locations = sorted(
            (entity for entity in entities if entity.get("entity_group") in {"LOCATION", "GPE"}),
            key=lambda entity: entity["start"], reverse=True,
        )
        for entity in locations:
            characters[entity["start"]:entity["end"]] = LOCATION_TOKEN
        masked.append("".join(characters))
    return masked


def optionally_mask_locations(examples: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    if not config.mask_locations:
        return examples
    cache_path = Path(config.masked_text_cache) if config.masked_text_cache else None
    cached = pd.DataFrame(columns=["run_id", "id", "masked_text"])
    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"run_id": str, "id": str})
        required = {"run_id", "id", "masked_text"}
        if not required.issubset(cached.columns) or cached.duplicated(["run_id", "id"]).any():
            raise ValueError("masked-text cache must have unique run_id, id, masked_text columns")
    frame = examples.merge(cached[["run_id", "id", "masked_text"]], on=["run_id", "id"], how="left")
    missing = frame["masked_text"].isna()
    if missing.any():
        frame.loc[missing, "masked_text"] = mask_locations(frame.loc[missing, "text"].tolist(), config.ner_batch_size)
    frame["text"] = frame.pop("masked_text")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame[["run_id", "id", "text"]].rename(columns={"text": "masked_text"}).to_csv(cache_path, index=False)
    return frame


def encode(frame: pd.DataFrame, tokenizer, label_columns: list[str], max_length: int) -> EncodedPosts:
    encodings = tokenizer(
        frame["text"].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    labels = torch.tensor(frame[label_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
    return EncodedPosts(dict(encodings), labels)


def metrics(logits: Tensor, labels: Tensor, threshold: float) -> dict[str, float]:
    predictions = torch.sigmoid(logits).ge(threshold)
    truth = labels.bool()
    true_positive = (predictions & truth).sum().item()
    false_positive = (predictions & ~truth).sum().item()
    false_negative = (~predictions & truth).sum().item()
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_match = predictions.eq(truth).all(dim=1).float().mean().item()
    return {"micro_precision": precision, "micro_recall": recall, "micro_f1": f1, "exact_match": exact_match}


def per_label_metrics(logits: Tensor, labels: Tensor, label_columns: list[str], threshold: float) -> dict[str, dict[str, float]]:
    predictions, truth = torch.sigmoid(logits).ge(threshold), labels.bool()
    results = {}
    for index, label in enumerate(label_columns):
        predicted, actual = predictions[:, index], truth[:, index]
        tp = (predicted & actual).sum().item()
        fp = (predicted & ~actual).sum().item()
        fn = (~predicted & actual).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        results[label] = {"precision": precision, "recall": recall,
                          "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                          "support": int(actual.sum().item())}
    return results


def prediction_tensors(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            outputs = model(**{key: value.to(device) for key, value in batch.items()})
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> dict[str, float]:
    logits, labels = prediction_tensors(model, loader, device)
    return metrics(logits, labels, threshold)


def batches(loader: DataLoader, device: torch.device) -> Iterable[dict[str, Tensor]]:
    for batch in loader:
        yield {key: value.to(device) for key, value in batch.items()}


def main() -> None:
    config = parse_args()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    examples, label_columns = load_examples(config)
    examples = optionally_mask_locations(examples, config)
    train_frame, valid_frame, test_frame = chronological_split(examples, config)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": [LOCATION_TOKEN]})
    datasets = [encode(frame, tokenizer, label_columns, config.max_length) for frame in (train_frame, valid_frame, test_frame)]
    train_set, valid_set, test_set = datasets
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size)
    test_loader = DataLoader(test_set, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_columns), problem_type="multi_label_classification", ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Weight rare positive labels more heavily, while avoiding division by zero.
    positives = train_set.labels.sum(dim=0)
    pos_weight = ((len(train_set) - positives) / positives.clamp(min=1)).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    first = next(iter(train_loader))
    print(json.dumps({
        "device": str(device), "splits": {"train": len(train_set), "valid": len(valid_set), "test": len(test_set)},
        "input_ids_shape": list(first["input_ids"].shape), "attention_mask_shape": list(first["attention_mask"].shape),
        "labels_shape": list(first["labels"].shape), "embedding_shape": list(model.get_input_embeddings().weight.shape),
        "num_labels": len(label_columns),
    }, indent=2))

    best_f1, best_state = -1.0, None
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in batches(train_loader, device):
            labels = batch.pop("labels")
            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = loss_function(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        validation = evaluate(model, valid_loader, device, config.threshold)
        print(json.dumps({"epoch": epoch, "train_loss": running_loss / len(train_loader), **validation}))
        if validation["micro_f1"] > best_f1:
            best_f1 = validation["micro_f1"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    test_logits, test_labels = prediction_tensors(model, test_loader, device)
    test_metrics = metrics(test_logits, test_labels, config.threshold)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_metadata.json").write_text(json.dumps({
        "config": asdict(config), "labels": label_columns, "best_validation_micro_f1": best_f1,
        "test_metrics": test_metrics,
        "per_label_test_metrics": per_label_metrics(test_logits, test_labels, label_columns, config.threshold),
        "split_dates": {name: [str(frame["post_day"].min()), str(frame["post_day"].max())] for name, frame in
                        {"train": train_frame, "valid": valid_frame, "test": test_frame}.items()},
    }, indent=2) + "\n")
    print(json.dumps({"saved_to": str(output_dir), "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
