"""Discover clusters of similar Reddit posts without hand-labelled targets.

Posts are embedded with the base Twitter-RoBERTa encoder, mean-pooled across
non-padding tokens, reduced with PCA, L2-normalized, and clustered with cosine
k-means.  The output is exploratory: a cluster number is not intrinsically a
sentiment.  Use its representative terms and posts to interpret, rename,
merge, or reject clusters after reviewing the results.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
STOPWORDS = {
    "a", "about", "all", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have",
    "i", "if", "in", "is", "it", "its", "just", "me", "my", "not", "of", "on", "or", "so", "that", "the",
    "this", "to", "was", "we", "with", "you", "your",
}


class PostDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        return self.texts[index]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/socialtest.sqlite")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="artifacts/post-clusters")
    parser.add_argument("--clusters", type=int, default=20, help="Number of exploratory clusters to discover.")
    parser.add_argument("--pca-components", type=int, default=20,
                        help="Number of principal components retained before clustering.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100, help="Maximum k-means updates.")
    parser.add_argument("--seed", type=int, default=42)
    values = parser.parse_args()
    if (values.clusters < 2 or values.pca_components < 1 or values.max_length < 4
            or values.batch_size < 1 or values.iterations < 1):
        parser.error("clusters must be at least 2; all other numeric options must be positive")
    return values


def load_posts(database: str, run_id: str) -> pd.DataFrame:
    with sqlite3.connect(database) as connection:
        frame = pd.read_sql_query(
            "SELECT run_id, id, post_day, subreddit, title, selftext FROM posts WHERE run_id = ?",
            connection, params=(run_id,), dtype={"run_id": str, "id": str},
        )
    frame["text"] = (frame["title"].fillna("").astype(str) + " " + frame["selftext"].fillna("").astype(str)).str.strip()
    frame = frame[frame["text"].ne("")].reset_index(drop=True)
    if frame.empty:
        raise ValueError("this run has no non-empty posts")
    return frame


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Convert [batch, sequence, 768] into one [batch, 768] vector per post."""
    weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


def embed(texts: list[str], tokenizer, encoder, batch_size: int, max_length: int, device: torch.device) -> torch.Tensor:
    def collate(batch: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    vectors = []
    encoder.eval()
    with torch.no_grad():
        for batch in DataLoader(PostDataset(texts), batch_size=batch_size, collate_fn=collate):
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = encoder(**batch).last_hidden_state
            vectors.append(mean_pool(hidden, batch["attention_mask"]).cpu())
    return functional.normalize(torch.cat(vectors), dim=1)


def reduce_pca(vectors: torch.Tensor, components: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project embeddings onto their leading principal components.

    Returns normalized projected vectors plus the explained-variance ratios and
    component matrix.  Keeping the transformation makes the clustering space
    inspectable and reusable for later posts.
    """
    if components > min(vectors.shape):
        raise ValueError(
            f"cannot retain {components} PCA components from embeddings with shape {list(vectors.shape)}"
        )
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    # pca_lowrank returns V with one input-feature vector per component.
    _, singular_values, principal_axes = torch.pca_lowrank(centered, q=components, center=False)
    projected = centered @ principal_axes[:, :components]
    total_variance = centered.pow(2).sum() / (len(centered) - 1)
    explained_variance = singular_values[:components].pow(2) / (len(centered) - 1)
    explained_variance_ratio = explained_variance / total_variance
    return functional.normalize(projected, dim=1), explained_variance_ratio, principal_axes[:, :components]


def cosine_kmeans(vectors: torch.Tensor, clusters: int, iterations: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    if clusters > len(vectors):
        raise ValueError(f"cannot create {clusters} clusters from only {len(vectors)} posts")
    generator = torch.Generator().manual_seed(seed)
    centers = vectors[torch.randperm(len(vectors), generator=generator)[:clusters]].clone()
    for update in range(1, iterations + 1):
        similarity = vectors @ centers.T  # [posts, 768] @ [768, clusters] -> [posts, clusters]
        assignments = similarity.argmax(dim=1)
        next_centers = []
        for cluster in range(clusters):
            members = vectors[assignments == cluster]
            # Retain the old center for an empty cluster rather than producing NaNs.
            next_centers.append(members.mean(dim=0) if len(members) else centers[cluster])
        next_centers = functional.normalize(torch.stack(next_centers), dim=1)
        if torch.equal(next_centers, centers):
            return assignments, centers, update
        centers = next_centers
    return assignments, centers, iterations


def top_terms(texts: list[str], limit: int = 12) -> list[str]:
    tokens = (word.lower() for text in texts for word in re.findall(r"[A-Za-z][A-Za-z']{2,}", text))
    counts = Counter(word for word in tokens if word not in STOPWORDS)
    return [word for word, _ in counts.most_common(limit)]


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    posts = load_posts(args.database, args.run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    embeddings = embed(posts["text"].tolist(), tokenizer, encoder, args.batch_size, args.max_length, device)
    vectors, explained_variance_ratio, principal_axes = reduce_pca(embeddings, args.pca_components)
    assignments, centers, updates = cosine_kmeans(vectors, args.clusters, args.iterations, args.seed)
    similarity = (vectors @ centers.T).max(dim=1).values
    posts["cluster"] = assignments.numpy()
    posts["cluster_similarity"] = similarity.numpy()

    summary = []
    for cluster in range(args.clusters):
        members = posts[posts["cluster"] == cluster].sort_values("cluster_similarity", ascending=False)
        summary.append({
            "cluster": cluster,
            "post_count": len(members),
            "mean_similarity": round(float(members["cluster_similarity"].mean()), 4) if len(members) else None,
            "top_terms": top_terms(members["text"].tolist()),
            "representative_posts": members[["id", "subreddit", "title", "cluster_similarity"]].head(5).to_dict("records"),
        })
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    posts.drop(columns="text").to_csv(output / "post_clusters.csv", index=False)
    torch.save({
        "mean": embeddings.mean(dim=0),
        "components": principal_axes,
        "explained_variance_ratio": explained_variance_ratio,
    }, output / "pca_transform.pt")
    (output / "cluster_summary.json").write_text(json.dumps({
        "model": MODEL_NAME, "posts": len(posts), "embedding_shape": list(embeddings.shape),
        "pca_embedding_shape": list(vectors.shape), "pca_components": args.pca_components,
        "pca_explained_variance_ratio": explained_variance_ratio.tolist(),
        "pca_cumulative_explained_variance": float(explained_variance_ratio.sum()),
        "centers_shape": list(centers.shape), "kmeans_updates": updates, "clusters": summary,
    }, indent=2) + "\n")
    print(json.dumps({"saved_to": str(output), "embedding_shape": list(embeddings.shape),
                      "pca_embedding_shape": list(vectors.shape), "centers_shape": list(centers.shape),
                      "pca_cumulative_explained_variance": float(explained_variance_ratio.sum()),
                      "kmeans_updates": updates}, indent=2))


if __name__ == "__main__":
    main()
