from data_init import get_comments, get_posts
import pandas as pd
#from openai import OpenAI
import torch
import torch.nn as nn
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines.pt_utils import KeyDataset
import tqdm
from typing import Any

LOCATION_ENTITY_TYPES = {"LOCATION", "GPE"}
LOCATION_TOKEN = "[LOCATION]"

#client = OpenAI()

# response = client.embeddings.create(
#     input="Your text string goes here", model="text-embedding-3-small"
# )

roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)

tokenizer.add_special_tokens({'additional_special_tokens': [LOCATION_TOKEN]})

model = AutoModelForSequenceClassification.from_pretrained(
    roberta_model_name,
    num_labels=20,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True
)
model.resize_token_embeddings(len(tokenizer))

ner_pipeline = pipeline(
    "token-classification", 
    model="dbmdz/bert-large-cased-finetuned-conll03-english", 
    aggregation_strategy="simple"
)

def mask_location(text: object, entities: list[dict[str, Any]]) -> str:
    if not isinstance(text, str):
        return ""
    
    location_entities = sorted(
        (
            entity for entity in entities if entity.get("entity_group") in LOCATION_ENTITY_TYPES
        ),
        key=lambda entity: entity["start"],
        reverse=True # sorting location_entities by greater start index first
    )

    characters = list(text)
    for entity in location_entities:
        start, end = entity["start"], entity["end"]
        characters[start:end] = LOCATION_TOKEN

    return "".join(characters)

def batch_mask_locations(texts: list[object], ner_pipeline, batch_size: int = 32) -> list[str]:
    clean_texts = [text if isinstance(text, str) else "" for text in texts]

    ner_results = ner_pipeline(clean_texts, batch_size=batch_size, truncation=True)

    masked_texts = []
    for text, entities in zip(clean_texts, ner_results):
        masked_texts.append(mask_location(text, entities))

    return masked_texts

df_comments = get_comments("test-month-no-askreddit")
df_posts = get_posts("test-month-no-askreddit")
df_posts["full_text"] = (
    df_posts["title"].fillna("").astype(str) + " " + df_posts["selftext"].fillna("").astype(str)
)


df_posts["masked_text"] = batch_mask_locations(df_posts["full_text"].tolist(), ner_pipeline, batch_size=16)








