import os
from datasets import load_dataset
import pandas as pd

BATCH_SIZE = 64

DATASET = "myyycroft/RiDiC_cars_self_consistency_15_generations"
DATASET_TITLE_FIELD = "title_en"
DATASET_COUNTRY_FIELD = "country"
DATASET_DESCRIPTION_FIELD = "gpt-5-chat"

MODEL = "all-MiniLM-L6-v2"

CACHE_FILE_NAME = "embeddings_cache.pkl"

def build_dataframe():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)

    ds = load_dataset(DATASET)
    df = ds["train"].to_pandas()
    df = df[[DATASET_TITLE_FIELD, DATASET_COUNTRY_FIELD, DATASET_DESCRIPTION_FIELD]].copy()

    texts = df[DATASET_DESCRIPTION_FIELD].fillna("").tolist()
    all_embeddings = [None] * len(texts)

    valid_indices = [i for i, t in enumerate(texts) if t.strip() != ""]
    valid_texts = [texts[i] for i in valid_indices]

    embeddings = model.encode(
        valid_texts,
        batch_size = BATCH_SIZE,
        show_progress_bar = True
    )

    for idx, emb in zip(valid_indices, embeddings):
        all_embeddings[idx] = emb.tolist()

    df['embeddings'] = all_embeddings
    return df

def get_dataframe():
    if os.path.exists(CACHE_FILE_NAME):
        df = pd.read_pickle(CACHE_FILE_NAME)
        print(f"Loaded from cache: {CACHE_FILE_NAME}")
    else:
        df = build_dataframe()
        df.to_pickle(CACHE_FILE_NAME)
        print(f"Saved in cache: {CACHE_FILE_NAME}")

    success_count = df["embeddings"].notna().sum()
    print(f"\n Embeddings are ready for {success_count}/{len(df)} records!")
    print(df.head())

    return df


