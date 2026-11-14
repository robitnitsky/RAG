import os
from datasets import load_dataset
import pandas as pd
import dataset_fields as DATASET_FIELD

BATCH_SIZE = 64

DATASET = "myyycroft/RiDiC_cars_self_consistency_15_generations"

MODEL = "BAAI/bge-base-en-v1.5"

# BGE expects this prefix on search queries (not on documents)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CACHE_FILE_NAME = "embeddings_cache.pkl"

def build_dataframe():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)

    ds = load_dataset(DATASET)
    df = ds["train"].to_pandas()
    df = df[[DATASET_FIELD.TITLE, DATASET_FIELD.COUNTRY, DATASET_FIELD.DESCRIPTION]].copy()

    descriptions = df[DATASET_FIELD.DESCRIPTION].fillna("").tolist()
    titles = df[DATASET_FIELD.TITLE].fillna("").tolist()
    countries = df[DATASET_FIELD.COUNTRY].fillna("").tolist()
    all_embeddings = [None] * len(descriptions)

    # Skip records without a description; title/country alone carry too little signal
    valid_indices = [i for i, d in enumerate(descriptions) if d.strip() != ""]
    valid_texts = [
        f"{titles[i]}. Country: {countries[i]}. {descriptions[i]}"
        for i in valid_indices
    ]

    embeddings = model.encode(
        valid_texts,
        batch_size = BATCH_SIZE,
        show_progress_bar = True
    )

    for idx, emb in zip(valid_indices, embeddings):
        all_embeddings[idx] = emb.tolist()

    df[DATASET_FIELD.EMBEDDINGS] = all_embeddings
    return df

def get_dataframe():
    if os.path.exists(CACHE_FILE_NAME):
        df = pd.read_pickle(CACHE_FILE_NAME)
        print(f"Loaded from cache: {CACHE_FILE_NAME}")
    else:
        df = build_dataframe()
        df.to_pickle(CACHE_FILE_NAME)
        print(f"Saved in cache: {CACHE_FILE_NAME}")

    success_count = df[DATASET_FIELD.EMBEDDINGS].notna().sum()
    print(f"\n Embeddings are ready for {success_count}/{len(df)} records!")
    print(df.head())

    return df


