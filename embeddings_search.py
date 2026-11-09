import numpy as np
from sentence_transformers import SentenceTransformer
from build_dataframe import get_dataframe, MODEL
import dataset_fields as DATASET_FIELD
from search_utils import print_results

df = get_dataframe()
df = df[df[DATASET_FIELD.EMBEDDINGS].notna()].reset_index(drop=True)

model = SentenceTransformer(MODEL)

corpus_embeddings = np.array(df[DATASET_FIELD.EMBEDDINGS].tolist(), dtype=np.float32)
corpus_embeddings /= np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

def search(query: str, top_k: int = 10) -> list[dict]:
    query_embedding = model.encode(query).astype(np.float32)
    query_embedding /= np.linalg.norm(query_embedding)

    scores = corpus_embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "rank": rank,
            "doc_id": int(idx),
            "score": float(scores[idx]),
            "title": df.iloc[idx][DATASET_FIELD.TITLE],
            "overview": df.iloc[idx][DATASET_FIELD.DESCRIPTION][:150],
        }
        for rank, idx in enumerate(top_indices, start=1)
    ]

TEST_QUERY = "electric car with 450 horse power from 2008 to 2012"

results = search(TEST_QUERY)

print_results(results, f"Dense: '{TEST_QUERY}'")
