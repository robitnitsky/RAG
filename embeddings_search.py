import numpy as np
from sentence_transformers import SentenceTransformer
from build_dataframe import get_dataframe, MODEL
import dataset_fields as DATASET_FIELD
from search_utils import build_results, print_results

df = get_dataframe()
df = df[df[DATASET_FIELD.EMBEDDINGS].notna()].reset_index(drop=True)

model = SentenceTransformer(MODEL)

corpus_embeddings = np.array(df[DATASET_FIELD.EMBEDDINGS].tolist(), dtype=np.float32)
corpus_embeddings /= np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

def search_dense(query: str, top_k: int = 10) -> list[dict]:
    query_embedding = model.encode(query).astype(np.float32)
    query_embedding /= np.linalg.norm(query_embedding)

    scores = corpus_embeddings @ query_embedding
    return build_results(df, scores, top_k)

if __name__ == "__main__":
    TEST_QUERY = "electric car with 450 horse power from 2008 to 2012"

    results = search_dense(TEST_QUERY)

    print_results(results, f"Dense: '{TEST_QUERY}'")
