import numpy as np
from sentence_transformers import SentenceTransformer
from build_dataframe import get_dataframe, MODEL, QUERY_PREFIX
import dataset_fields as DATASET_FIELD
from search_utils import build_results, print_results

df = get_dataframe()

# Keep the full df so doc_ids match BM25 / hybrid (row position in the full df).
# Rows without embeddings stay as zero vectors and are masked out at search time.
has_emb = df[DATASET_FIELD.EMBEDDINGS].notna().to_numpy()

model = SentenceTransformer(MODEL)

valid_embeddings = np.array(df.loc[has_emb, DATASET_FIELD.EMBEDDINGS].tolist(), dtype=np.float32)
valid_embeddings /= np.linalg.norm(valid_embeddings, axis=1, keepdims=True)

corpus_embeddings = np.zeros((len(df), valid_embeddings.shape[1]), dtype=np.float32)
corpus_embeddings[has_emb] = valid_embeddings

def search_dense(query: str, top_k: int = 10) -> list[dict]:
    query_embedding = model.encode(QUERY_PREFIX + query).astype(np.float32)
    query_embedding /= np.linalg.norm(query_embedding)

    scores = corpus_embeddings @ query_embedding
    scores[~has_emb] = -np.inf  # never rank docs without embeddings
    return build_results(df, scores, top_k)

if __name__ == "__main__":
    TEST_QUERY = "american muscle car with more then 300 horse power"

    results = search_dense(TEST_QUERY)

    print_results(results, f"Dense: '{TEST_QUERY}'")
