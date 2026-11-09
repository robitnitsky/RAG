from collections import defaultdict
import numpy as np
import result_fields as RESULT_FIELD
from search_utils import build_results, print_results
from build_dataframe import get_dataframe
from bm25_manual import search_bm25
from embeddings_search import search_dense

df = get_dataframe()

def reciprocal_rank_fusion(
        results_list: list[list[dict]],
        k: int = 60,
        top_k: int = 10,
) -> list[dict]:
    rrf_scores = defaultdict(float)

    for results in results_list:
        for r in results:
            rrf_scores[r[RESULT_FIELD.DOC_ID]] += 1.0 / (k + r[RESULT_FIELD.RANK])

    scores = np.zeros(len(df))
    for doc_id, score in rrf_scores.items():
        scores[doc_id] = score

    return build_results(df, scores, top_k)

def search_hybrid(query: str, top_k: int = 10) -> list[dict]:
    bm25_results = search_bm25(query, top_k = 50)
    dense_results = search_dense(query, top_k = 50)

    return reciprocal_rank_fusion([bm25_results, dense_results], top_k = top_k)

if __name__ == "__main__":
    TEST_QUERY = "electric car with 450 horse power from 2008 to 2012"

    bm25_res = search_bm25(TEST_QUERY, top_k = 5)
    dense_res = search_dense(TEST_QUERY, top_k = 5)
    hybrid_res = search_hybrid(TEST_QUERY, top_k = 5)

    print_results(bm25_res, f"BM25: '{TEST_QUERY}'")
    print_results(dense_res, f"Dense: '{TEST_QUERY}'")
    print_results(hybrid_res, f"Hybrid (RRF): '{TEST_QUERY}'")