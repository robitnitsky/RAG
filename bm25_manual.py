import re
import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict
from build_dataframe import get_dataframe, DATASET_TITLE_FIELD, DATASET_DESCRIPTION_FIELD

df = get_dataframe()

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

inverted_index = defaultdict(set)

for doc_id, text in enumerate(df[DATASET_DESCRIPTION_FIELD].fillna("")):
    for token in tokenize(text):
        inverted_index[token].add(doc_id)

print(f"Unique tokens: {len(inverted_index)}")
print(f"\n The word 'tesla' is met in {len(inverted_index["tesla"])} documents")
print(f"\n The word 'car' is met in {len(inverted_index["car"])} documents")
print(f"\n The word 'battery' is met in {len(inverted_index["battery"])} documents")
print(f"\n The word 'the' is met in {len(inverted_index["the"])} documents")

query = "electric car"
query_tokens = tokenize(query)

matching_docs = None
for token in query_tokens:
    docs = inverted_index.get(token, set())
    matching_docs = docs.copy() if matching_docs is None else matching_docs & docs

print(f"Query: '{query}'")
print(f"Documents which contain ALL query words: {len((matching_docs or set()))}")

if matching_docs:
    for doc_id in sorted(matching_docs)[:5]:
        print(f"    [{doc_id}] {df.iloc[doc_id][DATASET_TITLE_FIELD]}: {df.iloc[doc_id][DATASET_DESCRIPTION_FIELD][:100]}...")

corpus_tokenized = [tokenize(text) for text in df[DATASET_DESCRIPTION_FIELD].fillna("")]
bm25 = BM25Okapi(corpus_tokenized)

print(f"BM25 is built for {len(corpus_tokenized)} documents")

def search_bm25(query: str, top_k: int = 10) -> list[dict]:
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            "rank": len(results) + 1,
            "doc_id": int(idx),
            "score": float(scores[idx]),
            "title": df.iloc[idx][DATASET_TITLE_FIELD],
            "overview": df.iloc[idx][DATASET_DESCRIPTION_FIELD][:150],
        })

    return results

def print_results(results: list[dict], label: str = ""):
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    for r in results:
        print(f"  #{r['rank']} [score={r['score']:.4f}] {r['title']}")
        print(f"     {r['overview']}...")
        print()


EXACT_TEST_QUERY = "Ford Mustang"
results = search_bm25(EXACT_TEST_QUERY)
print_results(results, f"BM25 Exact Query: '{EXACT_TEST_QUERY}'")

SEMANTIC_TEST_QUERY = "american muscle car with more then 300 horse power"
results = search_bm25(SEMANTIC_TEST_QUERY)
print_results(results, f"BM25 Semantic Query: '{SEMANTIC_TEST_QUERY}'")