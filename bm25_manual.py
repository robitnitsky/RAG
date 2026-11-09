import re
from rank_bm25 import BM25Okapi
from collections import defaultdict
from build_dataframe import get_dataframe
import dataset_fields as DATASET_FIELD
from search_utils import build_results, print_results

df = get_dataframe()

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

inverted_index = defaultdict(set)

for doc_id, text in enumerate(df[DATASET_FIELD.DESCRIPTION].fillna("")):
    for token in tokenize(text):
        inverted_index[token].add(doc_id)

corpus_tokenized = [tokenize(text) for text in df[DATASET_FIELD.DESCRIPTION].fillna("")]
bm25 = BM25Okapi(corpus_tokenized)

def search_bm25(query: str, top_k: int = 10) -> list[dict]:
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    return build_results(df, scores, top_k)


if __name__ == "__main__":
    print(f"Unique tokens: {len(inverted_index)}")
    print(f"\n The word 'tesla' is met in {len(inverted_index['tesla'])} documents")
    print(f"\n The word 'car' is met in {len(inverted_index['car'])} documents")
    print(f"\n The word 'battery' is met in {len(inverted_index['battery'])} documents")
    print(f"\n The word 'the' is met in {len(inverted_index['the'])} documents")

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
            print(f"    [{doc_id}] {df.iloc[doc_id][DATASET_FIELD.TITLE]}: {df.iloc[doc_id][DATASET_FIELD.DESCRIPTION][:100]}...")

    print(f"BM25 is built for {len(corpus_tokenized)} documents")

    EXACT_TEST_QUERY = "Ford Mustang"
    results = search_bm25(EXACT_TEST_QUERY)
    print_results(results, f"BM25 Exact Query: '{EXACT_TEST_QUERY}'")

    SEMANTIC_TEST_QUERY = "american muscle car with more then 300 horse power"
    results = search_bm25(SEMANTIC_TEST_QUERY)
    print_results(results, f"BM25 Semantic Query: '{SEMANTIC_TEST_QUERY}'")