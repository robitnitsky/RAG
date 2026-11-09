import numpy as np
import dataset_fields as DATASET_FIELD
import result_fields as RESULT_FIELD


def build_results(df, scores, top_k: int = 10) -> list[dict]:
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            RESULT_FIELD.RANK: rank,
            RESULT_FIELD.DOC_ID: int(idx),
            RESULT_FIELD.SCORE: float(scores[idx]),
            RESULT_FIELD.TITLE: df.iloc[idx][DATASET_FIELD.TITLE],
            RESULT_FIELD.OVERVIEW: df.iloc[idx][DATASET_FIELD.DESCRIPTION][:150],
        }
        for rank, idx in enumerate(top_indices, start=1)
    ]


def print_results(results: list[dict], label: str = ""):
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    for r in results:
        print(f"  #{r[RESULT_FIELD.RANK]} [score={r[RESULT_FIELD.SCORE]:.4f}] {r[RESULT_FIELD.TITLE]}")
        print(f"     {r[RESULT_FIELD.OVERVIEW]}...")
        print()
