from anthropic import Anthropic
from hybrid_search import search_hybrid
from build_dataframe import get_dataframe
from search_utils import print_results
import dataset_fields as DATASET_FIELD

from config import ANTHROPIC_API_KEY

client = Anthropic(api_key=ANTHROPIC_API_KEY)
df = get_dataframe()

def rag_answer(question: str, top_k: int = 10) -> str:
    results = search_hybrid(question, top_k = top_k)

    context_parts = []
    for r in results:
        doc = df.iloc[r["doc_id"]]
        context_parts.append(
            f"Title: {doc[DATASET_FIELD.TITLE]}\n"
            f"Country {doc[DATASET_FIELD.COUNTRY]}\n"
            f"Description {doc[DATASET_FIELD.DESCRIPTION]}"
        )

    context = "\n\n---\n\n".join(context_parts)

    promt = f"""You are a car intusiast.
    Answer the user's question based ONLY on the provided information below.
    If user asks fully electric or fully gasoline car, you can suggest a hybrid car as last chance, but only if it's presented in the provided information below.
    If the answer cannot be found in the provided context, say so.
    ##Motivation:
    if you follow the instructions and find the most reliable variant, you'll be payed with significant reward.

    ## Retrieved Cars Information:

    {context}

    ##User Question:
    {question}
    
    ##Answer:"""

    message = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": promt,
            }
        ],
    )

    answer = "".join(block.text for block in message.content if block.type == "text")
    return answer, results

TEST_QUESTION = "I need a diesel track for heavy duty on the building sites. It must be durable, powerful and able to tow heavy trailers."

answer, retrieved = rag_answer(TEST_QUESTION, 20)

print("RETRIEVED DOCUMENTS:")
print_results(retrieved)
print("\n" + "=" * 60)
print("LLM ANSWER:")
print("=" * 60)
print(answer)