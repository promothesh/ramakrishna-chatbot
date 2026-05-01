import os
import sys
from rag_chain import build_chain

INDEX_DIR = sys.argv[1] if len(sys.argv) > 1 else "faiss_index"

TESTS = [
    {
        "question": "What is thakur's mother's name?",
        "expected_terms": ["chandra devi", "chandramani", "chandradevi"],
    },
    {
        "question": "Who is thakur's father?",
        "expected_terms": ["khudiram", "kshudiram"],
    },
    {
        "question": "What is thakur's childhood name?",
        "expected_terms": ["gadadhar"],
    },
    {
        "question": "When was thakur born?",
        "expected_terms": ["february 18, 1836", "1836"],
    },
]


def contains_expected(text: str, expected_terms: list[str]) -> bool:
    text = text.lower()
    return any(term.lower() in text for term in expected_terms)


def main():
    print(f"Evaluating index: {INDEX_DIR}")
    chain = build_chain(INDEX_DIR)

    answer_hits = 0
    context_hits = 0

    for i, test in enumerate(TESTS, 1):
        q = test["question"]
        expected = test["expected_terms"]
        result = chain.invoke({"input": q, "chat_history": []})

        answer = result["answer"]
        context_text = "\n\n".join(d.page_content for d in result["context"])

        answer_ok = contains_expected(answer, expected)
        context_ok = contains_expected(context_text, expected)
        answer_hits += int(answer_ok)
        context_hits += int(context_ok)

        print(f"\n[{i}] {q}")
        print(f"    Answer hit : {'YES' if answer_ok else 'NO'}")
        print(f"    Context hit: {'YES' if context_ok else 'NO'}")
        print(f"    Answer     : {answer[:220].replace(chr(10), ' ')}")

    total = len(TESTS)
    print("\nSummary")
    print(f"  Answer accuracy : {answer_hits}/{total}")
    print(f"  Context accuracy: {context_hits}/{total}")


if __name__ == "__main__":
    main()
