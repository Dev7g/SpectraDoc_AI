from graph_pipeline import build_graph


def main():
    app = build_graph()

    print("Multi-Modal Multi-Agent RAG (Gemini Flash) with Memory")
    print("Type 'exit' to quit.\n")

    chat_history = []  # list of {"question": str, "answer": str}

    while True:
        q = input("Question: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        if not q:
            continue

        state = {
            "question": q,
            "chat_history": chat_history,
            "text_docs": None,
            "image_docs": None,
            "general_answer": None,
            "critical_text": None,
            "critical_image": None,
            "text_answer": None,
            "image_answer": None,
            "final_answer": None,
        }

        result = app.invoke(state)

        answer = result.get("final_answer", "")
        chat_history = result.get("chat_history", chat_history)

        print("\nFINAL ANSWER:\n", answer)
        print("-" * 60)


if __name__ == "__main__":
    main()
