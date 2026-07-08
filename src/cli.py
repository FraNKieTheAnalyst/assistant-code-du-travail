"""
Jalon 5 - Interface en ligne de commande.

Deux sous-commandes :
    python -m src.cli index    -> construit/persiste la base (jalons 1-2)
    python -m src.cli chat     -> boucle interactive de questions-reponses

La boucle "chat" recharge la base existante SANS reindexer (contrainte du
sujet), et gere un historique court de conversation pour les questions de
suivi ("et pour un CDD ?").
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import CORPUS_PATH
from src.corpus_builder import build_corpus
from src.chunking import build_chunks
from src.indexing import index_chunks, load_index_meta
from src.retrieval import hybrid_search
from src.decomposition import decompose_question
from src.generation import generate_answer
from src.groq_client import chat as groq_chat

MAX_HISTORY_TURNS = 3

REWRITE_SYSTEM_PROMPT = (
    "Tu reformules une question de suivi en une question autonome et complete, "
    "en t'appuyant sur l'historique de conversation fourni. Si la question est "
    "deja autonome, renvoie-la telle quelle. Reponds uniquement avec la question "
    "reformulee, sans commentaire."
)


def cmd_index(_args) -> None:
    print("=== Indexation (jalons 1-2) ===")
    documents = build_corpus()
    chunks = build_chunks(documents)
    index_chunks(chunks)
    print("Indexation terminee. Vous pouvez lancer : python -m src.cli chat")


def rewrite_with_history(question: str, history: list) -> str:
    if not history:
        return question
    history_text = "\n".join(f"Q: {h['question']}\nR (resume): {h['answer_summary']}" for h in history)
    prompt = f"Historique :\n{history_text}\n\nNouvelle question : {question}"
    try:
        return groq_chat(REWRITE_SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=100)
    except Exception:
        return question  # repli silencieux : on traite la question telle quelle


def cmd_chat(_args) -> None:
    meta = load_index_meta()
    if not meta:
        print("Aucune base indexee trouvee. Lancez d'abord : python -m src.cli index")
        return

    print("=== Assistant Code du travail (RAG) ===")
    print(f"Base indexee le {meta.get('last_index_run')} - {meta.get('total_chunks')} chunks - "
          f"modele d'embedding : {meta.get('embedding_model')}")
    print("Tapez votre question, ou 'quitter' pour sortir.\n")

    history = []

    while True:
        try:
            question = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFin de la session.")
            break

        if not question:
            continue
        if question.lower() in {"quitter", "exit", "quit"}:
            print("Fin de la session.")
            break

        standalone_question = rewrite_with_history(question, history)
        sub_questions = decompose_question(standalone_question)

        retrieval_result = hybrid_search(sub_questions)
        answer = generate_answer(standalone_question, retrieval_result)

        print(f"\nAssistant >\n{answer}\n")

        history.append({"question": standalone_question, "answer_summary": answer[:200]})
        history = history[-MAX_HISTORY_TURNS:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistant Code du travail (RAG).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Construit et persiste la base vectorielle + BM25.").set_defaults(func=cmd_index)
    subparsers.add_parser("chat", help="Lance la boucle interactive de questions-reponses.").set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
