"""
Jalon 3 - Validation du retrieval AVANT de brancher le LLM.

Pour chaque question de test, on connait l'article attendu. On verifie qu'il
remonte dans le top-k du retrieval hybride. Si ce n'est pas le cas, le
probleme vient du chunking, de l'embedding ou du corpus -- pas du LLM.

Ces memes questions sont reutilisees comme jeu d'evaluation au Jalon 4
(verifier que la reponse finale cite bien le bon article).

Lancer avec : python -m pytest tests/test_retrieval.py -v
(necessite que la base ait deja ete indexee : python -m src.cli index)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.retrieval import hybrid_search

TEST_CASES = [
    ("Quelle est la duree legale du travail par semaine ?", "L3121-27"),
    ("Combien de jours de conges payes acquiert-on par mois de travail ?", "L3141-3"),
    ("Comment fonctionne la rupture conventionnelle ?", "L1237-11"),
    ("Quelle est la duree du preavis en cas de licenciement ?", "L1234-1"),
    ("Qu'est-ce que le harcelement moral au travail ?", "L1152-1"),
]


def test_expected_article_in_top_k():
    failures = []
    for question, expected_article in TEST_CASES:
        result = hybrid_search([question], use_hyde=False)  # sans HyDE pour un test rapide/deterministe
        retrieved_ids = [c["metadata"].get("article_id") for c in result["chunks"]]
        if expected_article not in retrieved_ids:
            failures.append((question, expected_article, retrieved_ids))

    if failures:
        detail = "\n".join(
            f"  - Q: {q!r} -> attendu {exp}, obtenu {got}" for q, exp, got in failures
        )
        raise AssertionError(f"{len(failures)} question(s) n'ont pas retrouve l'article attendu :\n{detail}")


if __name__ == "__main__":
    test_expected_article_in_top_k()
    print("Toutes les questions de test retrouvent l'article attendu dans le top-k.")
