"""
Decomposition de requete (Jalon 6, amelioration).

Beaucoup de questions utilisateur sont en realite composees de plusieurs
sous-questions independantes ("quels sont mes droits en cas de licenciement
economique, et j'ai droit a combien de mois de preavis ?"). Chercher sur la
question entiere melange les signaux et peut faire remonter des chunks qui
ne repondent qu'a une partie de la question, en ratant l'autre.

On demande donc au LLM de decomposer la question en sous-questions atomiques
independantes. Chaque sous-question est ensuite traitee separement par le
pipeline de recherche (HyDE + hybride), et les contextes recuperes sont
fusionnes avant l'appel de generation final.

Compromis assume (a documenter dans le compte rendu) : cela ajoute un appel
LLM et potentiellement plusieurs recherches supplementaires -> latence accrue.
On decompose donc seulement si la question semble composee (heuristique
simple : longueur + presence de connecteurs), pour eviter ce cout sur les
questions simples.
"""

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.groq_client import chat

DECOMPOSITION_SYSTEM_PROMPT = (
    "Tu decomposes une question sur le droit du travail francais en "
    "sous-questions atomiques et independantes, uniquement si la question "
    "d'origine en contient plusieurs. Si elle est deja simple, renvoie-la "
    "telle quelle. Reponds STRICTEMENT avec une sous-question par ligne, "
    "sans numerotation, sans commentaire, sans introduction."
)

_COMPOUND_HINTS = re.compile(r"\bet\b|\bou\b|\?.+\?|,.*(et|ou|combien|quels)", re.IGNORECASE)


def looks_compound(question: str) -> bool:
    """Heuristique simple pour eviter un appel LLM inutile sur une question
    deja simple (economie de latence/couts, cf. compromis documente)."""
    return len(question) > 60 and bool(_COMPOUND_HINTS.search(question))


def decompose_question(question: str) -> list:
    """Retourne une liste de sous-questions. Retombe sur [question] si la
    question est simple, ou si l'appel LLM echoue."""
    if not looks_compound(question):
        return [question]

    try:
        raw = chat(DECOMPOSITION_SYSTEM_PROMPT, question, temperature=0.0, max_tokens=200)
    except Exception as exc:
        print(f"[Decomposition] Appel LLM indisponible ({exc}), question traitee telle quelle.")
        return [question]

    sub_questions = [line.strip("- ").strip() for line in raw.split("\n") if line.strip()]
    return sub_questions if sub_questions else [question]


if __name__ == "__main__":
    print(decompose_question(
        "Quels sont mes droits en cas de licenciement economique et combien de temps de preavis dois-je avoir ?"
    ))
    print(decompose_question("Combien de jours de conges payes par mois ?"))
