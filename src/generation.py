"""
Jalon 4 - Generation avec citations.

Design cle (a expliquer dans le README, Q2 et Q4/Q5) :

  - Le prompt systeme interdit d'inventer un numero d'article et impose de
    ne citer que ceux presents dans le contexte fourni.
  - MAIS on ne fait pas une confiance aveugle au prompt : l'avertissement
    juridique est aussi ajoute par le CODE (assemble_final_answer), pas
    seulement demande au LLM. C'est la reponse a la contrainte du sujet
    ("un assistant qui l'oublie, meme une fois sur dix, echoue") : un LLM
    peut oublier une instruction de temps en temps, du code Python non.
  - Si aucun chunk pertinent n'est retrouve (corpus vide sur le sujet), on
    NE FAIT MEME PAS d'appel LLM : on renvoie un refus canonique. Cela
    supprime tout risque d'hallucination sur les questions hors corpus,
    au prix d'un peu de flexibilite (voir compte rendu, limites connues).
  - Le prompt gere explicitement les cas conditionnels (taille d'entreprise,
    convention collective) en demandant une reponse generale assortie de
    reserves, et la frontiere conseil juridique / information factuelle en
    demandant au modele d'orienter vers un professionnel des qu'une question
    releve d'une appreciation au cas par cas (ex: "mon licenciement est-il
    abusif ?").
"""

import sys
from pathlib import Path
from datetime import date

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import LEGAL_DISCLAIMER, CORPUS_DATE_NOTICE_TEMPLATE, HARD_REFUSAL_THRESHOLD
from src.groq_client import chat

SYSTEM_PROMPT = """Tu es un assistant d'information sur le Code du travail francais.

Regles strictes, non negociables :
1. Tu ne repereponds QU'a partir du contexte numerote fourni ci-dessous. Si le contexte ne
   permet pas de repondre, dis explicitement : "Je ne trouve pas cette information dans ma base."
   N'invente JAMAIS un numero d'article, un chiffre ou une regle qui n'est pas dans le contexte.
2. Chaque affirmation factuelle doit etre rattachee a un numero d'article present dans le
   contexte (ex: "(article L3141-3)"). N'utilise jamais un numero d'article absent du contexte.
3. Si la question depend de la taille de l'entreprise, d'une convention collective, ou de
   circonstances particulieres non precisees, donne la regle generale du Code du travail, cite
   l'article, et indique explicitement que la reponse peut varier selon la convention collective
   ou la situation de l'entreprise.
4. Si la question demande une APPRECIATION ou une INTERPRETATION au cas par cas (ex: "mon
   licenciement est-il abusif ?", "puis-je attaquer mon employeur ?"), ne donne PAS de verdict
   personnel. Rappelle la ou les regles legales generales pertinentes (avec citation), explique
   les criteres que la loi ou la jurisprudence retiennent en general, et indique clairement que
   seul un professionnel (avocat, inspection du travail, conseil de prud'hommes) peut trancher
   sur un cas precis.
5. Ne donne jamais de conseil personnalise du type "vous devriez faire...". Informe, ne conseille pas.
6. Reponds en francais, de maniere claire et concise.
"""


def format_context(chunks: list) -> str:
    """Numerote le contexte pour que le LLM puisse s'y referer sans ambiguite."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        meta = c["metadata"]
        article_id = meta.get("article_id") or "synthese thematique"
        theme = meta.get("theme", "")
        texte = meta.get("texte_brut") or c["text"]
        lines.append(f"[{i}] Article {article_id} - theme: {theme}\n{texte}")
    return "\n\n".join(lines)


def build_user_prompt(question: str, chunks: list) -> str:
    context = format_context(chunks)
    return f"Contexte (extraits du Code du travail) :\n\n{context}\n\nQuestion : {question}"


def generate_answer(question: str, retrieval_result: dict, corpus_date: str = None) -> str:
    """Genere la reponse finale. `retrieval_result` est la sortie de
    hybrid_search() (voir retrieval.py) : {"chunks", "confidence", "low_confidence"}.
    """
    chunks = retrieval_result["chunks"]
    confidence = retrieval_result.get("confidence", 0.0)

    if not chunks or confidence < HARD_REFUSAL_THRESHOLD:
        # Refus garanti par le CODE, sans appel LLM, dans deux cas :
        #   - aucun chunk retrouve du tout
        #   - le meilleur score de similarite est trop bas pour que le corpus
        #     ait un rapport credible avec la question (seuil calibre via
        #     tests/calibrate_confidence.py)
        # Cela evite de dependre uniquement du prompt pour le refus hors corpus.
        return assemble_final_answer(
            "Je ne trouve pas cette information dans ma base de connaissances actuelle. "
            "Il est possible que le sujet ne soit pas couvert par le corpus indexe, ou que "
            "la question sorte du champ du droit du travail francais.",
            low_confidence=False,
            corpus_date=corpus_date,
        )

    user_prompt = build_user_prompt(question, chunks)
    try:
        raw_answer = chat(SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        raw_answer = (
            "Une erreur technique empeche l'appel au modele de langage "
            f"({exc}). Voici cependant les articles les plus pertinents trouves "
            "dans la base : " + ", ".join(
                str(c["metadata"].get("article_id")) for c in chunks if c["metadata"].get("article_id")
            )
        )

    sources = sorted({
        c["metadata"]["article_id"] for c in chunks if c["metadata"].get("article_id")
    })
    if sources:
        raw_answer += "\n\nArticles sources : " + ", ".join(sources)

    return assemble_final_answer(
        raw_answer,
        low_confidence=retrieval_result.get("low_confidence", False),
        corpus_date=corpus_date,
    )


def assemble_final_answer(answer_text: str, low_confidence: bool, corpus_date: str = None) -> str:
    """Assemble la reponse finale. C'est ICI, en code, que l'avertissement
    juridique est garanti a 100% - pas seulement dans le prompt LLM."""
    parts = [answer_text]

    if low_confidence:
        parts.append(
            "\n\n*Score de confiance faible : les articles retrouves correspondent "
            "peut-etre imparfaitement a votre question. Verifiez sur legifrance.gouv.fr.*"
        )

    parts.append(CORPUS_DATE_NOTICE_TEMPLATE.format(date=corpus_date or str(date.today())))
    parts.append(LEGAL_DISCLAIMER)
    return "\n".join(parts)
