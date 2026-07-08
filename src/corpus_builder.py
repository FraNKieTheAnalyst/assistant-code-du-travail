"""
Jalon 1 - Preparation des donnees.

Ce module lit le corpus source (seed_corpus.json, ou un corpus plus complet
si vous branchez l'option A/API Legifrance ou l'option B/dump LEGI), le
nettoie, et produit une liste homogene de documents avec :
  - un identifiant stable (numero d'article)
  - un texte nettoye
  - des metadonnees (theme, titre, source, hash, date de mise a jour)

Le hash sert de base aux mises a jour incrementales (voir update_corpus.py) :
si le hash d'un article n'a pas change depuis la derniere indexation, on
ne le re-embedde pas.
"""

import json
import hashlib
import re
import sys
from pathlib import Path
from datetime import date

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import SEED_CORPUS_PATH, CORPUS_PATH


def clean_text(text: str) -> str:
    """Nettoie les scories courantes des textes juridiques (espaces multiples,
    caracteres d'encodage residuels, espaces avant ponctuation)."""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text


def compute_hash(article_id: str, texte: str) -> str:
    """Hash stable utilise pour detecter si un article a change entre deux
    indexations (voir update_corpus.py, jalon 'mises a jour incrementales')."""
    return hashlib.sha256(f"{article_id}::{texte}".encode("utf-8")).hexdigest()


def build_document(raw_article: dict, source: str) -> dict:
    """Transforme une entree brute du corpus en document normalise.

    Choix retenu (voir README, Q2 - traçabilite) :
    - le texte a EMBEDDER contient : numero d'article + titre + texte de loi,
      pour que la recherche vectorielle beneficie du contexte thematique.
    - le numero d'article est aussi duplique dans les METADONNEES, qui sont
      la source de verite utilisee par le prompt de generation pour la
      citation (on ne fait jamais confiance au texte libre pour extraire
      un numero d'article : on relit toujours les metadonnees du chunk).
    """
    article_id = raw_article["id"]
    titre = clean_text(raw_article["titre"])
    texte = clean_text(raw_article["texte"])
    theme = raw_article["theme"]
    date_maj = raw_article.get("date_maj", str(date.today()))

    texte_embedding = f"Article {article_id} ({titre}). {texte}"

    return {
        "id": article_id,
        "text": texte_embedding,
        "metadata": {
            "article_id": article_id,
            "titre": titre,
            "theme": theme,
            "texte_brut": texte,
            "source": source,
            "date_maj": date_maj,
            "hash": compute_hash(article_id, texte),
        },
    }


def build_corpus(seed_path: Path = SEED_CORPUS_PATH, output_path: Path = CORPUS_PATH) -> list:
    with open(seed_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    source = raw.get("meta", {}).get("source", "inconnue")
    documents = [build_document(a, source) for a in raw["articles"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": raw.get("meta", {}), "documents": documents},
            f, ensure_ascii=False, indent=2,
        )

    print(f"Corpus construit : {len(documents)} documents -> {output_path}")
    return documents


def quality_check(documents: list, n: int = 10) -> None:
    """Controle qualite du sujet (Jalon 1) : affiche n documents au hasard."""
    import random
    sample = random.sample(documents, min(n, len(documents)))
    print(f"\n--- Controle qualite : {len(sample)} documents au hasard ---")
    for d in sample:
        print(f"[{d['metadata']['article_id']}] ({d['metadata']['theme']}) {d['text'][:120]}...")


if __name__ == "__main__":
    docs = build_corpus()
    quality_check(docs)
