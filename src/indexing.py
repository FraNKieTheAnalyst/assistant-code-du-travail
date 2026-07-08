"""
Jalon 2 (partie 2) - Indexation et persistance.

Construit :
  - une base vectorielle Chroma (embeddings sentence-transformers), persistee
    sur disque dans CHROMA_PERSIST_DIR.
  - un index lexical BM25 (rank_bm25), persiste dans BM25_INDEX_PATH.
  - un fichier index_meta.json qui trace le nom du modele d'embedding et les
    hash des articles indexes -> permet de recharger sans reindexer, et sert
    de base aux mises a jour incrementales (update_corpus.py).

Contrainte du sujet : "au redemarrage, le systeme recharge la base sans
reindexer". C'est pour cela que index_chunks() verifie d'abord si un chunk
avec le meme hash existe deja avant de l'embedder.
"""

import json
import pickle
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIR, BM25_INDEX_PATH,
    INDEX_META_PATH, CORPUS_PATH,
)

_embedding_model = None


def get_embedding_model():
    """Charge le modele d'embedding (mis en cache pour eviter les rechargements)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_chroma_collection():
    """Recupere (ou cree) la collection Chroma.

    IMPORTANT : on force explicitement la metrique cosinus via
    "hnsw:space": "cosine". Sans cela, Chroma utilise par defaut la distance
    euclidienne au carre (L2), et non une distance cosinus - alors que tout
    le code de retrieval.py (calcul du score de confiance, similarity = 1 -
    dist) suppose une distance cosinus (comprise entre 0 et 2). Ce
    desaccord de metrique produisait des scores de confiance errones,
    ecrases pres de 0 pour des similarites cosinus reelles moyennes (~0.5) -
    detecte via tests/calibrate_confidence.py (mur de scores a 0.0000 sur
    des questions pourtant dans le corpus).

    ATTENTION : cette metadonnee n'est appliquee qu'A LA CREATION de la
    collection. Si une collection existante a ete creee AVANT ce correctif
    (donc avec la metrique L2 par defaut), il faut supprimer le dossier
    data/chroma_db/ et relancer une indexation complete - un simple
    `get_or_create_collection` sur une collection existante ne change pas
    sa metrique retroactivement. La verification ci-dessous alerte
    explicitement dans ce cas plutot que de laisser le bug se reproduire
    silencieusement.
    """
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name="code_du_travail",
        metadata={"embedding_model": EMBEDDING_MODEL_NAME, "hnsw:space": "cosine"},
    )

    space = (collection.metadata or {}).get("hnsw:space")
    if collection.count() > 0 and space != "cosine":
        print(
            "ATTENTION : la collection Chroma existante n'utilise pas la metrique "
            f"cosinus (metrique actuelle : {space!r}). Les scores de confiance seront "
            "incorrects. Supprimez le dossier data/chroma_db/ et relancez "
            "`python -m src.cli index` pour reconstruire la base avec la bonne metrique."
        )
    return collection


def _existing_hashes(collection) -> dict:
    """Retourne {chunk_id: hash} deja present dans la collection Chroma."""
    try:
        existing = collection.get(include=["metadatas"])
    except Exception:
        return {}
    hashes = {}
    for cid, meta in zip(existing["ids"], existing["metadatas"]):
        hashes[cid] = meta.get("hash")
    return hashes


def index_chunks(chunks: list, force_reembed: bool = False) -> dict:
    """Embedde et persiste les chunks. Ne re-embedde QUE les chunks nouveaux
    ou modifies (hash different), sauf si force_reembed=True.

    Retourne des statistiques (nb ajoutes / inchanges) utiles pour
    update_corpus.py et pour informer l'utilisateur en CLI.
    """
    collection = get_chroma_collection()
    existing_hashes = {} if force_reembed else _existing_hashes(collection)

    to_embed = []
    unchanged = 0
    for chunk in chunks:
        chunk_hash = chunk["metadata"].get("hash")
        if not force_reembed and existing_hashes.get(chunk["chunk_id"]) == chunk_hash and chunk_hash is not None:
            unchanged += 1
            continue
        to_embed.append(chunk)

    if to_embed:
        model = get_embedding_model()
        texts = [c["text"] for c in to_embed]
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

        # Chroma exige des metadonnees scalaires (pas de None) -> nettoyage
        clean_metas = []
        for c in to_embed:
            meta = {k: (v if v is not None else "") for k, v in c["metadata"].items()}
            clean_metas.append(meta)

        collection.upsert(
            ids=[c["chunk_id"] for c in to_embed],
            embeddings=embeddings,
            documents=[c["text"] for c in to_embed],
            metadatas=clean_metas,
        )

    save_index_meta(total_chunks=len(chunks), embedded_now=len(to_embed), unchanged=unchanged)
    build_bm25_index(chunks)

    stats = {"embedded": len(to_embed), "unchanged": unchanged, "total": len(chunks)}
    print(f"Indexation vectorielle : {stats['embedded']} chunks (re)embeddes, "
          f"{stats['unchanged']} inchanges (non re-embeddes), {stats['total']} au total.")
    return stats


def build_bm25_index(chunks: list) -> None:
    """Construit l'index lexical BM25. Contrairement au vectoriel, BM25 n'a
    pas de notion d'upsert incremental fine : on le reconstruit entierement
    a partir de la liste de chunks (operation rapide, pas d'appel reseau/LLM,
    donc pas de cout significatif meme a chaque mise a jour)."""
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunk_ids": [c["chunk_id"] for c in chunks],
            "texts": [c["text"] for c in chunks],
            "metadatas": [c["metadata"] for c in chunks],
        }, f)


def tokenize(text: str) -> list:
    """Tokenisation simple pour BM25 (minuscules, ponctuation retiree).
    Volontairement basique : BM25 est robuste a une tokenisation naive,
    et on evite une dependance NLP supplementaire pour un projet pedagogique."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)
    return text.split()


def save_index_meta(total_chunks: int, embedded_now: int, unchanged: int) -> None:
    meta = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "last_index_run": datetime.now(timezone.utc).isoformat(),
        "total_chunks": total_chunks,
        "embedded_this_run": embedded_now,
        "unchanged_this_run": unchanged,
    }
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_bm25_index() -> dict:
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def load_index_meta() -> dict:
    if not INDEX_META_PATH.exists():
        return {}
    with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    from src.chunking import build_chunks

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = build_chunks(data["documents"])
    index_chunks(chunks)
