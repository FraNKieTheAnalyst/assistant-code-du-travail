"""
Jalon 3 - Recherche, et amelioration Jalon 6 - recherche hybride.

Combine :
  - recherche vectorielle (Chroma), avec la requete transformee par HyDE
  - recherche lexicale BM25, sur la question brute (utile pour "que dit
    L3121-1 ?", ou le numero d'article est un signal exact que le vectoriel
    peut manquer)
  - fusion des deux classements par Reciprocal Rank Fusion (RRF), qui evite
    d'avoir a normaliser/comparer des scores de nature differente (cosine
    similarity vs score BM25).

Le score de confiance (Jalon 6) est calcule a partir du meilleur score
cosine du canal vectoriel : s'il est sous CONFIDENCE_THRESHOLD, l'utilisateur
est prevenu que la reponse peut etre peu fiable.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import TOP_K_VECTOR, TOP_K_BM25, TOP_K_FINAL, RRF_K, CONFIDENCE_THRESHOLD
from src.indexing import get_embedding_model, get_chroma_collection, load_bm25_index, tokenize
from src.hyde import generate_hypothetical_document


def vector_search(query_text: str, top_k: int = TOP_K_VECTOR) -> list:
    """Retourne une liste ordonnee de (chunk_id, score_cosine, document, metadata)."""
    model = get_embedding_model()
    collection = get_chroma_collection()
    query_embedding = model.encode([query_text], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )
    if not results["ids"] or not results["ids"][0]:
        return []

    out = []
    for cid, doc, meta, dist in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        similarity = 1 - dist  # Chroma renvoie une distance cosine (0 = identique)
        out.append((cid, similarity, doc, meta))
    return out


def bm25_search(query_text: str, top_k: int = TOP_K_BM25) -> list:
    """Retourne une liste ordonnee de (chunk_id, score_bm25, document, metadata)."""
    index = load_bm25_index()
    tokenized_query = tokenize(query_text)
    scores = index["bm25"].get_scores(tokenized_query)

    ranked = sorted(
        zip(index["chunk_ids"], scores, index["texts"], index["metadatas"]),
        key=lambda x: x[1], reverse=True,
    )
    return ranked[:top_k]


def rrf_fuse(rankings: list, k: int = RRF_K) -> dict:
    """Reciprocal Rank Fusion.
    `rankings` est une liste de classements, chacun une liste ordonnee de
    chunk_id (du plus pertinent au moins pertinent). Retourne un dict
    {chunk_id: score_fusionne}.
    """
    fused_scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return fused_scores


def hybrid_search_single(query_text: str, use_hyde: bool = True) -> list:
    """Recherche hybride pour UNE requete (ou sous-question). Retourne les
    chunks fusionnes avec leurs metadonnees, tries par score RRF, et le
    meilleur score cosine (pour le score de confiance)."""
    vector_query = generate_hypothetical_document(query_text) if use_hyde else query_text

    vec_results = vector_search(vector_query)
    bm25_results = bm25_search(query_text)

    vec_ranking = [r[0] for r in vec_results]
    bm25_ranking = [r[0] for r in bm25_results]
    fused_scores = rrf_fuse([vec_ranking, bm25_ranking])

    # table de correspondance chunk_id -> (document, metadata), peu importe
    # la source (vectorielle ou lexicale)
    lookup = {}
    for cid, _, doc, meta in vec_results:
        lookup[cid] = (doc, meta)
    for cid, _, doc, meta in bm25_results:
        lookup.setdefault(cid, (doc, meta))

    best_vector_score = max((s for _, s, _, _ in vec_results), default=0.0)

    ranked_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for cid, score in ranked_chunks:
        if cid not in lookup:
            continue
        doc, meta = lookup[cid]
        results.append({"chunk_id": cid, "text": doc, "metadata": meta, "rrf_score": score})

    return results, best_vector_score


def hybrid_search(sub_questions: list, use_hyde: bool = True, top_k_final: int = TOP_K_FINAL) -> dict:
    """Recherche hybride sur une ou plusieurs sous-questions (post-decomposition).

    Correction importante : sur une question composee (plusieurs sous-questions),
    on NE fusionne PAS tout dans un seul pool tronque au top_k_final global.
    Si on faisait ca, une sous-question dont les chunks ont un meilleur score
    RRF "mangerait" tout le budget, et les chunks pertinents pour l'AUTRE
    sous-question seraient elimines avant meme d'atteindre le LLM - ce qui a
    ete observe en test (question sur licenciement economique + preavis :
    les articles sur le licenciement economique disparaissaient).

    On reserve donc un quota minimum de chunks PAR sous-question, puis on
    fusionne en dedupliquant. Le total peut depasser top_k_final si plusieurs
    sous-questions sont posees : c'est voulu, une question composee a besoin
    de plus de contexte qu'une question simple.

    Retourne {"chunks": [...], "confidence": float, "low_confidence": bool}.
    """
    n = max(len(sub_questions), 1)
    # Quota par sous-question : au moins 3 chunks chacune, jamais moins que
    # top_k_final au total ne l'exigerait pour une seule question.
    per_question_k = max(3, top_k_final // n) if n > 1 else top_k_final

    all_chunks = {}
    best_score_overall = 0.0

    for sq in sub_questions:
        results, best_vector_score = hybrid_search_single(sq, use_hyde=use_hyde)
        best_score_overall = max(best_score_overall, best_vector_score)
        for r in results[:per_question_k]:
            existing = all_chunks.get(r["chunk_id"])
            if existing is None or r["rrf_score"] > existing["rrf_score"]:
                all_chunks[r["chunk_id"]] = r

    total_cap = top_k_final if n == 1 else top_k_final + (n - 1) * per_question_k
    final_chunks = sorted(all_chunks.values(), key=lambda x: x["rrf_score"], reverse=True)[:total_cap]

    return {
        "chunks": final_chunks,
        "confidence": best_score_overall,
        "low_confidence": best_score_overall < CONFIDENCE_THRESHOLD,
    }


if __name__ == "__main__":
    result = hybrid_search(["Combien de jours de conges payes par mois de travail ?"])
    for c in result["chunks"]:
        print(f"{c['rrf_score']:.4f} | {c['metadata']['article_id']} | {c['text'][:100]}")
    print("Confiance (meilleur score cosine):", result["confidence"])
