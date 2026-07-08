"""
Configuration centrale de l'assistant Code du travail.
Toutes les valeurs modifiables (chemins, modeles, seuils) sont ici,
pour eviter les "magic numbers" disperses dans le code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Chemins ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SEED_CORPUS_PATH = DATA_DIR / "seed_corpus.json"
CORPUS_PATH = DATA_DIR / "corpus.json"           # corpus "vivant", apres nettoyage/enrichissement
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_db")  # base vectorielle persistee
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"     # index lexical persiste
INDEX_META_PATH = DATA_DIR / "index_meta.json"    # trace le modele d'embedding utilise, dates, hash

# --- Embeddings ---
# Modele multilingue, leger, bon compromis qualite/vitesse pour du francais juridique.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Chunking ---
# Strategie retenue (voir README, Q1) : chunking par article (granularite fine),
# avec un chunk "resume de section" supplementaire par theme pour les questions transverses.
CHUNK_MAX_CHARS = 1200          # au-dela, un article est redecoupe (rare, la plupart des articles sont courts)
CHUNK_OVERLAP_CHARS = 150

# --- Recherche ---
TOP_K_VECTOR = 8
TOP_K_BM25 = 8
TOP_K_FINAL = 5                 # nombre de chunks envoyes au LLM apres fusion RRF
RRF_K = 60                      # constante standard de la Reciprocal Rank Fusion

# Deux seuils distincts sur le meilleur score cosine (score de confiance) :
#   - CONFIDENCE_THRESHOLD : sous ce seuil, on avertit l'utilisateur mais on
#     appelle quand meme le LLM (la question est peut-etre dans le corpus,
#     juste mal formulee).
#   - HARD_REFUSAL_THRESHOLD : sous ce seuil (plus bas), on considere que le
#     corpus n'a AUCUN rapport avec la question -> refus garanti par le CODE,
#     sans meme appeler le LLM. Cela ferme la faille identifiee en test : un
#     retrieval qui remonte toujours des chunks (meme hors-sujet) faisait
#     reposer le refus uniquement sur le prompt.
#
# Valeurs calibrees empiriquement avec tests/calibrate_confidence.py sur le
# corpus de 40 articles (voir COMPTE_RENDU.md pour le detail des mesures) :
#   - score min observe sur des questions reellement dans le corpus : 0.40
#   - score max observe sur du hors-sujet evident : 0.28
#   - score max observe sur des "pieges" juridiques proches (retraite
#     complementaire, fonction publique) : 0.50 -> RECOUVREMENT reel avec le
#     corpus. Un seuil unique ne separe donc pas parfaitement tous les cas :
#     HARD_REFUSAL_THRESHOLD est fixe sous le minimum du corpus (jamais de
#     refus a tort d'une vraie question), au prix de laisser passer certains
#     pieges vers le LLM plutot que vers le refus sans appel. Le prompt de
#     generation sert de seconde ligne de defense sur ces cas ambigus.
CONFIDENCE_THRESHOLD = 0.35
HARD_REFUSAL_THRESHOLD = 0.30

# --- LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = 0.1          # basse temperature : on veut de la precision, pas de la creativite
GROQ_MAX_TOKENS = 800

# --- Disclaimer juridique ---
# Ajoute par le CODE (pas seulement demande dans le prompt) pour garantir sa presence a 100%.
# Voir README, reponse a la question "avertissement juridique - contrainte technique".
LEGAL_DISCLAIMER = (
    "\n\n---\n"
    "*Cet assistant ne fournit pas de conseil juridique. "
    "Consultez un avocat ou l'inspection du travail pour votre situation personnelle.*"
)

CORPUS_DATE_NOTICE_TEMPLATE = (
    "Corpus a jour au {date}. Le droit du travail evolue (lois, ordonnances, jurisprudence) : "
    "verifiez les articles cites sur legifrance.gouv.fr avant toute decision."
)
