"""
Petit wrapper autour de l'API Groq (chat completions), partage par les
modules hyde.py, decomposition.py et generation.py.

On centralise l'appel ici pour :
  - eviter de dupliquer la gestion d'erreur/clé API dans 3 fichiers
  - pouvoir swap facilement de fournisseur LLM si besoin (un seul point de
    contact avec le reseau).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS

_client = None


def get_client():
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY manquante. Copiez .env.example en .env et renseignez votre cle "
                "(jamais commitee dans Git : verifiez votre .gitignore)."
            )
        from groq import Groq
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def chat(system_prompt: str, user_prompt: str, temperature: float = None, max_tokens: int = None) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=temperature if temperature is not None else GROQ_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else GROQ_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()
