# Compte rendu (1 page)

*À adapter avec vos propres mots avant la soutenance – ce qui suit reflète les
tests réellement effectués sur le pipeline. Complétez avec les éléments
propres à votre binôme (répartition du travail, temps passé, etc.).*

## Difficultés rencontrées

* **Clé API mal chargée au départ** : le fichier `.env` n'avait pas été créé
  correctement (`Copy-Item .env.example .env` avait échoué silencieusement
  sous PowerShell et créé un fichier au nom littéral `.env.example .env`).
  **Symptôme** : toutes les erreurs de HyDE/décomposition/génération tombaient en
  repli silencieux (« Appel LLM indisponible »), ce qui a masqué le problème un
  moment. **Leçon** : toujours vérifier `Test-Path .env` et le contenu avec
  `Get-Content .env` avant de blâmer le code.

* **Clé API exposée par erreur** : une capture d'écran partagée pendant le
  débogage affichait la clé Groq en clair. Révoquée et régénérée immédiatement
  sur console.groq.com. **Rappel** : même sans commit Git, une clé visible dans
  un partage d'écran ou un message doit être considérée comme compromise.

* **Retrieval incomplet sur les questions composées** : sur « Quels sont mes
  droits en cas de licenciement économique et combien de temps de préavis
  dois-je avoir ? », le premier retrieval ne remontait que les articles de
  préavis (L1234-1) et ratait ceux du licenciement économique (L1233-3),
  pourtant bien présents dans le corpus. **Cause** : la fusion des chunks des
  deux sous-questions (issues de la décomposition) était tronquée au
  top-k GLOBAL, et la sous-question au meilleur score RRF « mangeait » tout le
  budget de chunks avant l'appel au LLM. Corrigé en réservant un quota
  minimum de chunks PAR sous-question avant fusion (voir
  `retrieval.py::hybrid_search`). **Effet de bord assumé** : la réponse finale
  cite maintenant 8 à 9 articles au lieu de 3 à 4 sur une question composée, dont
  certains hors sujet (ex. : les congés payés remontent parfois sur une question de
  licenciement) que le LLM doit lui-même filtrer dans sa réponse – un
  compromis rappel/précision assumé plutôt que résolu.

* **Le LLM peut digresser sur des articles absents du contexte** : sur la
  même question, la réponse commençait par « Je ne trouve pas les
  informations relatives aux articles L3121-27 et L3121-28... », des numéros
  qui n'avaient pourtant aucun rapport avec la question. **Hypothèse** : ces
  numéros apparaissaient en référence croisée dans le texte brut d'un autre
  article du contexte, et le modèle a cru (à tort) que la question les
  concernait. Ce n'est pas une violation de la contrainte de citation (les
  « articles sources » affichés viennent toujours des métadonnées des chunks
  retrouvés, jamais du texte libre du LLM), mais cela montre que le corps de la
  réponse peut évoquer un numéro non fiable sans le citer comme source – une
  nuance à garder en tête si l'on voulait durcir davantage la contrainte.

* **Corpus initial trop réduit (26 articles)** : les premiers tests
  manquaient d'articles sur le licenciement économique (L1233-x) et
  échouaient donc à répondre complètement à certaines questions, non par bug
  mais par absence de données. Étoffé à 40 articles (3 à 8 par thème).


* **Métrique de distance Chroma non spécifiée (bug critique sur le score de
  confiance)** : la calibration (`tests/calibrate_confidence.py`) a révélé
  un « mur » de scores exactement à 0.0000, y compris sur des questions
  clairement dans le corpus (« Qu'est-ce que le SMIC ? »). **Cause** : Chroma
  utilise par défaut la distance euclidienne au carré (L2), pas une
  distance cosinus, sauf si on le précise explicitement à la création de la
  collection (`metadata={"hnsw:space": "cosine"}`). Notre calcul
  `similarity = 1 - dist` supposait une distance cosinus : le désaccord de
  métrique écrasait les scores réels (~0.5 de similarité cosinus, fréquente
  entre phrases juridiques en français, même sans rapport de fond) près de 0.
  Corrigé dans `indexing.py::get_chroma_collection`. **Point d'attention** :
  cette métadonnée n'est appliquée qu'à la création de la collection, donc
  la base persistée existante a dû être entièrement reconstruite (suppression
  de `data/chroma_db/` puis `python -m src.cli index`) : une simple mise à
  jour incrémentale n'aurait pas suffi. Une vérification automatique au
  chargement alerte désormais si ce désaccord se reproduit.

* **Calibration du seuil de confiance : recouvrement réel sur les « pièges »
  juridiques proches** : une fois le bug de métrique corrigé, la
  calibration sur 16 questions dans le corpus et 12 hors corpus (6
  évidentes + 6 « pièges » : droit fiscal, fonction publique, divorce,
  retraite complémentaire, délits routiers) a donné :

  * score minimum observé dans le corpus : **0.4035**
  * score maximum observé sur du hors sujet évident : **0.2800** (bonne
    séparation, marge confortable)
  * score maximum observé sur les « pièges » : **0.4969** (recouvrement réel
    avec le corpus : « retraite complémentaire Agirc-Arrco » et « fonction
    publique territoriale » obtiennent un score plus élevé que la question
    la plus faible du corpus, « congé de fractionnement »)

  Nous en concluons qu'un seuil de similarité seul ne peut pas séparer
  parfaitement le droit du travail des domaines juridiques adjacents sur un
  corpus de cette taille (40 articles) avec ce modèle d'embedding. **Choix
  assumé** : `HARD_REFUSAL_THRESHOLD = 0.30`, placé sous le minimum du corpus
  (jamais de refus à tort d'une vraie question), mais au-dessus du hors sujet
  évident. **Conséquence acceptée** : certains « pièges » passent le seuil dur et
  sont transmis au LLM avec un avertissement de confiance faible plutôt que
  d'être refusés sans appel ; le prompt de génération sert alors de seconde
  ligne de défense (et s'est montré fiable sur nos tests, ex. : « règles de
  circulation routière », correctement refusée par le LLM lui-même).

* **Tests finaux de validation des questions 4 et 5 (réponses conditionnelles
  et frontière du conseil juridique)** :

  * *« À partir de combien de salariés faut-il un CSE ? »* → réponse correcte
    et sourcée (L2311-2, seuil de 11 salariés), sans réserve superflue, ce
    qui est le comportement attendu ici, car ce seuil est une règle d'ordre
    public non modulable par accord d'entreprise, contrairement au préavis
    testé par ailleurs. **Bémol** : la liste « Articles sources » contenait
    trois articles sans rapport (harcèlement sexuel, rupture conventionnelle),
    signe que le bruit de retrieval touche aussi les questions simples, pas
    seulement les questions composées (voir question 1 du README).
  * *« Mon licenciement est-il abusif si mon employeur ne m'a donné aucun
    motif ? »* → comportement globalement conforme à la règle 4 (aucun
    verdict rendu, rappel des obligations procédurales L1232-2/L1232-6,
    mention du recours L1235-3, renvoi vers un professionnel). **Limite
    observée** : le LLM a intégré l'article L2311-2 (seuil du CSE) dans son
    raisonnement de façon disproportionnée, suggérant qu'une absence de
    consultation du CSE « pourrait être considérée comme un manquement » alors
    que ce n'est pertinent que pour les licenciements économiques collectifs,
    pas pour un licenciement individuel sans motif énoncé. Le modèle étaye un
    chunk récupéré mais marginal plutôt que de l'ignorer, illustration
    concrète de la tension rappel/bruit déjà identifiée, cette fois avec un
    risque (atténué par le renvoi final vers un professionnel) de
    surinterprétation plutôt que de simple verbosité.


## Axes d'amélioration retenus (voir aussi le README, section dédiée)

Par ordre de priorité :

1. Filtrage post-génération des numéros d'articles hors contexte (regex sur
   la réponse finale, comparée aux métadonnées des chunks retrouvés) : répond
   directement aux deux limites de digression/surinterprétation observées.
2. Extension du corpus (Option A ou B) pour réduire le bruit et le
   recouvrement avec les domaines juridiques adjacents.
3. Réduction du bruit du quota par sous-question (ne l'élargir que si la
   première passe ne couvre pas déjà tous les thèmes de la question).

## Décisions de conception

* **Chunking hybride** (article + résumé de thème, voir README Q1) plutôt
  que du chunking par section : priorité donnée à la précision de citation.
* **Recherche hybride BM25 + vectorielle avec fusion RRF** plutôt qu'une
  pondération manuelle des scores (échelles non comparables directement).
* **Avertissement juridique assemblé par le code**, pas seulement demandé au
  LLM dans le prompt : garantit sa présence dans 100 % des réponses.
* **Deux seuils de confiance distincts** (avertissement souple vs refus dur
  sans appel au LLM) plutôt qu'un seuil unique, pour fermer la faille du refus
  qui dépendait entièrement du prompt.
* **Quota de chunks par sous-question** plutôt qu'un pool global tronqué,
  suite au bug de retrieval décrit ci-dessus.

## Ce que nous ferions avec plus de temps

* Étendre le corpus via l'API Legifrance (Option A) pour couvrir davantage
  d'articles par thème et réduire le bruit dans les chunks retrouvés sur les
  questions composées.
* Resserrer le prompt de génération pour éviter les digressions sur des
  numéros d'articles mentionnés en référence croisée mais hors sujet.
* Calibrer `HARD_REFUSAL_THRESHOLD` sur un jeu de test plus large que les 5
  questions du jalon 3 (voir `tests/calibrate_confidence.py`), avec des
  questions ambiguës en plus des cas clairement dans/hors corpus.
* Mesurer et documenter la latence réelle du pipeline complet (HyDE +
  décomposition + recherche hybride + génération peut représenter 3 à 4
  appels au LLM par question) afin de décider si un mode « rapide » est
  nécessaire en démonstration.
