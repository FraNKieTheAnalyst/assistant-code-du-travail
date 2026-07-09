# Assistant Code du travail (RAG)

Assistant en ligne de commande qui répond à des questions de droit du travail
français en citant systématiquement les articles du Code du travail sur
lesquels il s'appuie, sans utiliser LangChain ni LlamaIndex.

## Architecture du pipeline

```
Question utilisateur
   -> Réécriture avec historique (questions de suivi, type « et pour un CDD ? »)
   -> Décomposition en sous-questions (si la question est composée)
   -> Pour chaque sous-question :
        - HyDE : génération d'une réponse hypothétique (style juridique)
        - Recherche vectorielle (Chroma) sur cette réponse hypothétique
        - Recherche lexicale BM25 sur la question brute
        - Fusion des deux classements (Reciprocal Rank Fusion)
   -> Déduplication + top-k global
   -> Score de confiance (meilleur score cosine)
   -> Génération (Groq, température basse) avec citations obligatoires
   -> Assemblage final : avertissement juridique + date du corpus,
      ajoutés par le CODE (pas seulement demandés au LLM)
```

## Structure du depot

```
```text
src/
  config.py          # chemins, modèles, seuils - toute la configuration centralisée
  corpus_builder.py  # Jalon 1 : nettoyage, normalisation, hash par article
  chunking.py        # Jalon 2 : chunking hybride (article + résumé de thème)
  indexing.py        # Jalon 2 : embeddings + Chroma + BM25, persistance
  update_corpus.py   # Mise à jour incrémentale (freshness, cf. Q3)
  hyde.py            # Amélioration : Hypothetical Document Embeddings
  decomposition.py   # Amélioration : décomposition de questions composées
  retrieval.py       # Jalon 3 : recherche hybride BM25 + vectoriel + RRF
  groq_client.py     # Wrapper API Groq partagé
  generation.py      # Jalon 4 : prompt, citations, disclaimer garanti par le code
  cli.py             # Jalon 5 : interface interactive + historique

data/
  seed_corpus.json   # Corpus source (Option C, à compléter via A/B - voir la section « Corpus »)

tests/
  test_retrieval.py  # Jalon 3 : 5 questions de test, article attendu connu
```

```bash
python -m venv venv

# Windows PowerShell :
.\venv\Scripts\Activate.ps1

# macOS/Linux :
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # puis renseignez GROQ_API_KEY dans .env
```

## Lancement

> **Note ponctuelle (correctif de métrique de distance)** : si votre dossier
> `data/chroma_db/` a été créé avant le correctif de `indexing.py`
> (métrique cosinus explicite), supprimez-le avant de réindexer, sinon les
> scores de confiance resteront erronés :
>
> ```bash
> rm -rf data/chroma_db      # macOS/Linux
> Remove-Item -Recurse -Force data/chroma_db   # Windows PowerShell
> ```

```bash
# 1. Indexation (à faire une seule fois, ou après modification du corpus)
python -m src.cli index

# 2. Boucle interactive de questions-réponses
python -m src.cli chat

# 3. Mise à jour incrémentale du corpus (jalon Freshness)
python -m src.update_corpus

# 4. Validation du retrieval (Jalon 3, avant d'utiliser le LLM)
python -m pytest tests/test_retrieval.py -v

# 5. Calibration du seuil de refus strict (Jalon 6) - à refaire après
#    toute modification importante du corpus
python -m tests.calibrate_confidence
```
## Corpus

Le corpus fourni (`data/seed_corpus.json`) est un corpus **Option C réduit**,
saisi manuellement pour amorcer le pipeline (40 articles, 8 thèmes couverts,
3 à 8 articles par thème).

**À compléter** avant la soutenance via :
- Option A (API Legifrance) : remplacer `corpus_builder.build_corpus` par un
  appel à l'API pour récupérer davantage d'articles par thème.
- Option B (dump LEGI/data.gouv.fr) : écrire un script `legi_xml_parser.py`
  qui produit un fichier au même format que `seed_corpus.json`, puis appeler
  `python -m src.update_corpus --source data/mon_corpus_legi.json`.

Le format attendu (voir `seed_corpus.json`) : une liste d'objets avec `id`
(numéro d'article), `theme`, `titre`, `texte`, `date_maj`.

## Questions de réflexion

> Rédaction finale, basée sur l'ensemble des tests réellement effectués sur
> le pipeline (voir aussi COMPTE_RENDU.md pour le détail chronologique des
> bugs rencontrés et corrigés).

### 1. Granularité du chunking

Nous retenons une **approche hybride** (`src/chunking.py`) : chunking par
article (un chunk = un article = un numéro de citation exact), complétée par
un chunk de synthèse par thème qui liste les articles couverts.

**Avantages du chunking par article** : précision de citation maximale (le
numéro remonté est toujours exact, jamais ambigu sur « quel article dans la
section »), pas de dilution sémantique entre articles sans rapport direct
dans un même chunk, chunks courts donc peu de risque de dépasser la fenêtre
de contexte.

**Inconvénients observés** : un article isolé perd parfois le contexte de sa
section (ex. : L1237-14 mentionne « la partie la plus diligente » sans rappeler
que c'est dans le cadre d'une rupture conventionnelle). C'est précisément ce
que le chunk de résumé de thème atténue : sur nos tests, il n'a jamais été
cité comme source (nos questions test correspondent directement à un article), mais
il agit comme filet de sécurité pour des questions plus larges ou mal
formulées.

**L'approche hybride s'est révélée nécessaire, pas seulement théorique** : le
bug de retrieval rencontré en test (voir COMPTE_RENDU.md) a montré un effet
de bord concret de cette granularité fine sur les questions composées.
Question testée : *« Quels sont mes droits en cas de licenciement économique
et combien de temps de préavis dois-je avoir ? »*. Avant correctif, la
sous-question « préavis » (issue de la décomposition) obtenait un meilleur
score RRF et « mangeait » tout le budget de chunks avant l'appel au LLM ; les
articles sur le licenciement économique (L1233-3), pourtant bien présents
dans le corpus, disparaissaient complètement de la réponse. Corrigé en
réservant un quota minimum de chunks par sous-question
(`retrieval.py::hybrid_search`) plutôt qu'un pool global tronqué. Après
correctif, la même question cite bien L1233-3 et L1234-1.

**Conséquence assumée du correctif** : une question composée envoie désormais
8 à 9 chunks au LLM au lieu de 5, donc plus de bruit (articles hors sujet dans
le contexte ; on a par exemple vu des articles sur les congés payés
apparaître dans les sources d'une réponse sur le licenciement). Nous avons
privilégié le rappel : rater un article pertinent est pire, dans un assistant
juridique, que fournir un peu de contexte superflu que le LLM sait filtrer
dans sa réponse (confirmé en test : le LLM ignore correctement les articles
hors sujet plutôt que de les citer à tort).

### 2. Traçabilité

Le numéro d'article est présent **à la fois** dans le texte embeddé (préfixe
« Article L... (titre)... ») et dans les métadonnées (`article_id`). Ce n'est
pas redondant : le texte embeddé améliore le matching sémantique (le modèle
d'embedding « voit » le numéro comme partie du contexte), tandis que les
métadonnées sont la **source de vérité** utilisée par le CODE, jamais par le
texte libre du LLM, pour construire la liste finale « Articles sources »
affichée à l'utilisateur.

Le prompt de génération interdit explicitement d'utiliser un numéro absent du
contexte fourni (règles 1 et 2 de `SYSTEM_PROMPT`), et le contexte est numéroté
(`[1]`, `[2]`...) plutôt que de compter sur le LLM pour retrouver les bons
numéros dans un texte libre.

**Limite réelle observée en test** : sur la question composée « licenciement
économique + préavis », le LLM a spontanément commenté des numéros d'articles
(L3121-27, L3121-28 — relatifs à la durée du travail, sans rapport avec la
question) absents du sujet, probablement des références croisées présentes
dans le texte brut d'un autre chunk du contexte. Ce n'est pas une violation
de la règle de citation au sens strict : la liste « Articles sources » reste
construite exclusivement à partir des métadonnées des chunks retrouvés,
jamais du texte du LLM. Cela montre cependant que le corps de la réponse peut
évoquer un numéro non fiable sans le citer comme source officielle. Une
garantie plus stricte consisterait à valider par expression régulière, après
génération, que tout numéro d'article mentionné dans le corps de la réponse
fait bien partie des métadonnées des chunks retrouvés, puis à le filtrer dans
le cas contraire — amélioration identifiée mais non implémentée, faute de
temps.

### 3. Fraîcheur

Chaque réponse affiche une mention de la date du corpus
(`CORPUS_DATE_NOTICE_TEMPLATE`, basée sur `date_maj` par article), et
`update_corpus.py` permet une réindexation incrémentale : seuls les articles
dont le hash a changé (texte modifié) sont réembeddés, les autres sont
ignorés — confirmé en test à deux reprises (`0 inchangés` puis, après
extension du corpus de 26 à 40 articles, uniquement les 14 nouveaux articles
réembeddés, les 26 premiers ignorés).

La fréquence de mise à jour raisonnable pour ce corpus est la suivante : le
droit du travail change surtout via des lois de finances (revalorisation du
SMIC en janvier), des ordonnances ponctuelles et de la jurisprudence de la
Cour de cassation, qui fait évoluer l'interprétation sans modifier le texte
lui-même. Une vérification mensuelle des articles suivis, complétée par une
vérification immédiate après toute loi ou ordonnance touchant le droit du
travail annoncée au Journal officiel, constituerait un bon compromis pour un
usage réel.

Pour automatiser avec l'API Legifrance (Option A), l'API expose la date de
dernière modification de chaque article. Il suffirait d'interroger
périodiquement les articles suivis, de comparer cette date à celle stockée
dans nos métadonnées (`date_maj`), puis de ne déclencher `update_corpus.py`
que sur les articles dont la date a changé. La logique de hash/upsert de notre
pipeline resterait identique ; seule la source des données changerait.

**Limite complémentaire découverte lors du calibrage du score de confiance**
(question 3 bis, pertinente aussi pour la fraîcheur et la fiabilité globale) :
un bug de métrique de distance sur Chroma (métrique L2 par défaut au lieu du
cosinus) faussait complètement nos scores de confiance, avec un « mur » de
scores exactement à 0.0000, y compris sur des questions clairement présentes
dans le corpus. Corrigé en forçant `hnsw:space: cosine` à la création de la
collection (`indexing.py::get_chroma_collection`), avec reconstruction
complète de la base nécessaire (la métrique n'étant fixée qu'à la création).

Ce bug aurait pu, si nous ne l'avions pas détecté via une calibration
empirique, faire refuser à tort des questions parfaitement présentes dans le
corpus une fois le seuil de refus strict activé. Cela illustre concrètement
l'importance de valider le retrieval (jalon 3) avant de faire confiance à un
score.

### 4. Réponses conditionnelles

Le prompt système (règle 3 de `SYSTEM_PROMPT`) demande explicitement une
réponse générale assortie de réserves lorsque la question dépend de la taille
de l'entreprise ou d'une convention collective, plutôt qu'une question de
clarification préalable. Choix assumé : une réponse immédiate avec réserve
(« la règle générale est X, mais vérifiez votre convention collective ou la
taille de votre entreprise ») est plus utile qu'un aller-retour supplémentaire
pour un assistant en ligne de commande sans gestion d'état de session
complexe.

Nous l'avons observé concrètement sur la question composée testée
(licenciement économique + préavis) : la réponse générée se terminait par
*« les droits spécifiques peuvent varier en fonction de la convention
collective ou de la situation de l'entreprise »* et renvoyait vers un
professionnel, exactement le comportement attendu de la règle 3, obtenu
sans qu'on ait eu besoin de forcer une question de clarification.

**Test confirmé en soutenance** sur le cas idéal du corpus (`L2311-2`, seuil
de 11 salariés pour le CSE) :

```text
> À partir de combien de salariés faut-il un CSE dans l'entreprise ?

Selon l'article L2311-2, un comité social et économique (CSE) est mis en
place dans les entreprises d'au moins onze salariés, dès lors que cet
effectif est atteint pendant douze mois consécutifs.

Articles sources : L1153-5, L1237-12, L2143-3, L2311-2, L2312-8
```

La réponse cite correctement L2311-2, sans réserve superflue cette fois.
C'est en fait le comportement attendu : le seuil de 11 salariés est une
règle d'ordre public (un plancher légal fixe), pas une règle qui varie selon
la convention collective ou un accord d'entreprise, contrairement au
préavis ou à l'indemnité de rupture conventionnelle testés par ailleurs. Le
système ne rajoute donc pas de réserve inutile ici : il n'invente pas une
nuance qui n'existe pas juridiquement pour ce cas précis.

**Point de bruit observé** (à rapprocher de la question 1) : la liste
« Articles sources » contient trois articles sans rapport direct avec la
question (L1153-5 : harcèlement sexuel, L1237-12 : rupture conventionnelle,
L2312-8 : attributions du CSE plutôt que son seuil de création), conséquence
du quota de chunks par sous-question qui privilégie le rappel. Le corps de la
réponse, lui, reste focalisé et correct : le bruit apparaît dans les
sources listées, mais n'a pas pollué le texte généré.

### 5. Frontière du conseil juridique

Le prompt distingue explicitement (règle 4 de `SYSTEM_PROMPT`) les questions
factuelles, auxquelles le système répond directement avec citation, des
questions d'appréciation au cas par cas (ex. : « Mon licenciement est-il
abusif ? »), pour lesquelles il doit rappeler la règle générale et les
critères retenus par la loi et la jurisprudence, sans jamais rendre de
verdict personnel, puis orienter vers un professionnel.

Le critère de distinction retenu est le suivant : une question factuelle
possède une réponse unique et objective dans le texte de loi (un chiffre, un
délai, une définition). Une question d'appréciation demande d'appliquer la
loi à des faits précis et non fournis (était-ce réellement un motif
économique ? Y avait-il un vice du consentement ?), ce que seul un juge ou un
professionnel peut trancher sur un dossier complet.

**Confirmé en test sur le refus hors corpus**, qui constitue le cas limite de
« frontière » le plus radical : la question *« Quelles sont les règles de
circulation routière ? »* est systématiquement refusée, et ce refus est
désormais garanti par deux mécanismes complémentaires plutôt qu'un seul :

1. **Seuil dur côté code** (`HARD_REFUSAL_THRESHOLD`) : calibré
   empiriquement à 0,30 (voir COMPTE_RENDU.md). La question obtient un
   score de confiance de 0,28, sous le seuil ; le refus canonique est donc
   renvoyé **sans même appeler le LLM** (confirmé en test : la réponse est
   exactement le message codé, sans « Articles sources » ni avertissement de
   confiance faible, signe que le chemin LLM n'a pas été emprunté).

2. **Prompt de génération** (règle 1) comme seconde ligne de défense pour
   les cas qui passent le seuil dur mais restent hors sujet : confirmé sur
   nos « pièges » juridiques adjacents (retraite complémentaire, fonction
   publique), dont le score (jusqu'à 0,50) dépasse le seuil dur mais qui,
   avant le durcissement du seuil, étaient déjà correctement refusés par le
   LLM lui-même sur la base du contexte fourni.

**Test confirmé en soutenance** sur la distinction factuel / appréciation :

```text
> Mon licenciement est-il abusif si mon employeur ne m'a donné aucun motif ?

Selon l'article L1232-6 [...] cette lettre comporte l'énoncé du ou des
motifs invoqués par l'employeur. Si votre employeur ne vous a donné aucun
motif, cela pourrait être considéré comme un manquement aux procédures
légales. [...] L'article L1235-3 précise que si le licenciement d'un
salarié survient pour une cause qui n'est pas réelle et sérieuse, le juge
octroie au salarié une indemnité [...] Il est important de noter que seul
un professionnel [...] peut trancher sur un cas précis.
```

Le comportement attendu de la règle 4 est respecté : le système ne dit
jamais « oui, c'est abusif » ni « non, ça ne l'est pas ». Il rappelle les
règles procédurales pertinentes (obligation de motiver, L1232-6), le recours
existant en cas de cause non réelle et sérieuse (L1235-3), puis renvoie
explicitement vers un professionnel pour trancher.

**Limite réelle observée** : la réponse a aussi intégré l'article L2311-2
(seuil de création du CSE) dans son raisonnement, en suggérant que l'absence
de consultation du CSE « pourrait également être considérée comme un
manquement ». Il s'agit d'un raisonnement juridiquement disproportionné ici
(la consultation du CSE est surtout déterminante pour les licenciements
économiques collectifs, pas pour un licenciement individuel sans motif
énoncé). Le LLM a étayé un article récupéré mais peu pertinent plutôt que
de l'ignorer silencieusement, illustrant à nouveau la tension rappel / bruit
déjà identifiée en question 1 : plus on envoie de chunks pour éviter de
rater un article pertinent, plus le LLM risque de surinterpréter un chunk
marginal. Cela reste sans conséquence grave ici (l'avertissement final et le
renvoi vers un professionnel couvrent le risque), mais constitue un axe
d'amélioration concret (voir section suivante).

## Choix techniques justifiés

- **RRF plutôt que pondération des scores** : la similarité cosinus et le
  score BM25 ne sont pas sur la même échelle ; RRF évite d'avoir à les
  normaliser ou les comparer directement.
- **HyDE** : comble l'écart de style entre les questions familières et les
  articles de loi. Un appel LLM supplémentaire par sous-question ; le document
  hypothétique n'est jamais montré à l'utilisateur.
- **Décomposition conditionnelle** : déclenchée seulement si la question
  semble composée (heuristique + LLM), afin d'éviter un coût systématique.
- **Avertissement légal garanti par le code** (`assemble_final_answer`) et non
  uniquement par le prompt : répond directement à la contrainte du sujet
  (« un assistant qui l'oublie, même une fois sur dix, échoue »).
- **Quota de chunks par sous-question (pas de pool global tronqué)** : sur une
  question composée, chaque sous-question issue de la décomposition réserve
  un nombre minimum de chunks garanti, au lieu de fusionner l'ensemble dans un
  pool unique tronqué au top-k global. Sans cela, une sous-question au meilleur
  score RRF « mangeait » tout le budget et faisait disparaître les chunks
  pertinents pour l'autre sous-question avant même l'appel au LLM, comme
  observé en test sur « droits en cas de licenciement économique et durée de
  préavis ».

- **Refus sans appel au LLM si aucun chunk n'est retrouvé ou si le score de
  confiance est inférieur à `HARD_REFUSAL_THRESHOLD`** : la recherche hybride
  remonte presque toujours des chunks, même hors sujet (BM25 et la similarité
  cosinus ne renvoient pratiquement jamais un ensemble vide). S'appuyer
  uniquement sur « aucun chunk » ne suffisait donc pas à garantir le refus.
  Le seuil dur, calibré avec `tests/calibrate_confidence.py`, ferme cette
  faille : le refus ne dépend plus du bon vouloir du prompt sur les questions
  hors corpus.

## Axes d'amélioration

Priorisés par impact probable sur le barème et l'usage réel, sur la base de
tout ce que les tests ont révélé.

### Priorité haute

- **Filtrage post-génération des numéros d'articles hors contexte** : valider
  par expression régulière que tout numéro d'article mentionné dans le corps
  de la réponse fait bien partie des métadonnées des chunks retrouvés, puis le
  signaler ou le retirer dans le cas contraire. Répond directement aux deux
  limites observées en questions 2 et 5 (digression sur L3121-27/28 hors
  sujet ; surinterprétation de L2311-2 dans une réponse sur un licenciement
  individuel). Amélioration ciblée, à fort impact sur la fiabilité perçue,
  faisable en quelques lignes dans `generation.py`.

- **Étendre le corpus via l'API Legifrance (Option A) ou le dump LEGI
  (Option B)** : 40 articles restent un corpus réduit. Un corpus plus large
  réduirait à la fois le bruit dans les sources et le recouvrement observé
  entre les « pièges » juridiques adjacents et le corpus réel.

- **Réduire le bruit du quota par sous-question** : le correctif qui réserve
  un quota minimum par sous-question résout un vrai bug mais reste
  surgénéreux en chunks (8 à 9 au lieu de 5). Une piste serait de ne conserver
  le quota élargi que si la première passe (sans quota) ne couvre pas déjà
  tous les thèmes distincts de la question.

### Priorité moyenne

- **Recalibrer `HARD_REFUSAL_THRESHOLD` après tout agrandissement du corpus**.
- **Mode rapide pour la démonstration** (`chat --fast`).
- **Étoffer le jeu de tests du jalon 3**.

### Priorité basse / exploratoire

- **Historique de conversation plus robuste**.
- **Comparer un modèle d'embedding plus volumineux**.

## Limites connues / à compléter

- Corpus réduit (Option C) : à étendre via l'Option A ou B avant la soutenance.
- Le modèle d'embedding (`paraphrase-multilingual-MiniLM-L12-v2`) est léger ;
  un modèle plus volumineux pourrait améliorer le rappel sur un corpus plus
  large.
- **Seuils de confiance calibrés empiriquement**
  (`tests/calibrate_confidence.py`, voir COMPTE_RENDU.md) : `HARD_REFUSAL_THRESHOLD = 0.30`
  sépare fiablement le corpus (minimum observé : 0,40) du hors sujet évident
  (maximum observé : 0,28), mais pas des domaines juridiques adjacents
  (« pièges » comme la retraite complémentaire ou la fonction publique).
  Cette limite est structurelle à un corpus de cette taille et repose sur le
  prompt de génération comme seconde ligne de défense pour ces cas ambigus.
  Recalibrer le seuil après tout agrandissement significatif du corpus.
