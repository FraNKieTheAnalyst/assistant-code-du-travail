# Assistant Code du travail (RAG)

Assistant en ligne de commande qui repond a des questions de droit du travail
francais en citant systematiquement les articles du Code du travail sur
lesquels il s'appuie, sans utiliser LangChain ni LlamaIndex.

## Architecture du pipeline

```
Question utilisateur
   -> Reecriture avec historique (questions de suivi type "et pour un CDD ?")
   -> Decomposition en sous-questions (si la question est composee)
   -> Pour chaque sous-question :
        - HyDE : generation d'une reponse hypothetique (style juridique)
        - Recherche vectorielle (Chroma) sur cette reponse hypothetique
        - Recherche lexicale BM25 sur la question brute
        - Fusion des deux classements (Reciprocal Rank Fusion)
   -> Deduplication + top-k global
   -> Score de confiance (meilleur score cosine)
   -> Generation (Groq, temperature basse) avec citations obligatoires
   -> Assemblage final : avertissement juridique + date du corpus
      ajoutes par le CODE (pas seulement demandes au LLM)
```

## Structure du depot

```
src/
  config.py          # chemins, modeles, seuils - toute la config centralisee
  corpus_builder.py  # Jalon 1 : nettoyage, normalisation, hash par article
  chunking.py        # Jalon 2 : chunking hybride (article + resume de theme)
  indexing.py        # Jalon 2 : embeddings + Chroma + BM25, persistance
  update_corpus.py   # Mise a jour incrementale (freshness, cf Q3)
  hyde.py            # Amelioration : Hypothetical Document Embeddings
  decomposition.py   # Amelioration : decomposition de questions composees
  retrieval.py        # Jalon 3 : recherche hybride BM25 + vectoriel + RRF
  groq_client.py      # Wrapper API Groq partage
  generation.py       # Jalon 4 : prompt, citations, disclaimer garanti par le code
  cli.py              # Jalon 5 : interface interactive + historique
data/
  seed_corpus.json    # Corpus source (Option C, a completer via A/B - voir section corpus)
tests/
  test_retrieval.py   # Jalon 3 : 5 questions de test, article attendu connu
```

## Installation

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

> **Note ponctuelle (correctif de metrique de distance)** : si votre dossier
> `data/chroma_db/` a ete cree avant le correctif de `indexing.py`
> (metrique cosinus explicite), supprimez-le avant de reindexer, sinon les
> scores de confiance resteront errones :
> ```bash
> rm -rf data/chroma_db      # macOS/Linux
> Remove-Item -Recurse -Force data/chroma_db   # Windows PowerShell
> ```

```bash
# 1. Indexation (a faire une seule fois, ou apres modification du corpus)
python -m src.cli index

# 2. Boucle interactive de questions-reponses
python -m src.cli chat

# 3. Mise a jour incrementale du corpus (jalon freshness)
python -m src.update_corpus

# 4. Validation du retrieval (Jalon 3, avant d'utiliser le LLM)
python -m pytest tests/test_retrieval.py -v

# 5. Calibration du seuil de refus strict (Jalon 6) - a refaire apres
#    toute modification importante du corpus
python -m tests.calibrate_confidence
```

## Corpus

Le corpus fourni (`data/seed_corpus.json`) est un corpus **Option C reduit**,
saisi manuellement pour amorcer le pipeline (40 articles, 8 themes couverts,
3 a 8 articles par theme).
**A completer** avant la soutenance via :
- Option A (API Legifrance) : remplacer `corpus_builder.build_corpus` par un
  appel a l'API pour recuperer davantage d'articles par theme.
- Option B (dump LEGI/data.gouv.fr) : ecrire un script `legi_xml_parser.py`
  qui produit un fichier au meme format que `seed_corpus.json`, puis appeler
  `python -m src.update_corpus --source data/mon_corpus_legi.json`.

Le format attendu (voir `seed_corpus.json`) : une liste d'objets avec `id`
(numero d'article), `theme`, `titre`, `texte`, `date_maj`.

## Questions de reflexion

> Redaction finale, basee sur l'ensemble des tests reellement effectues sur
> le pipeline (voir aussi COMPTE_RENDU.md pour le detail chronologique des
> bugs rencontres et corriges).

### 1. Granularite du chunking

Nous retenons une **approche hybride** (`src/chunking.py`) : chunking par
article (un chunk = un article = un numero de citation exact), complete par
un chunk de synthese par theme qui liste les articles couverts.

**Avantages du chunking par article** : precision de citation maximale (le
numero remonte est toujours exact, jamais ambigu sur "quel article dans la
section"), pas de dilution semantique entre articles sans rapport direct
dans un meme chunk, chunks courts donc peu de risque de depasser la fenetre
de contexte.

**Inconvenients observes** : un article isole perd parfois le contexte de sa
section (ex: L1237-14 mentionne "la partie la plus diligente" sans rappeler
que c'est dans le cadre d'une rupture conventionnelle). C'est precisement ce
que le chunk de resume de theme attenue : sur nos tests, il n'a jamais ete
cite comme source (nos questions test matchent directement un article), mais
il agit comme filet de securite pour des questions plus larges ou mal
formulees.

**L'approche hybride s'est revelee necessaire, pas seulement theorique** : le
bug de retrieval rencontre en test (voir COMPTE_RENDU.md) a montre un effet
de bord concret de cette granularite fine sur les questions COMPOSEES.
Question testee : *"Quels sont mes droits en cas de licenciement economique
et combien de temps de preavis dois-je avoir ?"*. Avant correctif, la
sous-question "preavis" (issue de la decomposition) obtenait un meilleur
score RRF et "mangeait" tout le budget de chunks avant l'appel au LLM ; les
articles sur le licenciement economique (L1233-3), pourtant bien presents
dans le corpus, disparaissaient completement de la reponse. Corrige en
reservant un quota minimum de chunks PAR sous-question
(`retrieval.py::hybrid_search`) plutot qu'un pool global tronque. Apres
correctif, la meme question cite bien L1233-3 ET L1234-1.

Consequence assumee du correctif : une question composee envoie desormais
8-9 chunks au LLM au lieu de 5, donc plus de bruit (articles hors-sujet dans
le contexte - on a par exemple vu des articles sur les conges payes
apparaitre dans les sources d'une reponse sur le licenciement). Nous avons
priorise le rappel : rater un article pertinent est pire, dans un assistant
juridique, que fournir un peu de contexte superflu que le LLM sait filtrer
dans sa reponse (confirme en test : le LLM ignore correctement les articles
hors-sujet plutot que de les citer a tort).

### 2. Tracabilite

Le numero d'article est present **a la fois** dans le texte embedde (prefixe
"Article L... (titre)...") et dans les metadonnees (`article_id`). Ce n'est
pas redondant : le texte embedde ameliore le matching semantique (le modele
d'embedding "voit" le numero comme partie du contexte), tandis que les
metadonnees sont la **source de verite** utilisee par le CODE, jamais par le
texte libre du LLM, pour construire la liste finale "Articles sources"
affichee a l'utilisateur.

Le prompt de generation interdit explicitement d'utiliser un numero absent du
contexte fourni (regle 1-2 de `SYSTEM_PROMPT`), et le contexte est numerote
(`[1]`, `[2]`...) plutot que de compter sur le LLM pour retrouver les bons
numeros dans un texte libre.

**Limite reelle observee en test** : sur la question composee licenciement
economique + preavis, le LLM a spontanement commente des numeros d'articles
(L3121-27, L3121-28 - relatifs a la duree du travail, sans rapport avec la
question) absents du sujet, probablement des references croisees presentes
dans le texte brut d'un autre chunk du contexte. Ce n'est pas une violation
de la regle de citation au sens strict - la liste "Articles sources" reste
construite exclusivement a partir des metadonnees des chunks retrouves,
jamais du texte du LLM - mais cela montre que le CORPS de la reponse peut
evoquer un numero non fiable sans le citer comme source officielle. Une
garantie plus stricte consisterait a valider par regex, apres generation, que
tout numero d'article mentionne dans le corps de la reponse fait bien partie
des metadonnees des chunks retrouves, et a le filtrer sinon - amelioration
identifiee mais non implementee, faute de temps.

### 3. Fraicheur

Chaque reponse affiche une mention de date de corpus
(`CORPUS_DATE_NOTICE_TEMPLATE`, basee sur `date_maj` par article), et
`update_corpus.py` permet une reindexation incrementale : seuls les articles
dont le hash a change (texte modifie) sont re-embeddes, les autres sont
ignores - confirme en test a deux reprises (`0 inchanges` puis, apres
extension du corpus de 26 a 40 articles, uniquement les 14 nouveaux articles
re-embeddes, les 26 premiers ignores).

Frequence de mise a jour raisonnable pour ce corpus : le droit du travail
change surtout via des lois de finances (revalorisation du SMIC en janvier),
des ordonnances ponctuelles, et de la jurisprudence de la Cour de cassation
qui fait evoluer l'interpretation sans changer le texte lui-meme. Une
verification mensuelle des articles suivis, plus une verification immediate
apres toute loi ou ordonnance touchant le droit du travail annoncee au
Journal Officiel, serait un bon compromis pour un usage reel.

Pour automatiser avec l'API Legifrance (Option A) : l'API expose la date de
derniere modification de chaque article ; il suffirait d'interroger
periodiquement les articles suivis, de comparer cette date a celle stockee
dans nos metadonnees (`date_maj`), et de ne declencher `update_corpus.py` que
sur les articles dont la date a change - la logique de hash/upsert de notre
pipeline resterait identique, seule la source des donnees changerait.

**Limite complementaire decouverte en calibrant le score de confiance**
(question 3 bis, pertinente aussi pour la fraicheur/fiabilite globale) : un
bug de metrique de distance sur Chroma (metrique L2 par defaut au lieu de
cosinus) faussait completement nos scores de confiance, avec un "mur" de
scores exactement a 0.0000 y compris sur des questions clairement dans le
corpus. Corrige en forcant `hnsw:space: cosine` a la creation de la
collection (`indexing.py::get_chroma_collection`), avec reconstruction
complete de la base necessaire (la metrique n'est fixee qu'a la creation).
Ce bug aurait pu, si nous ne l'avions pas detecte via calibration empirique,
faire refuser a tort des questions parfaitement dans le corpus une fois le
seuil de refus strict active - illustration concrete de l'importance de
valider le retrieval (jalon 3) avant de faire confiance a un score.

### 4. Reponses conditionnelles

Le prompt systeme (regle 3 de `SYSTEM_PROMPT`) demande explicitement une
reponse generale assortie de reserves quand la question depend de la taille
de l'entreprise ou d'une convention collective, plutot qu'une question de
clarification prealable. Choix assume : une reponse immediate avec reserve
("la regle generale est X, mais verifiez votre convention collective ou la
taille de votre entreprise") est plus utile qu'un aller-retour supplementaire
pour un assistant en ligne de commande sans gestion d'etat de session
complexe.

Nous l'avons observe concretement sur la question composee testee
(licenciement economique + preavis) : la reponse generee se terminait par
*"les droits specifiques peuvent varier en fonction de la convention
collective ou de la situation de l'entreprise"* et renvoyait vers un
professionnel - exactement le comportement attendu de la regle 3, obtenu
sans qu'on ait eu besoin de forcer une question de clarification.

**Test confirme en soutenance** sur le cas ideal du corpus (`L2311-2`, seuil
de 11 salaries pour le CSE) :
```
> A partir de combien de salaries faut-il un CSE dans l'entreprise ?

Selon l'article L2311-2, un comite social et economique (CSE) est mis en
place dans les entreprises d'au moins onze salaries, des lors que cet
effectif est atteint pendant douze mois consecutifs.
Articles sources : L1153-5, L1237-12, L2143-3, L2311-2, L2312-8
```
La reponse cite correctement L2311-2, sans reserve superflue cette fois -
et c'est en fait le comportement attendu : le seuil de 11 salaries est une
regle d'ordre public (un plancher legal fixe), pas une regle qui varie selon
la convention collective ou un accord d'entreprise, contrairement au
preavis ou a l'indemnite de rupture conventionnelle testes par ailleurs. Le
systeme ne rajoute donc pas de reserve inutile ici - il n'invente pas une
nuance qui n'existe pas juridiquement pour ce cas precis.

**Point de bruit observe** (a rapprocher de la question 1) : la liste
"Articles sources" contient 3 articles sans rapport direct avec la question
(L1153-5 harcelement sexuel, L1237-12 rupture conventionnelle, L2312-8
attributions du CSE plutot que son seuil de creation) - consequence du
quota de chunks par sous-question qui privilegie le rappel. Le corps de la
reponse, lui, reste focalise et correct : le bruit apparait dans les
sources listees mais n'a pas pollue le texte genere.

### 5. Frontiere du conseil juridique

Le prompt distingue explicitement (regle 4 de `SYSTEM_PROMPT`) les questions
factuelles, auxquelles le systeme repond directement avec citation, des
questions d'APPRECIATION au cas par cas (ex: "mon licenciement est-il
abusif ?"), pour lesquelles il doit rappeler la regle generale et les
criteres retenus par la loi/jurisprudence, sans jamais rendre de verdict
personnel, et orienter vers un professionnel.

Critere de distinction retenu : une question factuelle a une reponse unique
et objective dans le texte de loi (un chiffre, un delai, une definition). Une
question d'appreciation demande d'appliquer la loi a des faits precis et non
fournis (etait-ce reellement un motif economique ? y avait-il un vice du
consentement ?) - ce que seul un juge ou un professionnel peut trancher sur
un dossier complet.

**Confirme en test sur le refus hors-corpus**, qui est le cas limite de
"frontiere" le plus radical : la question *"Quelles sont les regles de
circulation routiere ?"* est systematiquement refusee, et ce refus est
desormais garanti par deux mecanismes complementaires plutot qu'un seul :
1. **Seuil dur cote code** (`HARD_REFUSAL_THRESHOLD`) : calibre
   empiriquement a 0.30 (voir COMPTE_RENDU.md) - la question obtient un
   score de confiance de 0.28, sous le seuil, donc le refus canonique est
   renvoye SANS meme appeler le LLM (confirme en test : la reponse est
   exactement le message code, sans "Articles sources" ni avertissement de
   confiance faible, signe que le chemin LLM n'a pas ete emprunte).
2. **Prompt de generation** (regle 1) comme seconde ligne de defense pour
   les cas qui passent le seuil dur mais restent hors-sujet : confirme sur
   nos "pieges" juridiques adjacents (retraite complementaire, fonction
   publique), dont le score (jusqu'a 0.50) depasse le seuil dur mais qui,
   avant le durcissement du seuil, etaient deja correctement refuses par le
   LLM lui-meme sur la base du contexte fourni.

**Test confirme en soutenance** sur la distinction factuel/appreciation :
```
> Mon licenciement est-il abusif si mon employeur ne m'a donne aucun motif ?

Selon l'article L1232-6 [...] cette lettre comporte l'enonce du ou des
motifs invoques par l'employeur. Si votre employeur ne vous a donne aucun
motif, cela pourrait etre considere comme un manquement aux procedures
legales. [...] L'article L1235-3 precise que si le licenciement d'un
salarie survient pour une cause qui n'est pas reelle et serieuse, le juge
octroie au salarie une indemnite [...] Il est important de noter que seul
un professionnel [...] peut trancher sur un cas precis.
```
Le comportement attendu de la regle 4 est respecte : le systeme ne dit
jamais "oui, c'est abusif" ni "non, ca ne l'est pas". Il rappelle les regles
procedurales pertinentes (obligation de motiver, L1232-6), le recours
existant en cas de cause non reelle et serieuse (L1235-3), et renvoie
explicitement vers un professionnel pour trancher.

**Limite reelle observee** : la reponse a aussi integre l'article L2311-2
(seuil de creation du CSE) dans son raisonnement, en suggerant que l'absence
de consultation du CSE "pourrait egalement etre consideree comme un
manquement" - un raisonnement juridiquement disproportionne ici (la
consultation du CSE est surtout determinante pour les licenciements
economiques collectifs, pas pour un licenciement individuel sans motif
enonce). Le LLM a etaye un article recupere mais peu pertinent plutot que
de l'ignorer silencieusement, illustrant a nouveau la tension rappel/bruit
deja identifiee en question 1 : plus on envoie de chunks pour eviter de
rater un article pertinent, plus le LLM risque de sur-interpreter un chunk
marginal. C'est reste sans consequence grave ici (l'avertissement final et
le renvoi vers un professionnel couvrent le risque), mais c'est un axe
d'amelioration concret (voir section suivante).

## Choix techniques justifies

- **RRF plutot que ponderation de scores** : cosine similarity et score BM25
  ne sont pas sur la meme echelle ; RRF evite d'avoir a les normaliser/comparer
  directement.
- **HyDE** : comble l'ecart de style entre questions familieres et articles
  de loi. Un appel LLM supplementaire par sous-question ; document hypothetique
  jamais montre a l'utilisateur.
- **Decomposition conditionnelle** : declenchee seulement si la question
  semble composee (heuristique + LLM), pour eviter un cout systematique.
- **Avertissement legal garanti par le code** (`assemble_final_answer`) et non
  uniquement par le prompt : repond directement a la contrainte du sujet
  ("un assistant qui l'oublie, meme une fois sur dix, echoue").
- **Quota de chunks par sous-question (pas de pool global tronqué)** : sur une
  question composee, chaque sous-question issue de la decomposition reserve
  un nombre minimum de chunks garanti, au lieu de fusionner tout dans un pool
  unique tronque au top-k global. Sans cela, une sous-question au meilleur
  score RRF "mangeait" tout le budget et faisait disparaitre les chunks
  pertinents pour l'autre sous-question avant meme l'appel au LLM - observe
  concretement en test sur "droits en cas de licenciement economique et
  duree de preavis" (les articles de licenciement economique disparaissaient).

- **Refus sans appel LLM si aucun chunk retrouve OU score de confiance sous
  `HARD_REFUSAL_THRESHOLD`** : la recherche hybride remonte presque toujours
  des chunks, meme hors sujet (BM25 et le cosine ne renvoient jamais un
  ensemble vide en pratique) ; s'appuyer uniquement sur "aucun chunk" ne
  suffisait donc pas a garantir le refus. Le seuil dur, calibre avec
  `tests/calibrate_confidence.py`, ferme cette faille : le refus ne depend
  plus du bon vouloir du prompt sur les questions hors corpus.

## Axes d'amelioration

Priorises par impact probable sur le barème et l'usage reel, sur la base de
tout ce que les tests ont revele.

### Priorite haute

- **Filtrage post-generation des numeros d'articles hors contexte** : valider
  par regex que tout numero d'article mentionne dans le CORPS de la reponse
  fait bien partie des metadonnees des chunks retrouves, et le signaler ou le
  retirer sinon. Repond directement aux deux limites observees en question 2
  et 5 (digression sur L3121-27/28 hors-sujet ; sur-interpretation de L2311-2
  dans une reponse sur un licenciement individuel). Amelioration ciblee, a
  fort impact sur la fiabilite percue, faisable en quelques lignes dans
  `generation.py`.

- **Etendre le corpus via l'API Legifrance (Option A) ou le dump LEGI
  (Option B)** : 40 articles restent un corpus reduit. Un corpus plus large
  reduirait a la fois le bruit dans les sources (moins d'articles marginaux
  matches par defaut) et le recouvrement observe entre "pieges" juridiques
  adjacents et corpus reel lors de la calibration.

- **Reduire le bruit du quota par sous-question** : le correctif qui reserve
  un quota minimum par sous-question (question 1) resout un vrai bug mais
  sur-genereux en chunks (8-9 au lieu de 5). Piste : ne garder le quota
  elargi que si la premiere passe (sans quota) ne couvre pas deja tous les
  themes distincts de la question - complexite supplementaire, mais reduirait
  le bruit sans reintroduire le bug initial.

### Priorite moyenne

- **Recalibrer `HARD_REFUSAL_THRESHOLD` apres tout agrandissement du
  corpus** : le seuil actuel (0.30) est valide pour 40 articles ; il devra
  etre reverifie avec `tests/calibrate_confidence.py` des que le corpus
  change significativement (voir Limites connues).
- **Mode rapide pour la demonstration** : desactiver HyDE et decomposition a
  la demande (`chat --fast`) pour reduire la latence en soutenance, ou toute
  demo devant un public qui n'a pas besoin de voir chaque amelioration a
  l'oeuvre.
- **Etoffer le jeu de test du jalon 3** : 5 questions couvrent difficilement
  tous les cas ; les 16 questions de `tests/calibrate_confidence.py`
  pourraient etre reutilisees comme jeu de regression plus large pour
  `test_retrieval.py`.

### Priorite basse / exploratoire

- **Historique de conversation plus robuste** : la reecriture de question de
  suivi (`cli.py::rewrite_with_history`) repose sur un seul appel LLM sans
  garde-fou explicite ; non teste en profondeur sur des enchainements de
  plus de 2-3 questions.
- **Comparaison d'un modele d'embedding plus gros** : verifier si un modele
  plus lourd que `paraphrase-multilingual-MiniLM-L12-v2` ameliore le
  recouvrement observe sur les "pieges" juridiques adjacents, au prix d'une
  indexation plus lente.

## Limites connues / a completer

- Corpus reduit (Option C) : a etendre via Option A ou B avant la soutenance.
- Le modele d'embedding (`paraphrase-multilingual-MiniLM-L12-v2`) est leger ;
  un modele plus gros pourrait ameliorer le rappel sur un corpus plus large.
- **Seuils de confiance calibres empiriquement** (`tests/calibrate_confidence.py`,
  voir COMPTE_RENDU.md) : `HARD_REFUSAL_THRESHOLD = 0.30` separe fiablement
  le corpus (min observe 0.40) du hors-sujet evident (max observe 0.28),
  mais PAS des domaines juridiques adjacents ("pieges" comme la retraite
  complementaire ou la fonction publique, qui montent jusqu'a 0.50). Cette
  limite est structurelle a un corpus de cette taille et repose sur le
  prompt de generation comme seconde ligne de defense pour ces cas ambigus.
  Recalibrez avec le script apres tout agrandissement significatif du corpus.
