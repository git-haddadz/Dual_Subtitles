# Pipeline Dual Subtitles

Ce document decrit le traitement d'une video, de sa decouverte au rendu ASS
pedagogique. Le notebook Colab et la CLI appellent le meme package Python.

## Vue D'ensemble

```text
Video
  -> audio, diarisation et transcription complete
  -> SRT disponible immediatement
  -> tokenisation non destructive et contextes
  -> morphologie et diacritisation
  -> NER, translitteration et memoire d'entites
  -> glosses Google quasi mot a mot
  -> validation
  -> rendu ASS
```

Les etapes generiques emploient les termes source et cible. La morphologie, la
diacritisation, le NER et la translitteration fournis actuellement sont
specialises pour une source arabe.

## 1. Configuration Et Reprise

`ProcessingConfig` centralise les chemins, langues, modeles, seuils, fenetres
contextuelles et parametres de rendu. Les valeurs par defaut importantes sont:

- glosses Google Translate via `deep-translator`;
- morphologie et NER CAMeL Tools;
- diacritisation CATT;
- deux sous-titres de contexte avant et apres, sans franchir une pause de huit
  secondes et dans une limite de 256 sous-tokens;
- seuil de diacritisation visible de `0.78`;
- source 64 px et glose 28 px sur une base `1280x720`.

Les valeurs invalides sont refusees avant le chargement des modeles.

`process_directory(...)` trie les videos, charge les services lourds au premier
besoin et isole les erreurs par video. `skip_existing` conserve les sorties non
vides. Si seul le SRT existe, toute sa transcription est relue avant de
regenerer l'ASS, sans relancer Whisper ou pyannote.

Le callback `on_video_complete` annonce chaque fichier termine. Le callback
`on_progress(video, stage, current, total)` expose en direct les etapes
`morphology`, `diacritization`, `ner`, `glosses`, `validation` et `ass`.

## 2. Audio, Diarisation Et Transcription

MoviePy et ffmpeg extraient l'audio, puis pydub le normalise en mono 16 kHz.
Pyannote detecte les locuteurs lorsque la diarisation est activee; sinon un
locuteur unique couvre la video.

Les zones de parole sont filtrees, fusionnees et decoupees avant Whisper. Les
chunks conservent leurs timestamps et leur locuteur. Les recouvrements du
padding sont dedupliques, puis les fragments sont regroupes selon les limites
de duree et de mots.

La liste complete des `SubtitleSegment` d'une video est construite avant le
premier traitement linguistique. Le SRT est alors ecrit. Une erreur ulterieure
laisse donc une transcription testable et n'interrompt pas les autres videos.

## 3. Contrat D'annotation

Le remplacement structure de l'ancien `WordPair` repose sur:

- `AnnotatedToken`: surface exacte, offsets, type, morphologie, forme vocalisee,
  entite, translitteration, glose et confiances;
- `AnnotatedSpan`: intervalle lexical affiche comme une seule paire;
- `AnnotatedSubtitle`: segment, tokens et contextes;
- `AnnotatedVideo`: transcription annotee et memoire d'entites.

`WordPair` et `InterlinearTranslator` restent uniquement comme adaptateurs de
compatibilite; le pipeline principal ne les utilise plus.

## 4. Tokenisation Et Contextes

La tokenisation distingue mots et ponctuation en conservant les offsets exacts.
Les espaces, retours de ligne et signes restent dans le texte original entre
ces offsets. La reconstruction doit etre strictement identique a la sortie
Whisper; aucune normalisation ne remplace la surface affichee.

Chaque sous-titre reference une fenetre bornee de voisins. Ces tokens de
contexte alimentent la desambiguisation morphologique et le NER.

## 5. Traduction Lexicale

Les glosses affichees sont traduites mot par mot avec Google Translate via
`deep-translator`, comme dans la pipeline d'origine, puis mises en cache. Ce
service ne demande ni compte ni cle API, mais necessite une connexion et peut
appliquer ses propres limitations. Une erreur est consignee dans les logs et
conserve le mot source. Les sorties vides, repetitives, ponctuationnelles ou
trop longues sont egalement rejetees.

## 6. Morphologie Et Diacritisation

CAMeL Tools fournit le lemme, la racine, la categorie grammaticale, les
clitiques et les traits disponibles. En mode `auto`, les analyses MSA et
Egyptian sont comparees au niveau de la phrase et la mieux notee est retenue;
MSA reste le repli si les donnees dialectales manquent. Ces annotations sont
paralleles a la surface originale.

CATT propose une phrase vocalisee. Pour chaque mot:

1. retirer les diacritiques doit redonner exactement les memes lettres;
2. l'accord avec la forme CAMeL augmente la confiance;
3. les marques internes, shadda et sukun sont conservees;
4. une desinence grammaticale identifiee comme incertaine est retiree;
5. sous le seuil, le mot Whisper entier reste affiche sans diacritiques ajoutes.

Une indisponibilite de modele ajoute un avertissement interne et utilise ce
repli conservateur au lieu de bloquer le fichier.

## 7. Noms Et Memoire D'entites

Le NER CAMeL produit des etiquettes BIO fusionnees en spans `PERSON`,
`LOCATION`, `ORGANIZATION` ou `MISC`. Les variantes sont rapprochees au niveau
de la video avec une cle consonantique, le type et une similarite prudente.

Une forme latine Google compatible avec la translitteration devient la forme canonique.
Les etiquettes NER isolees sans nom propre cible credible sont rejetees; une
entite repetee dans la video peut aussi servir de corroboration. Sinon, le
systeme translittere la forme vocalisee avec une notation lisible (`sh`, `kh`,
`gh`, `q`, `ʿ`, `ā`, etc.). Cette forme est reutilisee dans toute la video. Il
ne s'agit ni d'un glossaire fixe ni d'une memoire utilisateur.

## 8. Glosses Et Validation

Chaque mot recoit directement la traduction Google de sa surface, comme dans
la pipeline d'origine. Une entite validee recoit sa forme latine canonique ou
sa translitteration locale.

Chaque mot lexical produit un span `TOKEN`. Seules une entite NER ou une
expression indivisible explicitement reconnue peut devenir un span multi-token.
La sortie reste donc quasi mot a mot et adaptee a l'apprentissage du vocabulaire.
La ponctuation n'est jamais traduite seule: elle reste rattachee a la surface
source voisine dans le rendu.

Avant le rendu, la validation controle la reconstruction, les chevauchements,
la couverture lexicale, les glosses vides et la coherence des entites. Les
confiances de morphologie, diacritisation, NER et glose restent dans
les objets et les logs; aucun symbole parasite n'est affiche.

## 9. Rendu ASS Pedagogique

Pillow mesure le texte avec les polices Noto fournies. Un repli deterministe est
disponible si le moteur de police est absent. Les TTF sont aussi copies dans le
dossier de sortie `fonts/` pour pouvoir etre installes sur la machine de
lecture. Pour chaque span, deux evenements ASS partagent le meme centre
horizontal:

- `SourceWord`: Noto Naskh Arabic, 64 px;
- `TargetGloss`: Noto Sans, 28 px.

Les paires sont placees de droite a gauche. Leur largeur est le maximum des deux
lignes avec padding. Une nouvelle rangee est creee avant tout debordement. Une
glose longue peut utiliser deux lignes, mais une entite n'est jamais decoupee et
la source n'est jamais reduite. La hauteur de chaque rangee tient compte des
metriques reelles et de l'espace necessaire aux diacritiques.

## 10. Installation Colab Et Locale

Outre les dependances Python, CAMeL requiert ses donnees locales:

```bash
pip install -r requirements.txt
pip install -e .
camel_data -i morphology-db-all
camel_data -i disambig-mle-all
camel_data -i ner-arabert
```

Le notebook verifie les versions, installe ces donnees une fois, redemarre
uniquement apres un changement d'environnement puis affiche chaque etape en
direct. Les modeles Whisper, pyannote, CAMeL et CATT sont telecharges au premier
usage et restent dans leurs caches locaux.

## 11. Verification

Les tests unitaires injectent des doubles et ne telechargent aucun modele. Les
tests qui chargent de vrais modeles doivent porter le marqueur `integration`.

```bash
ruff check src tests
ruff format --check src tests
pytest
```

Les controles couvrent notamment la tokenisation non destructive, les limites
de contexte, la diacritisation prudente, les entites repetees, les replis de
traduction et le centrage des evenements ASS.
