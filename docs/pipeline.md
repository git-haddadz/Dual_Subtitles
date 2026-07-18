# Pipeline Dual Subtitles

Ce document decrit le parcours complet d'une video, depuis sa decouverte
jusqu'a la generation des fichiers SRT et ASS.

## Vue D'ensemble

```text
Video MP4
  -> extraction et normalisation audio
  -> fenetres audio continues coupees sur les silences
  -> transcription Whisper avec timestamps de mots
  -> fusion des chevauchements et reprises ciblees
  -> diarisation et attribution des locuteurs
  -> reconstruction des segments SRT
  -> traduction litterale mot a mot
  -> ecriture SRT et rendu ASS interlineaire
```

Le notebook Colab et la CLI appellent le meme package Python. La logique metier
n'est pas dupliquee dans le notebook.

## 1. Configuration

La classe `ProcessingConfig`, dans
`src/dual_subtitles/core/config.py`, centralise:

- les dossiers d'entree, de sortie et de travail temporaire;
- les langues de transcription, source et cible;
- les modeles Whisper et pyannote;
- la duree cible et maximale des fenetres acoustiques;
- leur chevauchement et les seuils de detection des silences;
- les seuils de segmentation et de fusion des sous-titres;
- les seuils de detection et d'acceptation des reprises ciblees;
- la taille maximale des segments et sous-titres;
- l'activation des sorties SRT, ASS et de la diarisation;
- le peripherique d'execution CPU ou GPU.

La configuration refuse les valeurs negatives, les limites nulles et une
execution dans laquelle les sorties SRT et ASS seraient toutes les deux
desactivees.

## 2. Decouverte, Reprise Et Traitement Par Lot

`process_directory(...)`, dans `src/dual_subtitles/core/pipeline.py`:

1. verifie le dossier d'entree;
2. decouvre les fichiers correspondant a l'extension configuree;
3. trie les videos par nom;
4. determine les sorties encore necessaires;
5. charge les services lourds uniquement au premier besoin;
6. traite les videos l'une apres l'autre.

Une limite optionnelle permet de ne traiter que les premieres videos pendant un
essai. Un callback peut egalement etre appele apres chaque video avec:

- le chemin de la video;
- les sorties disponibles;
- l'erreur eventuelle.

### Reutilisation Des Sorties

Lorsque `skip_existing` est active:

- une sortie demandee deja presente et non vide est conservee;
- si le SRT existe mais que l'ASS manque, le SRT est relu pour regenerer
  uniquement l'ASS;
- Whisper et pyannote ne sont pas charges si aucune retranscription n'est
  necessaire.

Chaque erreur est journalisee sans interrompre les videos suivantes.

## 3. Extraction Et Normalisation Audio

`src/dual_subtitles/io/audio.py` assure:

- l'extraction de la piste audio avec MoviePy et ffmpeg;
- la conversion en WAV;
- la normalisation en mono 16 kHz avec pydub;
- le chargement de l'audio pour le decoupage des chunks.

Les fichiers WAV et chunks temporaires sont supprimes apres chaque video, y
compris lorsqu'une etape echoue.

## 4. Fenetres Acoustiques Et Transcription

Le pipeline ne decoupe plus l'audio selon les tours de parole avant Whisper.
Il cherche d'abord les silences, puis construit des fenetres continues de 28
secondes par defaut, limitees a 30 secondes, avec une seconde de
chevauchement. En l'absence d'un silence convenable, la limite temporelle
cible sert de repli.

`WhisperTranscriber` demande les timestamps de mots et utilise un faisceau de
5 candidats. Chaque timestamp est converti dans le temps global de la video.
Les occurrences issues des zones de chevauchement sont fusionnees seulement
si leur texte et leurs positions temporelles correspondent. Une repetition
reellement prononcee a un autre instant est conservee.

## 5. Controle Et Reprise Ciblee

Apres la premiere transcription, le pipeline recherche les passages suspects:

- confiance disponible inferieure au seuil configure;
- timestamp invalide;
- repetition rapprochee d'au moins trois occurrences;
- mot refuse par un validateur lexical optionnel.

Un passage suspect est retranscrit avec du contexte audio avant et apres. La
nouvelle version remplace l'original uniquement si son score comparatif depasse
le seuil d'amelioration. Sans preuve suffisante, le texte Whisper initial est
conserve. Le validateur lexical constitue un signal de detection et ne corrige
jamais directement un mot.

## 6. Diarisation Apres Transcription

`src/dual_subtitles/services/diarization.py` encapsule pyannote.

Lorsque la diarisation est activee:

1. `PyannoteDiarizer` lit `HUGGINGFACE_TOKEN`;
2. le pipeline `pyannote/speaker-diarization-3.1` est charge paresseusement,
   apres la transcription;
3. les tours de parole sont convertis en objets `Segment`;
4. chaque mot est attribue au locuteur qui le recouvre le plus longtemps;
5. une egalite conserve le locuteur precedent pour eviter les changements
   instables;
6. le pipeline pyannote est deplace sur CUDA lorsqu'un GPU est selectionne.

Sans diarisation, `SingleSpeakerDiarizer` attribue le meme locuteur a tous les
mots.

Le token Hugging Face est lu depuis l'environnement. Il n'est ni ecrit dans les
sous-titres ni conserve par le package.

## 7. Reconstruction Du SRT

`src/dual_subtitles/core/transcript.py` reconstruit les sous-titres depuis les
mots horodates. Une nouvelle unite commence selon les silences, la ponctuation,
le changement de locuteur, la duree maximale ou le nombre maximal de mots. Les
timestamps invalides sont elimines et les sous-titres longs recoivent un retour
a la ligne.

Sur CUDA, Whisper utilise `float16`. Sur CPU, il utilise `float32`.

La progression est journalisee segment par segment avec un pourcentage.

## 8. Traduction Mot A Mot

`src/dual_subtitles/services/translation.py` utilise `deep-translator`.

Chaque mot source est traduit independamment et produit un `WordPair`:

```text
WordPair(source="...", translation="...")
```

Les traductions sont mises en cache par mot afin d'eviter les appels repetes.
En cas d'echec du service, le mot source est conserve comme solution de
secours.

`InterlinearGoogleTranslator.interlinear(...)` reste disponible pour produire
une representation textuelle sur deux lignes. Le rendu ASS utilise directement
les paires structurees.

## 9. Rendu ASS Interlineaire

`src/dual_subtitles/io/subtitle_files.py` genere un script ASS sur une base
virtuelle `1280x720`.

Pour chaque `WordPair`, deux evenements partagent exactement la meme
coordonnee horizontale:

- `ArabicWord`: mot source, taille 48;
- `EnglishGloss`: traduction cible, taille 30.

Ces noms de styles sont internes au format actuel. Les langues restent
configurables dans `ProcessingConfig`.

Le moteur de placement:

1. estime la largeur du mot source et de sa traduction;
2. reserve une colonne selon le texte le plus large;
3. place les paires de droite a gauche dans la zone sure;
4. conserve la traduction centree sous son mot;
5. cree une nouvelle rangee lorsque la largeur disponible est depassee.

Une paire n'est jamais separee entre deux rangees. Les accolades et retours de
ligne sont neutralises avant l'ecriture afin de ne pas injecter de balises ASS
involontaires.

La progression de la traduction ASS est journalisee sous-titre par sous-titre.

## 10. Sorties SRT Et ASS

Le module `src/dual_subtitles/io/subtitle_files.py` fournit:

- `build_srt(...)` et `write_srt(...)`;
- `parse_srt(...)` pour reutiliser un SRT existant;
- `build_ass(...)` et `write_ass(...)`.

Le SRT contient la transcription lisible et ses timestamps. L'ASS contient les
evenements positionnes pour l'affichage interlineaire.

Chaque fichier est annonce dans les logs des qu'il est disponible. Le callback
de fin de video permet au notebook d'afficher `TERMINE` sans attendre la fin
du lot complet.

## 11. Execution Dans Google Colab

`subtitles_gen.ipynb` sert de runner:

1. monte Google Drive;
2. clone ou actualise le depot;
3. verifie et installe l'environnement;
4. demande le token Hugging Face;
5. active et affiche le GPU CUDA;
6. decouvre les videos;
7. lance le traitement avec progression en direct.

Les videos et sorties sont conservees dans Google Drive. Les fichiers
temporaires restent dans `/content`.

La cellule de traitement utilise `VIDEO_LIMIT = None` pour traiter tout le
dossier. La valeur `1` permet un essai rapide sur la premiere video.

## 12. Architecture Du Package

```text
src/dual_subtitles/
|-- cli.py
|-- main.py
|-- core/
|   |-- config.py
|   |-- pipeline.py
|   |-- segmentation.py
|   `-- transcript.py
|-- io/
|   |-- audio.py
|   `-- subtitle_files.py
|-- models/
|   `-- subtitle.py
|-- services/
|   |-- diarization.py
|   |-- transcription.py
|   `-- translation.py
`-- utils/
    `-- timestamps.py
```

- `core`: configuration, orchestration et regles de segmentation;
- `io`: lecture et ecriture des fichiers;
- `models`: objets de donnees et protocoles;
- `services`: integrations Whisper, pyannote et traduction;
- `utils`: fonctions generiques, notamment les timestamps;
- `cli.py`: interface de ligne de commande;
- `main.py`: point d'entree executable.

## 13. CLI Et Verification

Commande locale:

```bash
dual-subtitles process --input-dir ./videos --output-dir ./subtitles
```

Execution explicite du module:

```bash
python -m dual_subtitles.main process \
  --input-dir ./videos \
  --output-dir ./subtitles
```

Controles du depot:

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest
```
