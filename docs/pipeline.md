# Pipeline Dual Subtitles

Ce document decrit le pipeline experimental specialise pour une source arabe.
La langue cible des glosses reste configurable.

## Vue D'ensemble

```text
Video MP4
  -> extraction et normalisation audio
  -> diarisation pyannote
  -> profil vocal prudent par locuteur
  -> decoupage de chaque locuteur sur les pauses acoustiques
  -> transcription locale Cohere Arabic par phrase/locuteur
  -> validation et seconde passe des sorties suspectes
  -> segmentation de lisibilite sans franchir un locuteur
  -> ecriture SRT
  -> ecriture des metadonnees locuteur JSON
  -> traduction Google mot a mot
  -> rendu ASS interlineaire
```

Whisper, les fenetres chevauchantes de 30 secondes et la fusion de mots ont ete
retires de cette branche. Un seul modele determine le texte transcrit, y compris
lors d'une seconde passe ciblee:
`CohereLabs/cohere-transcribe-arabic-07-2026`.

## 1. Configuration

`ProcessingConfig` centralise:

- les dossiers d'entree, de sortie et de travail temporaire;
- les langues de transcription, source et cible;
- les modeles Cohere et pyannote;
- les durees minimale et maximale d'une unite de parole;
- le seuil de pause utilise pour approcher les limites de phrase;
- le nombre maximal de mots affiche dans un sous-titre;
- l'activation, la duree d'analyse et le seuil des profils vocaux;
- les sorties SRT et ASS, la diarisation et le peripherique d'execution.

Le modele Cohere de cette branche est specialise pour l'arabe et l'anglais. La
configuration par defaut utilise l'arabe comme langue de transcription.

## 2. Traitement Par Lot Et Reprise

`process_directory(...)` trie les videos, traite chaque erreur sans interrompre
le lot et appelle le callback public apres chaque video. Les services lourds
sont crees seulement lorsqu'une transcription ou un ASS manque.

Lorsque `skip_existing` est active:

- les sorties existantes et non vides sont conservees;
- un SRT existant peut servir a regenerer uniquement l'ASS;
- aucun modele de transcription n'est charge si le SRT est reutilisable.

Lorsque les profils vocaux sont actives, le fichier `.speakers.json` fait aussi
partie des sorties attendues. Son absence provoque une nouvelle analyse audio
afin de ne pas fabriquer ces metadonnees depuis le texte.

## 3. Audio

ffmpeg extrait la premiere piste audio. pydub la convertit en WAV mono 16 kHz,
format commun a pyannote et Cohere. Les fichiers temporaires sont supprimes
meme lorsqu'une video echoue.

## 4. Diarisation Avant Transcription

`PyannoteDiarizer` charge `pyannote/speaker-diarization-community-1` avec
`HUGGINGFACE_TOKEN`. La diarisation est terminee avant le premier chargement de
Cohere. Le WAV normalise est fourni directement comme tenseur afin de ne pas
dependre du decodeur torchcodec de Colab. La diarisation exclusive de
`community-1` fournit les frontieres utilisees par Cohere et evite de produire
deux sous-titres concurrents pendant une parole superposee. La diarisation
reguliere, qui conserve les chevauchements, reste utilisee pour exclure les
passages melanges de l'analyse des profils vocaux. Chaque tour conserve son
identifiant de locuteur.

Sans diarisation, la video constitue un seul tour `SPEAKER_00`. Ce mode ne peut
donc pas garantir qu'une unite ne contienne qu'un personnage.

### Profil Vocal Conserve Pour Le Post-traitement

Plusieurs tours propres et non chevauches de chaque locuteur sont analyses avec
une estimation locale de la frequence fondamentale. Le resultat est stocke sous
`perceived_voice_gender` avec une confiance, la frequence mediane et la quantite
d'audio analysee. La classification ne repose plus sur la seule mediane: les
quartiles de frequence doivent rester du meme cote de la zone ambigue et une
proportion suffisante des trames doit contenir une fondamentale fiable. Les
quartiles et cette proportion sont conserves dans le JSON. Une distribution
instable ou une zone acoustique ambigue produit `unknown`.

Cette estimation n'est ni une identite biologique ni une correction du texte.
Elle n'influence actuellement pas Cohere; elle est seulement conservee dans
`<video>.speakers.json` pour une future validation morphologique et
contextuelle.

## 5. Unites Locuteur/Phrase

`build_speaker_phrase_units(...)` applique les regles suivantes:

1. trier les tours de parole;
2. ignorer les micro-tours trop courts pour etre transcrits, tout en les
   conservant dans les metadonnees brutes;
3. reunir les fragments adjacents attribues au meme locuteur;
4. chercher les pauses acoustiques a l'interieur de chaque tour;
5. couper le tour sur ces pauses;
6. rattacher les fragments trop courts au fragment voisin du meme locuteur;
7. appliquer une limite dure de 12 secondes lorsqu'aucune pause utile
   n'apparait.

Une unite ne traverse jamais une frontiere de locuteur. Les 12 secondes sont
une limite de securite, pas une fenetre periodique.

## 6. Transcription Cohere

`CohereArabicTranscriber` est charge paresseusement lors de la premiere unite,
donc apres la diarisation. Il recoit uniquement l'audio de cette unite et
retourne son texte. Ses limites temporelles restent celles du tour et de la
pause acoustique: aucun autre ASR ni aligneur ne peut remplacer ses mots.

Le modele est execute localement et mis en cache par Hugging Face. Son depot est
soumis a une validation d'acces unique sur sa page Hugging Face. Le token sert
egalement a pyannote; les notebooks renseignent `HF_TOKEN` et
`HUGGINGFACE_TOKEN` avec la meme valeur.

Cohere et pyannote utilisent CUDA lorsqu'un GPU est selectionne. Cohere est
charge en `float16` sur GPU et en `float32` sur CPU.

Le nombre maximal de tokens generes est adapte a la duree de chaque unite, avec
un plancher qui laisse assez de marge a la tokenisation arabe et aux diacritiques.
Une sortie contenant un caractere Unicode de remplacement, une repetition
massive, un token demesure, une fin arabe manifestement tronquee ou trop de texte
pour l'audio est consideree comme suspecte. Cohere effectue alors une seconde
passe avec des contraintes anti-repetition et un budget de securite plus large.
Si celle-ci reste invalide, le passage est omis et consigne dans les logs au lieu
de contaminer le SRT avec une hallucination.

## 7. Construction Du SRT

La sortie de chaque unite devient un sous-titre portant le meme locuteur. Si le
texte depasse le nombre de mots configure, il est coupe sur la ponctuation ou
la limite de mots. Les sous-segments se partagent proportionnellement la duree
de l'unite d'origine et ne peuvent pas franchir sa frontiere de locuteur.

Cette estimation intra-phrase ne constitue pas un timestamp de mot. Le modele
Cohere n'en fournit pas; elle sert uniquement a eviter un bloc de texte trop
long tout en conservant des limites acoustiques fiables.

## 8. Traduction Et ASS

`InterlinearGoogleTranslator` utilise `deep-translator`. Chaque token source est
traduit independamment, mis en cache puis transforme en `WordPair`. En cas
d'echec de Google, le token source reste affiche comme repli; aucun modele de
traduction local concurrent n'est utilise.

Le rendu ASS cree deux evenements centres par paire: le mot source au-dessus et
sa glose cible en dessous. Les paires sont placees de droite a gauche et passent
a la rangee suivante lorsqu'elles depassent la zone sure.

## 9. Colab Et Acces Aux Modeles

Les deux notebooks installent directement `requirements.txt`, actualisent le
depot et affichent la progression video par video. Avant le premier lancement,
le compte lie au token doit avoir accepte les conditions de:

- `pyannote/speaker-diarization-community-1`;
- `CohereLabs/cohere-transcribe-arabic-07-2026`.

Les poids sont telecharges une fois dans le cache du runtime. Une nouvelle
session Colab peut toutefois devoir les telecharger de nouveau si son cache
local a disparu.

## 10. Architecture Et Verification

```text
src/dual_subtitles/
|-- cli.py
|-- core/
|   |-- config.py
|   |-- pipeline.py
|   `-- segmentation.py
|-- io/
|   |-- audio.py
|   |-- speaker_metadata.py
|   `-- subtitle_files.py
|-- models/
|   `-- subtitle.py
`-- services/
    |-- diarization.py
    |-- transcription.py
    |-- translation.py
    `-- voice_profile.py
```

Controles locaux:

```bash
pytest
ruff check src tests
```
