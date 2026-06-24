# Pipeline Dual Subtitles

Ce document explique la structure actuelle du projet et le chemin complet suivi
par une video, dans le meme ordre que l'ancien notebook Colab.

## Structure Des Dossiers

```text
Dual_Subtitles/
├── src/
│   └── dual_subtitles/
│       ├── __init__.py
│       ├── main.py
│       ├── cli.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── pipeline.py
│       │   └── segmentation.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── subtitle.py
│       ├── io/
│       │   ├── __init__.py
│       │   ├── audio.py
│       │   └── subtitle_files.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── diarization.py
│       │   ├── transcription.py
│       │   └── translation.py
│       └── utils/
│           ├── __init__.py
│           └── timestamps.py
```

## Sens Des Noms

- `core`: logique centrale du pipeline. Ce dossier contient ce qui decide
  l'ordre des etapes, les regles de segmentation et la configuration.
- `models`: objets de donnees simples. Ils decrivent les informations qui
  circulent dans le pipeline, sans appeler Whisper, pyannote ou le disque.
- `io`: entrees/sorties. Ce dossier lit ou ecrit des fichiers: audio, SRT, ASS.
- `services`: wrappers autour de dependances externes. C'est ici que le code
  parle a pyannote, Whisper/Transformers ou Google Translate.
- `utils`: fonctions generiques sans etat metier fort, comme le formatage des
  timestamps.
- `cli.py`: interface en ligne de commande pour lancer le pipeline en local.
- `main.py`: point d'entree executable qui appelle la CLI.

## 1. Notebook Colab

Fichier: `subtitles_gen.ipynb`

Le notebook est un runner Colab. Il ne contient plus la logique metier: il
prepare l'environnement et appelle le package Python.

Ordre des cellules:

1. monter Google Drive;
2. utiliser `/content/drive/MyDrive/Dual_Subtitles` comme dossier du repo;
3. faire `git pull` si le repo existe deja dans Drive;
4. faire `git clone` uniquement si le repo n'existe pas encore;
5. installer `ffmpeg`, `requirements.txt` et le package local;
6. demander le token Hugging Face;
7. traiter les videos de `/content/drive/MyDrive/sous-titres`.

Ce choix evite de recloner le repo dans `/content`, qui disparait a chaque
runtime Colab.

### Gestion Du Token Et De L'environnement

Le notebook ne lit pas `.env.example` et ne cree pas de fichier `.env`.

Dans Colab, le token Hugging Face est demande dans une cellule avec
`getpass.getpass(...)`, puis place uniquement dans la variable d'environnement
du runtime:

```python
os.environ["HUGGINGFACE_TOKEN"] = hf_token
```

Ensuite, `PyannoteDiarizer` lit cette variable depuis
`src/dual_subtitles/services/diarization.py`. Le token reste donc en memoire
pendant la session Colab, mais il n'est pas ecrit dans Google Drive, pas ajoute
au repo et pas stocke dans le notebook.

Le fichier `.env.example` est seulement un modele pour l'execution locale. Il
montre le nom attendu de la variable:

```env
HUGGINGFACE_TOKEN=your_token_here
```

Pour utiliser un vrai `.env` local, il faudrait soit l'exporter manuellement
dans le shell, soit ajouter une dependance comme `python-dotenv`. Le projet ne
charge pas automatiquement `.env` aujourd'hui.

## 2. Configuration

Fichier: `src/dual_subtitles/core/config.py`

`ProcessingConfig` remplace les variables globales de l'ancien notebook:

- dossiers d'entree, sortie et temporaire;
- langues de transcription et traduction;
- modeles Whisper et pyannote;
- seuils de segmentation, fusion et padding;
- activation SRT, ASS et diarisation.

## 3. Orchestration

Fichier: `src/dual_subtitles/core/pipeline.py`

Fonctions principales:

- `discover_videos(...)`: trouve les videos `.mp4`;
- `process_directory(...)`: charge les services une fois et traite chaque video;
- `process_video(...)`: remplace la grande boucle du notebook;
- `prepare_speech_segments(...)`: applique les regles de parole;
- `prepare_subtitles(...)`: nettoie et fusionne les sous-titres;
- `transcribe_segments(...)`: coupe les chunks audio et appelle Whisper.

`process_video(...)` suit cette sequence:

1. definir les chemins `.srt` et `.ass`;
2. si le `.srt` existe deja, le reutiliser et generer le `.ass` manquant;
3. extraire l'audio de la video;
4. convertir l'audio en WAV mono 16 kHz;
5. detecter les segments de parole;
6. transcrire chaque segment;
7. nettoyer/fusionner les sous-titres;
8. ecrire `.srt`;
9. ecrire `.ass`.

## 4. Modeles De Donnees

Fichier: `src/dual_subtitles/models/subtitle.py`

Contient les objets simples du domaine:

- `Segment`: portion audio avec debut, fin et locuteur;
- `SubtitleSegment`: portion sous-titree avec debut, fin, texte et locuteur;
- `InterlinearTranslator`: protocole attendu par la generation ASS.

## 5. Entrees Et Sorties

Fichiers:

- `src/dual_subtitles/io/audio.py`
- `src/dual_subtitles/io/subtitle_files.py`

`audio.py` gere:

- extraction audio depuis `.mp4` avec `moviepy`;
- normalisation mono 16 kHz avec `pydub`;
- chargement audio pour decouper les chunks.

`subtitle_files.py` gere:

- construction et ecriture SRT;
- lecture d'un SRT existant;
- construction et ecriture ASS.

## 6. Services Externes

Fichiers:

- `src/dual_subtitles/services/diarization.py`
- `src/dual_subtitles/services/transcription.py`
- `src/dual_subtitles/services/translation.py`

`diarization.py` charge pyannote et exige `HUGGINGFACE_TOKEN` si la diarisation
est activee.

`transcription.py` charge Whisper via `transformers.pipeline` et retourne des
`SubtitleSegment` timestamps.

`translation.py` utilise `deep-translator` pour construire la ligne traduite des
fichiers ASS.

## 7. Segmentation Et Nettoyage

Fichier: `src/dual_subtitles/core/segmentation.py`

Regroupe les anciennes constantes et boucles du notebook:

- suppression des segments trop courts;
- fusion de segments proches du meme locuteur;
- decoupage des segments trop longs;
- suppression des textes vides;
- correction des chevauchements;
- fusion de petits chunks Whisper;
- ajout de retours ligne dans les sous-titres longs.

## 8. Timestamps

Fichier: `src/dual_subtitles/utils/timestamps.py`

Contient les fonctions de formatage et parsing:

- `format_srt_timestamp(...)`;
- `format_ass_timestamp(...)`;
- `parse_srt_timestamp(...)`.

## 9. CLI

Fichiers:

- `src/dual_subtitles/cli.py`
- `src/dual_subtitles/main.py`

Commande principale:

```bash
dual-subtitles process --input-dir ./videos --output-dir ./subtitles
```

`main.py` permet aussi de lancer le package comme module Python si besoin.
