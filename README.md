# Dual Subtitles
<p align="center">
  <img src="docs/images/example-1.png" width="49%" alt="Example 1">
  <img src="docs/images/example-2.png" width="49%" alt="Example 2">
  <br>
  <img src="docs/images/example-3.png" width="49%" alt="Example 3">
  <img src="docs/images/example-4.png" width="49%" alt="Example 4">
</p>

Dual Subtitles transforme des videos `.mp4` en sous-titres bilingues lisibles:

- transcription `.srt` avec Whisper;
- rendu `.ass` pedagogique avec glosses quasi mot a mot contextualisees;
- diacritisation prudente, morphologie, detection et translitteration des noms;
- diarisation optionnelle des locuteurs avec pyannote;
- traitement linguistique local, sans LLM ni API de traduction;
- acceleration CUDA avec repli CPU pour tous les modeles.

La transcription complete d'une video est terminee avant l'analyse linguistique.
La traduction naturelle sert uniquement d'ancrage interne: le spectateur voit la
source et sa glose pedagogique, pas une seconde phrase concurrente.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/git-haddadz/Dual_Subtitles/blob/master/subtitles_gen.ipynb)

Le notebook maintenu est [subtitles_gen.ipynb](subtitles_gen.ipynb). Il utilise
Google Drive pour lire les videos et conserver les sous-titres generes.

## Installation Locale

Prerequis: Python 3.11+, `ffmpeg` et, pour la diarisation, un token Hugging Face
autorise a utiliser `pyannote/speaker-diarization-3.1`.

```bash
pip install -r requirements.txt
pip install -e .
camel_data -i morphology-db-all
camel_data -i disambig-mle-all
camel_data -i ner-arabert
export HUGGINGFACE_TOKEN=your_token_here
dual-subtitles process --input-dir ./videos --output-dir ./subtitles --verbose
```

Sous PowerShell:

```powershell
$env:HUGGINGFACE_TOKEN = "your_token_here"
```

Options utiles:

```bash
dual-subtitles process --input-dir ./videos --output-dir ./subtitles --no-ass
dual-subtitles process --input-dir ./videos --output-dir ./subtitles --no-diarization
```

Les modeles de traduction, d'alignement et de diacritisation sont telecharges
une fois puis reutilises depuis le cache local. La configuration par defaut est
specialisee pour une source arabe et une cible anglaise. Une autre langue cible
reste possible en fournissant un modele de traduction local compatible avec
`--translation-model`.

Le rendu 720p utilise Noto Naskh Arabic en 64 px pour donner la priorite a la
source et Noto Sans en 28 px pour les glosses. Les polices sont fournies avec le
package, copiees dans le dossier de sortie `fonts/` et servent aussi aux mesures
de placement. Installez-les sur la machine de lecture si le lecteur video ne
charge pas automatiquement les polices voisines.

## Documentation

Le fonctionnement interne est detaille dans [docs/pipeline.md](docs/pipeline.md).
