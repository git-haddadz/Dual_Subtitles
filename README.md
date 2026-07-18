# Dual Subtitles
<p align="center">
  <img src="docs/images/example-1.png" width="49%" alt="Example 1">
  <img src="docs/images/example-2.png" width="49%" alt="Example 2">
  <br>
  <img src="docs/images/example-3.png" width="49%" alt="Example 3">
  <img src="docs/images/example-4.png" width="49%" alt="Example 4">
</p>

Dual Subtitles transforme des videos `.mp4` en sous-titres bilingues lisibles:

- transcription arabe `.srt` avec Cohere Transcribe Arabic;
- rendu `.ass` mot a mot, avec traduction litterale alignee;
- diarisation optionnelle des locuteurs avec pyannote;
- decoupage par locuteur et pauses acoustiques avant transcription;
- acceleration CUDA pour Cohere et pyannote.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/git-haddadz/Dual_Subtitles/blob/master/subtitles_gen.ipynb)

Le notebook maintenu est [subtitles_gen.ipynb](subtitles_gen.ipynb). Il utilise
Google Drive pour lire les videos et conserver les sous-titres generes.

## Installation Locale

Prerequis: Python 3.11+, `ffmpeg` et un token Hugging Face autorise a utiliser
`pyannote/speaker-diarization-community-1` ainsi que
`CohereLabs/cohere-transcribe-arabic-07-2026`.

```bash
pip install -r requirements.txt
pip install -e .
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

## Documentation

Le fonctionnement interne est detaille dans [docs/pipeline.md](docs/pipeline.md).
