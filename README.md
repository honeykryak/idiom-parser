# Phraseological Expression Detector — «вешать лапшу на уши»

Classifies sentences containing **«лапша»** into three groups:

| Output file         | Group        | Rule                                                   |
| ------------------- | ------------ | ------------------------------------------------------ |
| `idiom.xlsx`        | Full idiom   | «лапша» + verb «вешать / повесить» (any form/tense)    |
| `segmentation.xlsx` | Segmentation | «лапша» used figuratively (deception), **no** «вешать» |
| `literal.xlsx`      | Literal      | «лапша» used literally (food)                          |

## Quick start

```bash
pip install -r requirements.txt
python classify.py data.xlsx
```

Three output files appear next to the input file.

## If HuggingFace is blocked (Russia)

Set the mirror **before** the first run so the model can be downloaded:

```powershell
# PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python classify.py data.xlsx
```

```bash
# bash / Linux
export HF_ENDPOINT="https://hf-mirror.com"
python classify.py data.xlsx
```

After the first run the model is cached locally and no internet is needed.

## Project structure

```
classify.py      — entry point (argv parsing, pipeline orchestration)
config.py        — meanings, model name, reference phrases
morphology.py    — Natasha-based lemmatisation
semantics.py     — Sentence Transformers–based similarity classification
io_excel.py      — Excel read / write helpers
requirements.txt — Python dependencies
```
