# bench-ocr-output-md

PDF → per-page Markdown with **PaddleOCR-VL** (GPU NVIDIA).

## Setup

```bash
# 1. PaddlePaddle GPU build matching your CUDA (example = CUDA 12.6)
python -m pip install paddlepaddle-gpu==3.2.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 2. PaddleOCR + document parser
pip install -r requirements.txt
```

First `run` downloads model weights automatically (cached in `~/.paddlex`).

## Use

```bash
# Drop PDFs in input/, then:
python run.py run                      # OCR every PDF in input/
python run.py run --pdf input/tarifs.pdf   # single file
python run.py run --combine            # also one merged .md per doc
python run.py run --json               # also structured JSON per page
python run.py run --force              # re-OCR even if output exists

python run.py list                     # show inputs + existing output
python run.py purge                    # delete all generated md (asks confirm)
python run.py purge --pdf tarifs -y    # delete one doc's output, no prompt
```

## Output layout

```
output/
└── <pdf_stem>/
    ├── page_0001.md
    ├── page_0002.md
    ├── <pdf_stem>.md      # only with --combine
    └── page_0001.json     # only with --json
```

Tables come out as inline HTML, formulas as LaTeX.
```
