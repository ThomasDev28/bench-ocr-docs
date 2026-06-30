#!/usr/bin/env python3
"""HTTP API for PaddleOCR-VL parsing.

Run (on the GPU machine):
    pip install -r requirements.txt        # includes fastapi + uvicorn
    python server.py                       # serves on 0.0.0.0:8000

Endpoints
---------
  GET  /health                 -> {"status": "ok", "model_loaded": bool}
  POST /parse   (multipart)    -> {"data": {"pages": [{page_number, content}, ...]}}
                 form field `file` = the PDF. Returns Markdown per page, writes nothing.
                 query ?json=true also includes structured JSON per page.

Query from your local machine:
    curl -F file=@tarifs.pdf http://REMOTE:8000/parse
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

ROOT = Path(__file__).resolve().parent

app = FastAPI(title="PaddleOCR-VL API")

# Load the model once, lazily, and reuse across requests.
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from paddleocr import PaddleOCRVL
        _pipeline = PaddleOCRVL()
    return _pipeline


def _read_markdown(res, tmp: Path, stem: str) -> str:
    """Get the page Markdown as a string by saving to a temp file and reading back.

    Robust to PaddleX result internals — we don't depend on an attribute name.
    """
    md_path = tmp / f"{stem}.md"
    res.save_to_markdown(save_path=str(md_path))
    # PaddleX may write either the exact path or a file inside it; handle both.
    if md_path.is_file():
        return md_path.read_text(encoding="utf-8")
    hits = list(tmp.glob(f"{stem}*.md"))
    return hits[0].read_text(encoding="utf-8") if hits else ""


# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.post("/parse")
async def parse(file: UploadFile = File(...), json: bool = False):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="file must be a .pdf")

    pipeline = get_pipeline()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pdf_path = tmp / file.filename
        pdf_path.write_bytes(await file.read())

        results = pipeline.predict(str(pdf_path))

        pages = []
        for i, res in enumerate(results, start=1):
            page = {"page_number": i, "content": _read_markdown(res, tmp, f"page_{i:04d}")}
            if json:
                jpath = tmp / f"page_{i:04d}.json"
                res.save_to_json(save_path=str(jpath))
                if jpath.is_file():
                    import json as _json
                    page["json"] = _json.loads(jpath.read_text(encoding="utf-8"))
            pages.append(page)

    return JSONResponse({"data": {"pages": pages}})


if __name__ == "__main__":
    import os
    uvicorn.run(
        "server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
