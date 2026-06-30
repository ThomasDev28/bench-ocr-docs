#!/usr/bin/env python3
"""OCR runner — convert PDFs to per-page Markdown with PaddleOCR-VL.

Commands
--------
  run    : OCR every PDF in the input dir, write Markdown to the output dir.
  purge  : delete generated Markdown (and JSON) from the output dir.
  list   : show input PDFs and existing output.

Examples
--------
  python run.py run
  python run.py run --input input --output output --combine
  python run.py run --pdf input/tarifs.pdf
  python run.py purge
  python run.py purge --pdf tarifs        # only that doc's output
  python run.py list
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "input"
DEFAULT_OUTPUT = ROOT / "output"


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
def load_pipeline():
    """Lazy-load PaddleOCR-VL so `purge`/`list` work without the heavy import."""
    try:
        from paddleocr import PaddleOCRVL
    except ImportError:
        sys.exit(
            "PaddleOCR-VL not installed.\n"
            "  pip install -r requirements.txt\n"
            "  (and a paddlepaddle-gpu build matching your CUDA — see requirements.txt)"
        )
    return PaddleOCRVL()


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _collect_pdfs(paths: list[str], in_dir: Path) -> list[Path]:
    """Resolve CLI args into a flat PDF list. Each arg = a PDF file or a dir."""
    out: list[Path] = []
    targets = [Path(p) for p in paths] if paths else [in_dir]
    for t in targets:
        if t.is_dir():
            out += sorted(t.glob("*.pdf"))
        elif t.is_file() and t.suffix.lower() == ".pdf":
            out.append(t)
        else:
            print(f"warn  ignored (not a PDF / not found): {t}")
    return out


def cmd_run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    pdfs = _collect_pdfs(args.paths, Path(args.input))

    if not pdfs:
        sys.exit("No PDF to process. Pass a PDF path or dir: python run.py run <path.pdf>")

    print(f"Loading PaddleOCR-VL … (first run downloads model weights)")
    pipeline = load_pipeline()

    for pdf in pdfs:
        doc_out = out_dir / pdf.stem
        if doc_out.exists() and not args.force:
            print(f"skip  {pdf.name}  (output exists, use --force to redo)")
            continue
        doc_out.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        print(f"ocr   {pdf.name} …", flush=True)
        results = pipeline.predict(str(pdf))  # one result object per page

        page_files: list[Path] = []
        for i, res in enumerate(results, start=1):
            stem = f"page_{i:04d}"
            res.save_to_markdown(save_path=str(doc_out / f"{stem}.md"))
            if args.json:
                res.save_to_json(save_path=str(doc_out / f"{stem}.json"))
            page_files.append(doc_out / f"{stem}.md")

        if args.combine:
            _combine(doc_out, pdf.stem, page_files)

        dt = time.time() - t0
        print(f"done  {pdf.name}  →  {doc_out}  ({len(page_files)} pages, {dt:.1f}s)")


def _combine(doc_out: Path, stem: str, page_files: list[Path]) -> None:
    """Concatenate per-page Markdown into one file with page separators."""
    combined = doc_out / f"{stem}.md"
    parts = []
    for i, pf in enumerate(sorted(page_files), start=1):
        if pf.exists():
            parts.append(f"<!-- page {i} -->\n\n{pf.read_text(encoding='utf-8')}")
    combined.write_text("\n\n---\n\n".join(parts), encoding="utf-8")
    print(f"      combined → {combined.name}")


# --------------------------------------------------------------------------- #
# purge
# --------------------------------------------------------------------------- #
def cmd_purge(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    if not out_dir.exists():
        print(f"Nothing to purge — {out_dir} does not exist.")
        return

    if args.pdf:
        target = out_dir / Path(args.pdf).stem
        targets = [target] if target.exists() else []
    else:
        targets = [p for p in out_dir.iterdir() if p.is_dir()]

    if not targets:
        print("Nothing to purge.")
        return

    print("Will delete:")
    for t in targets:
        print(f"  {t}")
    if not args.yes:
        if input("Confirm? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return

    for t in targets:
        shutil.rmtree(t)
        print(f"removed {t}")


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
def cmd_list(args: argparse.Namespace) -> None:
    in_dir = Path(args.input)
    out_dir = Path(args.output)

    print(f"Input  ({in_dir}):")
    pdfs = sorted(in_dir.glob("*.pdf")) if in_dir.exists() else []
    for p in pdfs:
        print(f"  {p.name}")
    if not pdfs:
        print("  (none)")

    print(f"\nOutput ({out_dir}):")
    docs = sorted(p for p in out_dir.iterdir() if p.is_dir()) if out_dir.exists() else []
    for d in docs:
        n = len(list(d.glob("page_*.md")))
        print(f"  {d.name}/  ({n} pages)")
    if not docs:
        print("  (none)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="OCR PDFs to per-page Markdown")
    r.add_argument("paths", nargs="*", help="PDF file(s) or dir(s). Omit = use --input")
    r.add_argument("--input", default=str(DEFAULT_INPUT), help="fallback dir if no path given")
    r.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output dir")
    r.add_argument("--combine", action="store_true", help="also write one merged .md per doc")
    r.add_argument("--json", action="store_true", help="also dump structured JSON per page")
    r.add_argument("--force", action="store_true", help="re-OCR even if output exists")
    r.set_defaults(func=cmd_run)

    pu = sub.add_parser("purge", help="delete generated Markdown/JSON")
    pu.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output dir")
    pu.add_argument("--pdf", help="purge only this doc's output (by stem)")
    pu.add_argument("--yes", "-y", action="store_true", help="skip confirmation")
    pu.set_defaults(func=cmd_purge)

    ls = sub.add_parser("list", help="show input PDFs and existing output")
    ls.add_argument("--input", default=str(DEFAULT_INPUT))
    ls.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ls.set_defaults(func=cmd_list)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
