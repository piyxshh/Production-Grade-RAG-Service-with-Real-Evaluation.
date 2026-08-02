"""
Document loaders: read raw files from corpus/raw/ and return
a list of {text: str, metadata: dict} objects.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus" / "raw"


def load_documents(corpus_dir: Path | str | None = None) -> list[dict]:
    """Load every supported file in `corpus_dir` into a list of documents.

    Each document is a dict: ``{"text": str, "metadata": {"filename": str}}``.
    Files that cannot be parsed or contain no text are skipped with a warning.
    """
    corpus_path = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
    if not corpus_path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")

    documents: list[dict] = []
    for path in sorted(corpus_path.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            text = _read_file(path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            continue
        if not text.strip():
            logger.warning("Skipping %s: empty document", path.name)
            continue
        documents.append({"text": text, "metadata": {"filename": path.name}})
    return documents


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return _read_text(path)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "Reading PDFs requires pypdf. Install it with: pip install pypdf"
        ) from exc
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)
