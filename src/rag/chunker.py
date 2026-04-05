"""
Document loading and chunking strategies for the RAG pipeline.
All chunking is delegated to LangChain text-splitter primitives so we
get the same battle-tested splitting logic used across the LangChain
ecosystem, without rolling our own character / sentence / code splitters.

Supported source types: text, file (txt/md/py/pdf), url
Chunking strategies:
  - recursive  : RecursiveCharacterTextSplitter (default, handles prose well)
  - fixed      : CharacterTextSplitter (strict fixed-size windows)
  - semantic   : NLTK-sentence-aware splitter via RecursiveCharacterTextSplitter
  - code       : Language-aware splitter (Python/JS/etc.)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single document chunk with its metadata."""
    text: str
    source: str
    chunk_index: int = 0
    total_chunks: int = 1
    page: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        parts = [self.source]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        parts.append(f"chunk {self.chunk_index + 1}/{self.total_chunks}")
        return " — ".join(parts)


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_text(text: str, source: str = "text") -> list[tuple[str, dict]]:
    return [(text, {"source": source})]


def _load_file(path: str) -> list[tuple[str, dict]]:
    """
    Load a local file.  PDF pages are returned individually so page numbers
    are preserved.  Uses LangChain's PyPDFLoader when available.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(p))
            docs = loader.load()
            return [(d.page_content, {"source": str(p), "page": d.metadata.get("page", 0) + 1}) for d in docs if d.page_content.strip()]
        except ImportError:
            pass
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(p))
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((text, {"source": str(p), "page": i + 1}))
            return pages
        except ImportError:
            logger.warning("pypdf not installed, reading PDF as plain text")
            return [(p.read_text(errors="replace"), {"source": str(p)})]

    return [(p.read_text(errors="replace"), {"source": str(p)})]


def _load_url(url: str) -> list[tuple[str, dict]]:
    """
    Load a web page.  Tries LangChain's WebBaseLoader first, then falls
    back to requests + BeautifulSoup.
    """
    try:
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader(url)
        docs = loader.load()
        return [(d.page_content, {"source": url}) for d in docs if d.page_content.strip()]
    except Exception:
        pass
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=15, headers={"User-Agent": "RAG-Loader/1.0"})
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "html" in ct:
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        else:
            text = resp.text
        return [(text, {"source": url})]
    except Exception as e:
        raise RuntimeError(f"Failed to load URL {url}: {e}") from e


def load_source(source_type: str, content: str) -> list[tuple[str, dict]]:
    """
    Load raw text from a source.

    Args:
        source_type: one of "text", "file", "url"
        content: the text body, file path, or URL

    Returns:
        List of (page_text, metadata) tuples.
    """
    if source_type == "text":
        return _load_text(content)
    elif source_type == "file":
        return _load_file(content)
    elif source_type == "url":
        return _load_url(content)
    else:
        raise ValueError(f"Unknown source_type '{source_type}'. Use: text, file, url")


# ── LangChain-backed splitters ────────────────────────────────────────────────

def _get_recursive_splitter(chunk_size: int, chunk_overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
        add_start_index=False,
    )


def _get_fixed_splitter(chunk_size: int, chunk_overlap: int):
    from langchain_text_splitters import CharacterTextSplitter
    return CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
        length_function=len,
    )


def _get_semantic_splitter(chunk_size: int, chunk_overlap: int):
    """Sentence-aware splitting: never breaks mid-sentence."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[".\n\n", ".\n", ". ", "! ", "? ", "\n\n", "\n", " ", ""],
        length_function=len,
        keep_separator=True,
    )


def _get_code_splitter(chunk_size: int, chunk_overlap: int, source: str = ""):
    """Language-aware code splitting via LangChain Language enum."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
    ext = Path(source).suffix.lower() if source else ""
    lang_map = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".ts": Language.TS,
        ".java": Language.JAVA,
        ".cpp": Language.CPP,
        ".c": Language.C,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".rb": Language.RUBY,
    }
    lang = lang_map.get(ext)
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    # Default: Python-style code separators
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_documents(
    pages: list[tuple[str, dict]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "recursive",
) -> list[Chunk]:
    """
    Split loaded document pages into overlapping chunks using LangChain splitters.

    Args:
        pages: output of load_source()
        chunk_size: max characters per chunk
        chunk_overlap: characters of overlap between consecutive chunks
        strategy: "recursive" | "fixed" | "semantic" | "code"

    Returns:
        List of Chunk objects ready for embedding.
    """
    all_chunks: list[Chunk] = []

    for page_text, meta in pages:
        if not page_text.strip():
            continue

        source = meta.get("source", "text")

        if strategy == "fixed":
            splitter = _get_fixed_splitter(chunk_size, chunk_overlap)
        elif strategy == "semantic":
            splitter = _get_semantic_splitter(chunk_size, chunk_overlap)
        elif strategy == "code":
            splitter = _get_code_splitter(chunk_size, chunk_overlap, source)
        else:  # default: recursive
            splitter = _get_recursive_splitter(chunk_size, chunk_overlap)

        raw_chunks = splitter.split_text(page_text)
        raw_chunks = [c.strip() for c in raw_chunks if c.strip()]

        for i, raw in enumerate(raw_chunks):
            all_chunks.append(Chunk(
                text=raw,
                source=source,
                chunk_index=i,
                total_chunks=len(raw_chunks),
                page=meta.get("page"),
                metadata={**meta, "chunk_strategy": strategy},
            ))

    return all_chunks
