"""
Semantic file indexer for the filesystem MCP server.

Provides on-demand RAG-style retrieval over large local files so that
read_file(path, query="...") returns the most relevant code sections
instead of a linear truncated head.

Design principles (production-grade, lightweight):
  - Code-aware chunking: AST-split for Python, header-split for Markdown,
    sliding-window for everything else.
  - In-memory LRU cache keyed by (filepath, mtime) — zero re-embedding
    cost on repeated reads of unchanged files.
  - Pure NumPy cosine similarity — no external vector DB required for
    single-file retrieval.
  - Embedding model: openai/text-embedding-3-small via OpenRouter
    (fast ≈1s, cheap $0.02/1M tokens, reuses existing infra).
  - Fallback: if embedding fails for any reason, return the top-K chunks
    selected by simple BM25-style token overlap as a free fallback.
"""

from __future__ import annotations

import ast
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Code-optimised embedding model available on OpenRouter (HK-compatible).
# Codestral-embed is specifically trained on code and outperforms general-purpose
# models on function/class retrieval tasks.
# Fallback: baai/bge-m3 (multilingual, 1024-dim, also HK-available).
EMBED_MODEL = "mistralai/codestral-embed-2505"
EMBED_MODEL_FALLBACK = "baai/bge-m3"

# Chunking parameters
WINDOW_LINES = 60        # lines per sliding-window chunk (non-Python)
WINDOW_OVERLAP = 15      # overlap between consecutive windows
MAX_CHUNK_CHARS = 4000   # hard cap so we stay well within 8k token limit

# Retrieval
TOP_K = 5                # chunks returned per query
MIN_SCORE = 0.0          # minimum cosine similarity to include (0 = return all top-K)

# Cache
CACHE_MAX_FILES = 30     # max distinct files kept in memory


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FileChunk:
    """One semantically coherent unit of a file."""
    path: str
    start_line: int
    end_line: int
    text: str           # content (may include line numbers for readability)
    label: str = ""     # human label, e.g. "def fibonacci" or "## Installation"
    embedding: list[float] = field(default_factory=list)


@dataclass
class _CacheEntry:
    mtime: float
    chunks: list[FileChunk]   # pre-embedded


# ── In-memory LRU cache ───────────────────────────────────────────────────────

_cache: dict[str, _CacheEntry] = {}   # key = absolute filepath
_cache_access: dict[str, float] = {}  # key -> last-access timestamp (for LRU eviction)


def _cache_get(filepath: str, mtime: float) -> Optional[list[FileChunk]]:
    entry = _cache.get(filepath)
    if entry and entry.mtime == mtime:
        _cache_access[filepath] = time.monotonic()
        return entry.chunks
    return None


def _cache_put(filepath: str, mtime: float, chunks: list[FileChunk]) -> None:
    if len(_cache) >= CACHE_MAX_FILES:
        # Evict the least-recently-used entry
        lru_key = min(_cache_access, key=lambda k: _cache_access.get(k, 0))
        _cache.pop(lru_key, None)
        _cache_access.pop(lru_key, None)
    _cache[filepath] = _CacheEntry(mtime=mtime, chunks=chunks)
    _cache_access[filepath] = time.monotonic()


# ── Chunking strategies ───────────────────────────────────────────────────────

def _number_lines(lines: list[str], start: int) -> str:
    """Render a list of lines with 1-based absolute line numbers."""
    return "\n".join(f"{start + i:4d} | {line}" for i, line in enumerate(lines))


def _chunk_python(path: str, source: str, lines: list[str]) -> list[FileChunk]:
    """
    AST-based chunking for Python files.

    Top-level classes and functions become individual chunks.
    Module-level code between definitions is grouped into a 'module preamble' chunk.
    Falls back to sliding-window if the AST cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _chunk_sliding_window(path, lines)

    # Collect top-level definitions with their line ranges
    top_nodes: list[tuple[int, int, str]] = []  # (start, end, label)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(node, "end_lineno", node.lineno)
            label = f"{'class' if isinstance(node, ast.ClassDef) else 'def'} {node.name}"
            top_nodes.append((node.lineno, end, label))

    if not top_nodes:
        return _chunk_sliding_window(path, lines)

    chunks: list[FileChunk] = []
    prev_end = 0

    # Preamble: everything before the first top-level definition
    if top_nodes[0][0] > 1:
        preamble_lines = lines[: top_nodes[0][0] - 1]
        if preamble_lines:
            chunks.append(FileChunk(
                path=path,
                start_line=1,
                end_line=top_nodes[0][0] - 1,
                text=_number_lines(preamble_lines, 1),
                label="module preamble",
            ))
    prev_end = top_nodes[0][0] - 1

    for start_ln, end_ln, label in top_nodes:
        # Gap between previous definition and this one
        if start_ln > prev_end + 1:
            gap = lines[prev_end: start_ln - 1]
            if any(l.strip() for l in gap):
                chunks.append(FileChunk(
                    path=path,
                    start_line=prev_end + 1,
                    end_line=start_ln - 1,
                    text=_number_lines(gap, prev_end + 1),
                    label="module-level code",
                ))

        body = lines[start_ln - 1: end_ln]
        # Respect MAX_CHUNK_CHARS: split very long definitions into windows
        text = _number_lines(body, start_ln)
        if len(text) > MAX_CHUNK_CHARS:
            sub = _chunk_sliding_window(path, body, base_line=start_ln)
            for c in sub:
                c.label = label
            chunks.extend(sub)
        else:
            chunks.append(FileChunk(
                path=path,
                start_line=start_ln,
                end_line=end_ln,
                text=text,
                label=label,
            ))
        prev_end = end_ln

    # Trailing code after last definition
    if prev_end < len(lines):
        tail = lines[prev_end:]
        if any(l.strip() for l in tail):
            chunks.append(FileChunk(
                path=path,
                start_line=prev_end + 1,
                end_line=len(lines),
                text=_number_lines(tail, prev_end + 1),
                label="module tail",
            ))

    return chunks


def _chunk_markdown(path: str, lines: list[str]) -> list[FileChunk]:
    """
    Header-based chunking for Markdown files.

    Each ## or ### section becomes a chunk. Content before the first
    header is a 'preamble' chunk.
    """
    chunks: list[FileChunk] = []
    current_start = 1
    current_label = "preamble"
    current_lines: list[str] = []

    header_re = re.compile(r"^#{1,3}\s+(.+)")

    for i, line in enumerate(lines, start=1):
        m = header_re.match(line)
        if m and current_lines:
            chunks.append(FileChunk(
                path=path,
                start_line=current_start,
                end_line=i - 1,
                text=_number_lines(current_lines, current_start),
                label=current_label,
            ))
            current_start = i
            current_label = m.group(1).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        chunks.append(FileChunk(
            path=path,
            start_line=current_start,
            end_line=len(lines),
            text=_number_lines(current_lines, current_start),
            label=current_label,
        ))

    return chunks or _chunk_sliding_window(path, lines)


def _chunk_sliding_window(
    path: str,
    lines: list[str],
    base_line: int = 1,
) -> list[FileChunk]:
    """
    Fixed-size sliding window with overlap — the universal fallback.

    Used for JSON, YAML, TypeScript, shell scripts, and any file whose
    structure cannot be parsed semantically.
    """
    chunks: list[FileChunk] = []
    step = WINDOW_LINES - WINDOW_OVERLAP
    total = len(lines)
    pos = 0

    while pos < total:
        end = min(pos + WINDOW_LINES, total)
        window = lines[pos:end]
        abs_start = base_line + pos
        abs_end = base_line + end - 1
        chunks.append(FileChunk(
            path=path,
            start_line=abs_start,
            end_line=abs_end,
            text=_number_lines(window, abs_start),
            label=f"lines {abs_start}–{abs_end}",
        ))
        if end == total:
            break
        pos += step

    return chunks


def chunk_file(path: str, source: str) -> list[FileChunk]:
    """Dispatch to the appropriate chunker based on file extension."""
    lines = source.splitlines()
    ext = Path(path).suffix.lower()

    if ext == ".py":
        return _chunk_python(path, source, lines)
    if ext in (".md", ".mdx", ".rst"):
        return _chunk_markdown(path, lines)
    return _chunk_sliding_window(path, lines)


# ── Embedding ─────────────────────────────────────────────────────────────────

async def _embed_chunks(chunks: list[FileChunk]) -> list[FileChunk]:
    """
    Embed all chunks in a single batched API call.

    Tries EMBED_MODEL first; falls back to EMBED_MODEL_FALLBACK if the
    primary is unavailable (e.g. region restriction).  If both fail, each
    chunk keeps an empty embedding vector and BM25 keyword scoring is used.
    """
    from src.rag.embeddings import EmbeddingModel

    texts = [f"{c.label}\n\n{c.text}" for c in chunks]

    for model_id in (EMBED_MODEL, EMBED_MODEL_FALLBACK):
        try:
            model = EmbeddingModel(model_id)
            vectors = await model.embed_batch(texts)
            for chunk, vec in zip(chunks, vectors):
                chunk.embedding = vec
            logger.debug("file_index: embedded %d chunks with %s", len(chunks), model_id)
            return chunks
        except Exception as exc:
            logger.warning(
                "file_index: embedding with %s failed (%s); trying next model", model_id, exc
            )

    logger.warning(
        "file_index: all embedding models failed for %s; BM25 fallback active",
        chunks[0].path if chunks else "?",
    )
    return chunks


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _bm25_score(query: str, text: str) -> float:
    """
    Lightweight BM25-inspired token overlap score used when embeddings
    are unavailable. Not a full BM25 implementation — just IDF-weighted
    token overlap normalized by query length.
    """
    query_tokens = set(re.findall(r"\w+", query.lower()))
    text_tokens = re.findall(r"\w+", text.lower())
    if not query_tokens or not text_tokens:
        return 0.0
    text_freq: dict[str, int] = {}
    for t in text_tokens:
        text_freq[t] = text_freq.get(t, 0) + 1
    score = sum(
        math.log(1 + text_freq.get(tok, 0))
        for tok in query_tokens
    )
    return score / len(query_tokens)


def _retrieve(
    query_vec: list[float],
    query_text: str,
    chunks: list[FileChunk],
    k: int = TOP_K,
) -> list[tuple[float, FileChunk]]:
    """
    Rank chunks by cosine similarity (or BM25 fallback) and return top-K.
    Consecutive chunks that are both in the top-K are merged into one
    contiguous result block to give the agent better context.
    """
    use_embeddings = bool(query_vec) and any(c.embedding for c in chunks)

    if use_embeddings:
        scored = [(1 - _cosine(query_vec, c.embedding), c) for c in chunks]
        # Sort ascending by distance (= descending similarity)
        scored.sort(key=lambda x: x[0])
        top = [(1 - dist, chunk) for dist, chunk in scored[:k]]
    else:
        scored = [(_bm25_score(query_text, c.text), c) for c in chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

    return top


# ── Public API ────────────────────────────────────────────────────────────────

async def semantic_read(
    filepath: str,
    query: str,
    k: int = TOP_K,
) -> str:
    """
    Entry point called by read_file when a large file needs semantic retrieval.

    Returns a formatted string containing the top-K most relevant chunks,
    with line-range headers so the agent can request adjacent context via
    start_line/end_line if needed.
    """
    abs_path = str(Path(filepath).resolve())
    try:
        p = Path(abs_path)
        if p.is_dir():
            return (
                f"[file_index] '{filepath}' is a directory. "
                "Use list_directory to browse it, or provide a specific file path."
            )
        mtime = os.path.getmtime(abs_path)
        source = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[file_index] Cannot read {filepath}: {exc}"

    # ── 1. Get or build chunk index ────────────────────────────────────────
    cached = _cache_get(abs_path, mtime)
    if cached is not None:
        chunks = cached
        logger.debug("file_index: cache hit for %s (%d chunks)", filepath, len(chunks))
    else:
        logger.debug("file_index: indexing %s …", filepath)
        chunks = chunk_file(filepath, source)
        chunks = await _embed_chunks(chunks)
        _cache_put(abs_path, mtime, chunks)
        logger.debug("file_index: indexed %s → %d chunks", filepath, len(chunks))

    if not chunks:
        return f"[file_index] No chunks extracted from {filepath}"

    # ── 2. Embed the query (same model that was used for the chunks) ──────
    query_vec: list[float] = []
    if any(c.embedding for c in chunks):
        from src.rag.embeddings import EmbeddingModel
        for model_id in (EMBED_MODEL, EMBED_MODEL_FALLBACK):
            try:
                model = EmbeddingModel(model_id)
                query_vec = await model.embed(query)
                break
            except Exception as exc:
                logger.warning("file_index: query embed failed with %s (%s)", model_id, exc)

    # ── 3. Retrieve top-K chunks ───────────────────────────────────────────
    results = _retrieve(query_vec, query, chunks, k=k)

    # Sort results by line number for readability
    results.sort(key=lambda x: x[1].start_line)

    # ── 4. Format output ───────────────────────────────────────────────────
    total_lines = len(source.splitlines())
    method = "semantic" if query_vec else "keyword (BM25)"
    header = (
        f"File: {filepath} ({total_lines} lines total)\n"
        f"Query: \"{query}\" — top {len(results)} chunks by {method} relevance\n"
        f"To read adjacent lines, call read_file(path, start_line=X, end_line=Y).\n"
    )

    sections: list[str] = []
    for score, chunk in results:
        sections.append(
            f"\n── {chunk.label} (lines {chunk.start_line}–{chunk.end_line}, score={score:.3f}) ──\n"
            + chunk.text
        )

    return header + "\n".join(sections)


def clear_cache(filepath: str | None = None) -> None:
    """Evict one file (or the entire cache) from the in-memory index."""
    global _cache, _cache_access
    if filepath is None:
        _cache.clear()
        _cache_access.clear()
    else:
        abs_path = str(Path(filepath).resolve())
        _cache.pop(abs_path, None)
        _cache_access.pop(abs_path, None)
