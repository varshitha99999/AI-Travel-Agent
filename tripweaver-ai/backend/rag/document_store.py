"""
RAG Document Store
------------------
Handles document ingestion, chunking, and retrieval using pure-Python TF-IDF.

No PyTorch, no TensorFlow, no sentence-transformers — works on Python 3.13
with zero compilation requirements.

For production use, swap _TFIDFRetriever for FAISS + HuggingFaceEmbeddings
once a stable embedding stack is available for your Python version.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Lightweight TF-IDF retriever (pure Python, no dependencies)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split into tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


class _TFIDFRetriever:
    """
    In-memory TF-IDF retriever.
    Stores (chunk_text, metadata) pairs and ranks by cosine similarity.
    """

    def __init__(self):
        self._chunks: List[Tuple[str, dict]] = []   # (text, metadata)
        self._tf: List[dict] = []                   # term frequencies per chunk
        self._df: Counter = Counter()               # document frequencies
        self._n: int = 0                            # total chunks

    def add(self, text: str, metadata: dict) -> None:
        tokens = _tokenize(text)
        if not tokens:
            return
        tf = Counter(tokens)
        # Normalise TF
        max_freq = max(tf.values())
        tf_norm = {t: c / max_freq for t, c in tf.items()}
        self._chunks.append((text, metadata))
        self._tf.append(tf_norm)
        for term in set(tokens):
            self._df[term] += 1
        self._n += 1

    def query(self, question: str, k: int = 4) -> List[Tuple[str, dict]]:
        if not self._chunks:
            return []
        q_tokens = _tokenize(question)
        if not q_tokens:
            return []

        scores = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for term in q_tokens:
                if term in tf:
                    idf = math.log((self._n + 1) / (self._df[term] + 1)) + 1
                    score += tf[term] * idf
            scores.append((score, i))

        scores.sort(reverse=True)
        return [
            (self._chunks[i][0], self._chunks[i][1])
            for score, i in scores[:k]
            if score > 0
        ]

    @property
    def size(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Text splitter (no LangChain dependency needed here)
# ---------------------------------------------------------------------------

def _split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    # Split on paragraph boundaries first, then by size
    paragraphs = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # If paragraph itself is too long, split by sentences
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                buf = ""
                for sent in sentences:
                    if len(buf) + len(sent) <= chunk_size:
                        buf = (buf + " " + sent).strip()
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = sent
                if buf:
                    current = buf
                else:
                    current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    # Add overlap: prepend tail of previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append(tail + " " + chunks[i])
        return overlapped

    return chunks


# ---------------------------------------------------------------------------
# Document loaders (minimal, no heavy deps)
# ---------------------------------------------------------------------------

def _load_pdf(path: str) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}") from e


def _load_text(path: str) -> str:
    """Read a plain-text or markdown file."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {e}") from e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TravelDocumentStore:
    """
    In-memory TF-IDF document store for uploaded travel documents.
    One instance per Chainlit/Streamlit user session.
    """

    def __init__(self):
        self._retriever = _TFIDFRetriever()
        self._doc_names: List[str] = []

    def add_file(self, file_path: str, file_name: str) -> int:
        """
        Load, chunk, and index a file.
        Returns the number of chunks added.
        """
        ext = Path(file_name).suffix.lower()
        try:
            if ext == ".pdf":
                text = _load_pdf(file_path)
            else:
                text = _load_text(file_path)
        except Exception as e:
            print(f"[RAG] Failed to load {file_name}: {e}")
            return 0

        if not text.strip():
            return 0

        chunks = _split_text(text, chunk_size=500, overlap=50)
        for i, chunk in enumerate(chunks):
            self._retriever.add(chunk, {"source": file_name, "chunk": i})

        self._doc_names.append(file_name)
        return len(chunks)

    def query(self, question: str, k: int = 4) -> str:
        """
        Retrieve the top-k most relevant chunks for a question.
        Returns a formatted string ready to inject into the LLM prompt.
        """
        results = self._retriever.query(question, k=k)
        if not results:
            return ""

        parts = ["📄 **Relevant info from your uploaded documents:**\n"]
        for i, (text, meta) in enumerate(results, 1):
            source = meta.get("source", "uploaded file")
            chunk_n = meta.get("chunk", "")
            label = f"{source}" + (f" (chunk {chunk_n + 1})" if chunk_n != "" else "")
            parts.append(f"**[{i}] {label}**\n{text.strip()}\n")

        return "\n".join(parts)

    def has_documents(self) -> bool:
        return self._retriever.size > 0

    @property
    def document_names(self) -> List[str]:
        return list(self._doc_names)

    def clear(self):
        self._retriever = _TFIDFRetriever()
        self._doc_names = []
