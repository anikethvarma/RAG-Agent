"""A tiny, dependency-free BM25 retriever over the markdown knowledge base.

Each `## section` of each markdown file becomes one chunk. Chunk ids look like
`travel_expense_2025#2`; the doc id (`travel_expense_2025`) is what the
retrieval metrics compare against the dataset's `gold_doc_ids`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_STOPWORDS = set(
    "a an and are as at be by can do does for from get how i if in is it my of on or "
    "per so that the their then there this to was we what when where which who will "
    "with you your me am".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9$]+(?:[.'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str

    def render(self) -> str:
        return f"[{self.doc_id}] {self.title}\n{self.text}"


def load_chunks(kb_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        doc_id = path.stem
        content = path.read_text(encoding="utf-8")
        doc_title_match = re.search(r"^# (.+)$", content, re.M)
        doc_title = doc_title_match.group(1).strip() if doc_title_match else doc_id
        # Split on level-2 headings; the preamble (before the first ##) is chunk 0.
        parts = re.split(r"^## ", content, flags=re.M)
        preamble = re.sub(r"^# .+$", "", parts[0], flags=re.M).strip()
        if preamble:
            chunks.append(Chunk(f"{doc_id}#0", doc_id, doc_title, preamble))
        for i, part in enumerate(parts[1:], start=1):
            heading, _, body = part.partition("\n")
            # Prefix the doc title so status lines like "SUPERSEDED" stay visible per chunk.
            chunks.append(Chunk(f"{doc_id}#{i}", doc_id, f"{doc_title} › {heading.strip()}", body.strip()))
    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._docs = [tokenize(c.title + " " + c.text) for c in chunks]
        self._tf = [Counter(d) for d in self._docs]
        self._avgdl = sum(len(d) for d in self._docs) / max(len(self._docs), 1)
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(set(d))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_dir(cls, kb_dir: Path) -> "BM25Retriever":
        return cls(load_chunks(kb_dir))

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self._tf):
            dl = len(self._docs[i])
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self._idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            scored.append((self.chunks[i], s))
        scored.sort(key=lambda x: x[1], reverse=True)
        # Like a vector store, always return top_k (even weak matches) — this is
        # what makes context_precision drop when top_k is large.
        return scored[:top_k]
