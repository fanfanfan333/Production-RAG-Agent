"""
Hybrid retrieval helpers (召回质量优化).

Pure-Python Okapi BM25 over character n-grams plus Reciprocal Rank Fusion
(RRF).  Deliberately dependency-free (no jieba / rank-bm25) so the Docker
image stays lean.

Why character bigrams?  The corpus is mostly Chinese, and word segmentation
without a tokenizer is unreliable.  Bigrams (plus whole ASCII runs for codes
like "ABX-300") give stable term statistics without any dependency — and
exact keyword recall is precisely what pure vector search tends to miss.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

_ASCII_RUN_RE = re.compile(r"[a-z0-9]+")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """
    Split *text* into BM25 terms.

    - ASCII alphanumeric runs are kept whole:  "abx-300" → ["abx", "300"]
    - CJK runs become character bigrams:        "知识库"  → ["知识", "识库"]
      (a single CJK char run of length 1 stays a unigram)

    Everything else (punctuation, whitespace, symbols) is dropped.
    """
    text = text.lower()
    tokens: list[str] = _ASCII_RUN_RE.findall(text)

    i = 0
    n = len(text)
    while i < n:
        if _CJK_CHAR_RE.match(text[i]):
            j = i
            while j < n and _CJK_CHAR_RE.match(text[j]):
                j += 1
            run = text[i:j]
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[t : t + 2] for t in range(len(run) - 1))
            i = j
        else:
            i += 1
    return tokens


@dataclass
class BM25Hit:
    """One scored document (index into the corpus list passed to BM25Index)."""

    index: int
    score: float


class BM25Index:
    """
    Okapi BM25 over a fixed corpus of strings.

    Built once per corpus; ``search()`` is cheap and can be called per query.
    Suitable for corpora up to ~10k chunks (see HYBRID_MAX_CORPUS_POINTS).
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_count = len(docs)
        self.doc_len: list[int] = []
        self.tfs: list[Counter] = []
        df: Counter = Counter()

        for doc in docs:
            toks = tokenize(doc)
            self.doc_len.append(len(toks))
            tf = Counter(toks)
            self.tfs.append(tf)
            for term in tf:
                df[term] += 1

        self.avgdl = (sum(self.doc_len) / self.doc_count) if self.doc_count else 0.0
        # BM25+ style non-negative idf
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_n: int) -> list[BM25Hit]:
        """Return the top_n corpus entries matching *query*, score-descending."""
        if self.doc_count == 0 or top_n <= 0:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []

        scores = [0.0] * self.doc_count
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(term)
                if not f:
                    continue
                norm_len = (
                    self.doc_len[i] / self.avgdl if self.avgdl > 0 else 1.0
                )
                denom = f + self.k1 * (1 - self.b + self.b * norm_len)
                scores[i] += idf * f * (self.k1 + 1) / denom

        hits = [
            BM25Hit(index=i, score=s)
            for i, s in enumerate(scores)
            if s > 0.0
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_n]


def rrf_fuse(rank_lists: list[list], k: int = 60) -> dict:
    """
    Reciprocal Rank Fusion over one or more ranked lists of hashable keys.

    Each list is ordered best-first.  Fused score of a key is
    ``sum(1 / (k + rank + 1))`` across every list containing it, where
    ``rank`` is 0-indexed.  Returns a dict key → fused score.
    """
    fused: dict = {}
    for ranks in rank_lists:
        for rank, key in enumerate(ranks):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
    return fused
