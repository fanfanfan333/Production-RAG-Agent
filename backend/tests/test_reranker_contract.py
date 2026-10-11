"""Reranking must preserve distinct evidence and return calibrated scores."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from app.services import reranker as rr


def test_dedup_preserves_different_tails():
    prefix = "共同前文" * 300
    chunks = [SimpleNamespace(text=prefix + tail) for tail in ("答案一", "答案二", "答案一")]
    assert rr.dedup_candidates(chunks) == chunks[:2]


@pytest.mark.parametrize("backend", ["flagreranker", "cross_encoder"])
@pytest.mark.parametrize("logits", [[-10.0], [-10.0, 10.0]])
def test_model_outputs_are_calibrated(monkeypatch, backend, logits):
    def predict(pairs, *, show_progress_bar, activation_fn):
        assert len(pairs) == len(logits)
        return activation_fn(np.asarray(logits))

    scorer = object.__new__(rr.ChunkReranker)
    scorer._backend = backend
    scorer._model = SimpleNamespace(
        predict=predict,
        compute_score=lambda pairs, normalize: np.asarray(logits),
    )
    monkeypatch.setattr(rr, "get_settings", lambda: SimpleNamespace(RERANKER_MAX_TEXT_CHARS=0))
    scores = scorer._score_pairs_sync("query", ["text"] * len(logits), [0.5] * len(logits))
    assert scores == pytest.approx([rr._sigmoid(x) for x in logits])
    assert scores[0] < 0.001


@pytest.mark.parametrize("result", [RuntimeError("inference failed"), [0.9], [0.9, float("nan")], [0.9, 1.1]])
def test_failed_or_invalid_scores_do_not_reuse_coarse_ranking(monkeypatch, result):
    async def score_pairs(*args):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(rr, "get_settings", lambda: SimpleNamespace(RERANKER_USE_QUERY_VARIANTS=False))
    monkeypatch.setattr(rr, "get_reranker", lambda: SimpleNamespace(
        score_pairs=score_pairs, backend_name=lambda: "test",
    ))
    chunks = [SimpleNamespace(text=text, score=0.8) for text in ("a", "b")]
    assert asyncio.run(rr.rerank_chunks("query", chunks, 2)) == []
    assert [c.score for c in chunks] == [0.8, 0.8]
