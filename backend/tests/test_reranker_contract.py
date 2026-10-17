"""Reranking must preserve distinct evidence and return calibrated scores."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from app.services import reranker as rr

def test_explicit_figure_number_excludes_other_images_without_changing_scores():
    target = SimpleNamespace(text="图表2\n收入531万元", content_type="image", score=.8)
    wrong = SimpleNamespace(text="图表6项目收入655万元", content_type="image", score=.99)
    text = SimpleNamespace(text="图表说明", content_type="text", score=.7)
    assert rr.filter_evidence_references("图表2中2025年的收入", [wrong, target, text]) == [target, text]
    assert target.score == .8
    assert rr.filter_evidence_references("图表2和图表6比较", [wrong, target]) == [wrong, target]
    assert rr.filter_evidence_references("2025年收入", [wrong, target]) == [wrong, target]
    assert rr.filter_evidence_references("图表2的收入", [SimpleNamespace(text="图表20收入", content_type="image")]) == []
    assert rr.image_reference_matches("图表2的收入", target)
    assert not rr.image_reference_matches("图表2的收入", wrong)
    assert rr.filter_evidence_references("图表2的收入", [wrong] * 12 + [target])[:12] == [target]

def test_dedup_preserves_different_tails():
    prefix = "共同前文" * 300
    chunks = [SimpleNamespace(text=prefix + tail) for tail in ("答案一", "答案二", "答案一")]
    assert rr.dedup_candidates(chunks) == chunks[:2]


def test_table_year_identity_keeps_undated_and_comparison_evidence():
    current = SimpleNamespace(text="2023年预算407万元", content_type="table")
    old = SimpleNamespace(text="2022年预算307万元", content_type="image")
    undated = SimpleNamespace(text="预算情况", content_type="table")
    multi = SimpleNamespace(text="| 年份 | 预算 |\n| --- | --- |\n| 2022 | 307 |\n| 2023 | 407 |", content_type="table")
    chunks = [old, current, undated, multi]
    assert rr.filter_evidence_references("2023年预算", chunks) == [current, undated, multi]
    assert rr.filter_evidence_references("2022年和2023年比较", chunks) == chunks
    assert rr.filter_evidence_references("ABX-2023预算", chunks) == chunks


@pytest.mark.parametrize("backend", ["flagreranker", "cross_encoder"])
@pytest.mark.parametrize("logits", [[-10.0], [-10.0, 10.0]])
def test_model_outputs_are_calibrated(monkeypatch, backend, logits):
    def predict(pairs, *, show_progress_bar, activation_fn, batch_size):
        assert len(pairs) == len(logits)
        assert batch_size == 4
        return activation_fn(np.asarray(logits))

    scorer = object.__new__(rr.ChunkReranker)
    scorer._backend = backend
    scorer._model = SimpleNamespace(
        predict=predict,
        compute_score=lambda pairs, normalize, batch_size: np.asarray(logits),
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


def test_shared_reranker_limits_concurrent_model_forwards(monkeypatch):
    import time
    active, peak = 0, 0
    def score(query, texts, vectors):
        nonlocal active, peak
        active += 1; peak = max(peak, active)
        time.sleep(.02)
        active -= 1
        return [.5] * len(texts)
    scorer = object.__new__(rr.ChunkReranker)
    monkeypatch.setattr(scorer, "_score_pairs_sync", score)
    async def run():
        return await asyncio.gather(*(scorer.score_pairs("q", ["text"], [.2]) for _ in range(3)))
    assert asyncio.run(run()) == [[.5]] * 3 and peak == 1


def test_reranker_scores_query_relevant_tail_within_existing_budget(monkeypatch):
    pairs_seen = []
    def compute_score(pairs, **kwargs):
        pairs_seen.extend(pairs)
        return [1.0]
    scorer = object.__new__(rr.ChunkReranker)
    scorer._backend = 'flagreranker'
    scorer._model = SimpleNamespace(compute_score=compute_score)
    monkeypatch.setattr(rr, 'get_settings', lambda: SimpleNamespace(RERANKER_MAX_TEXT_CHARS=600))
    text = 'Unrelated summary conditions. ' * 80 + '\n\nGraph indexing took 281 minutes for the Podcast dataset.'
    scorer._score_pairs_sync('Podcast graph indexing time minutes', [text], [.5])
    assert '281 minutes' in pairs_seen[0][1]
    assert len(pairs_seen[0][1]) <= 600
