from scripts.benchmark_retrieval import percentile


def test_latency_percentiles_use_nearest_rank():
    assert percentile([8, 1, 5, 3], .50) == 3
    assert percentile(list(range(1, 101)), .95) == 95
    assert percentile([2], .99) == 2
