from types import SimpleNamespace

import pytest

from app.services.vision import vision_service as vision


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "unit-test-only")
    monkeypatch.setattr(vision.VisionService, "_instance", None)
    return vision.VisionService()


def test_identical_image_and_question_only_infer_once(service, monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"response": "表格收入33500万元"})

    monkeypatch.setattr(vision.httpx, "post", post)
    for _ in range(2):
        assert service._generate_sync(b"table", "收入是多少？", timeout=1) == "表格收入33500万元"
    assert len(calls) == 1, "repeated identical VQA must not reload and run the model again"


@pytest.mark.parametrize("change", ["image", "prompt", "model", "context", "endpoint", "expiry"])
def test_changed_inputs_or_expiry_do_not_reuse_results(service, monkeypatch, change):
    calls = []
    clock = [100.0]
    monkeypatch.setattr(vision.time, "monotonic", lambda: clock[0])

    def post(url, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"response": "结果"})

    monkeypatch.setattr(vision.httpx, "post", post)
    service._generate_sync(b"table", "收入？", timeout=1)
    image, prompt = b"table", "收入？"
    settings = vision.get_settings()
    if change == "image":
        image = b"updated table"
    elif change == "prompt":
        prompt = "成本？"
    elif change == "model":
        monkeypatch.setattr(settings, "OLLAMA_VISION_MODEL", "other-model")
    elif change == "context":
        monkeypatch.setattr(settings, "OLLAMA_NUM_CTX", settings.chat_num_ctx + 1024)
    elif change == "endpoint":
        monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://other-server:11434")
    else:
        clock[0] += 300
    service._generate_sync(image, prompt, timeout=1)
    assert len(calls) == 2


def test_errors_and_empty_outputs_are_retried(service, monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError("model timed out")
        text = "" if len(calls) == 2 else "可用结果"
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"response": text})

    monkeypatch.setattr(vision.httpx, "post", post)
    assert service._generate_sync(b"table", "问题", timeout=1) is None
    assert service._generate_sync(b"table", "问题", timeout=1) is None
    assert service._generate_sync(b"table", "问题", timeout=1) == "可用结果"
    assert len(calls) == 3


def test_cache_evicts_old_inputs(service, monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"response": "结果"})

    monkeypatch.setattr(vision.httpx, "post", post)
    for index in range(33):
        service._generate_sync(str(index).encode(), "问题", timeout=1)
    service._generate_sync(b"0", "问题", timeout=1)
    assert len(calls) == 34


def test_oversized_response_is_returned_without_retaining_it(service, monkeypatch):
    calls = []
    text = "数" * 64001

    def post(url, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"response": text})

    monkeypatch.setattr(vision.httpx, "post", post)
    assert service._generate_sync(b"table", "问题", timeout=1) == text
    assert service._generate_sync(b"table", "问题", timeout=1) == text
    assert len(calls) == 2
    assert not service._results


def test_cached_vision_result_is_not_used_when_image_acl_is_missing(service, monkeypatch):
    import asyncio
    from app.services.nodes import multimodal_context_node as context
    from app.services.nodes import final_check_node
    from app.services.retrieval_service import RetrievedChunk
    from app.services.security_policy import ScopePredicate

    monkeypatch.setattr(vision.httpx, "post", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"response": "私有图片收入33500万元"}
    ))
    service._generate_sync(b"private-image", "问题", timeout=1)
    monkeypatch.setattr(context, "get_vision_service", lambda: service)

    def forbidden_read(*args):
        raise AssertionError("a denied image must not be loaded or looked up in the vision cache")

    async def no_audit(*args, **kwargs):
        pass

    monkeypatch.setattr(context, "_load_image_bytes", forbidden_read)
    monkeypatch.setattr(final_check_node, "audit_acl_drops", no_audit)
    chunk = RetrievedChunk("doc", "private.pdf", 1, 0, "33500", 0.9,
                           content_type="image", image_id="image", image_path="image.png")
    result = asyncio.run(context.build_multimodal_context(
        [chunk], "问题", pred=ScopePredicate(user_id="other", tenant_ids=frozenset()),
        view_index={}, materialized=True,
    ))
    assert result.sources == []
    assert "33500" not in result.context
    assert result.dropped_count == 1
