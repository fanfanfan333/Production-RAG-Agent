import ast
from pathlib import Path

import yaml

from app.services.rag_graph import _friendly_error


def test_local_profile_preserves_the_rag_evidence_window():
    profile = yaml.safe_load(
        (Path(__file__).parents[1] / "docker-compose.low-memory.yml").read_text(encoding="utf-8")
    )
    environment = profile["services"]["backend"]["environment"]
    assert int(environment["OLLAMA_NUM_CTX"]) >= 8192
    assert int(environment["MAX_CONCURRENT_EMBEDDINGS"]) == 1
    assert environment["OLLAMA_KEEP_ALIVE_SECONDS"] == "${OLLAMA_KEEP_ALIVE_SECONDS:-0}"
    assert environment["HF_HUB_OFFLINE"] == "1"


def test_context_overflow_is_explained_without_raw_server_json():
    message = _friendly_error(RuntimeError(
        '{"error":{"code":400,"message":"request (3969 tokens) exceeds the available '
        'context size (2048 tokens)","type":"exceed_context_size_error"}}'
    ))
    assert "上下文容量" in message
    assert "OLLAMA_NUM_CTX" in message
    assert '"code"' not in message


def test_all_ollama_factories_apply_the_model_residency_setting():
    root = Path(__file__).parents[1] / "app" / "services"
    count = 0
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ChatOllama":
                keywords = {kw.arg: kw.value for kw in node.keywords}
                assert "keep_alive" in keywords, path
                assert "OLLAMA_KEEP_ALIVE_SECONDS" in ast.unparse(keywords["keep_alive"]), path
                count += 1
    assert count == 7


def test_vision_request_applies_the_model_residency_setting(monkeypatch):
    from types import SimpleNamespace
    from app.services.vision import vision_service as vision

    monkeypatch.setattr(vision.get_settings(), "OLLAMA_KEEP_ALIVE_SECONDS", 0)
    requests = []

    def post(url, *, json, timeout):
        requests.append(json)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"response": "图片描述"})

    monkeypatch.setattr(vision.httpx, "post", post)
    assert vision.VisionService()._generate_sync(b"image", "prompt", timeout=1) == "图片描述"
    assert requests[0]["keep_alive"] == 0
