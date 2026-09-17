"""异步入库契约单测：受理即返回 · 判重 · 并发复用 · 续传 · 任务注册表.

背景：`POST /upload` 从"同步跑完整条管线"改成"受理即返回 202，管线丢后台"。
这个改动最容易踩的不是功能，而是**契约**——判重是否仍然生效、同一文件被并发
提交会不会起两个任务抢 CPU、残档能不能幂等续传、后台任务跑完是否从注册表摘掉。
这些都不需要数据库或向量库，把外部依赖换成桩即可精确验证。

不启动后端、不连 DB、不加载 torch / docling / paddle：

    python tests/test_async_ingestion.py          # 直接跑
    python -m pytest tests/test_async_ingestion.py -q
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


async def _noop(*_args, **_kwargs):
    return None


# 需要打桩的重依赖。**桩只在导入被测模块期间存在**，导入完立即还原
# sys.modules —— 否则这些假模块会污染同一个 pytest 会话里后收集的测试文件：
# 实测会让 test_image_pipeline.py 报
# `cannot import name 'contains_markdown_table' from 'app.services.chunker'
# (unknown location)`，而那个文件又用 sys.exit(0) 跳过，最终整套测试崩掉。
_STUB_MODULES = (
    "app.services",
    "app.services.audit_service",
    "app.services.chunker",
    "app.services.embedding_service",
    "app.services.parsers",
    "app.services.prompt_security",
    "app.services.storage",
    "app.services.vector_service",
    "app.services.document_service",
)


def _install_stubs() -> dict[str, object]:
    """装上桩并返回 sys.modules 的原状，供 `_restore_stubs` 还原。"""
    saved = {name: sys.modules.get(name) for name in _STUB_MODULES}

    # 让 `app.services.<x>` 能单独加载而不触发 app/services/__init__.py 的重依赖
    # （同 tests/test_evidence_gate.py 的做法）。
    pkg = types.ModuleType("app.services")
    pkg.__path__ = [str(_BACKEND_ROOT / "app" / "services")]
    sys.modules["app.services"] = pkg

    def _stub(name: str, **attrs) -> None:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod

    _stub("app.services.audit_service", record_audit=_noop)
    # ⚠️ 桩必须与 document_service 从 chunker 导入的名字**一一对应**。
    # 落下了会在 import 处报 "cannot import name 'X' from 'app.services.chunker'
    # (unknown location)" —— 报错指向 chunker 而真正的原因在测试桩，很容易查偏。
    # 本用例只用 prepare_upload（上传前段），不跑分块，因此返回空结构即可。
    _stub(
        "app.services.chunker",
        build_chunks=lambda **_kw: [],
        build_chunk_hierarchy=lambda **_kw: types.SimpleNamespace(
            children=[], parents=[]
        ),
        build_image_chunks=lambda *_a, **_kw: [],
    )
    _stub("app.services.embedding_service", embed_batch_with_retry=_noop)
    _stub("app.services.parsers", get_parser_for_file=lambda _name: None)
    # 桩要覆盖 document_service 的**全部**间接依赖，不能只给它直接 import 的那个
    # 名字：清洗走 text_cleaning，而 text_cleaning 延迟 import
    # ``mask_instruction_paragraphs``。漏掉它，本用例只因"假页缺 char_start、
    # clean_extraction 提前返回"而侥幸通过 —— 一旦夹具补齐页码字段就会在
    # import 处炸出 ImportError，报错位置离真正原因很远。
    _stub(
        "app.services.prompt_security",
        scan_document_text=lambda _t: None,
        mask_instruction_paragraphs=lambda t: (t, 0, ()),
    )
    _stub("app.services.storage", save_original=lambda *_a, **_kw: None)
    _stub(
        "app.services.vector_service",
        VectorPoint=object,
        upsert_vectors=_noop,
        generate_point_id=lambda *_a, **_kw: "pid",
        get_existing_point_ids=_noop,
    )
    return saved


def _restore_stubs(saved: dict[str, object]) -> None:
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


_saved_modules = _install_stubs()
try:
    _SPEC = importlib.util.spec_from_file_location(
        "app.services.document_service",
        _BACKEND_ROOT / "app" / "services" / "document_service.py",
    )
    ds = importlib.util.module_from_spec(_SPEC)
    sys.modules["app.services.document_service"] = ds
    _SPEC.loader.exec_module(ds)
finally:
    # 还原：本用例已拿到 ds 的引用，桩不再需要对外可见
    _restore_stubs(_saved_modules)

DocumentStatus = ds.DocumentStatus
NOW = datetime.now(tz=timezone.utc)


# ── 假文档行 / 假会话 ────────────────────────────────────────────────────────


class FakeDoc:
    """只需要 prepare_upload 会读到的字段。"""

    def __init__(self, status, **overrides):
        self.id = overrides.get("id", uuid.uuid4())
        self.status = status
        self.tenant_id = overrides.get("tenant_id", "default")
        self.department_id = overrides.get("department_id")
        self.access_level = overrides.get("access_level", "private")
        self.file_size = overrides.get("file_size", 123)
        self.page_count = overrides.get("page_count", 3)
        self.chunk_count = overrides.get("chunk_count", 9)
        self.image_count = overrides.get("image_count", 1)
        self.image_object_count = overrides.get("image_object_count", 1)
        self.created_at = overrides.get("created_at", NOW)


class FakeSession:
    def __init__(self, doc):
        self._doc = doc

    async def scalar(self, _query):
        return self._doc


class FakeSessionCtx:
    def __init__(self, doc):
        self._doc = doc

    async def __aenter__(self):
        return FakeSession(self._doc)

    async def __aexit__(self, *_exc):
        return False


def _patch_session(monkeypatch, doc):
    monkeypatch.setattr(ds, "get_db_session", lambda: FakeSessionCtx(doc))


async def _wait_until_idle(doc_id, timeout=3.0):
    elapsed = 0.0
    while ds.is_ingesting(doc_id) and elapsed < timeout:
        await asyncio.sleep(0.02)
        elapsed += 0.02


# ── 判重：已完成的文件不再重复入库 ───────────────────────────────────────────


def test_completed_document_returns_already_exists(monkeypatch):
    doc = FakeDoc(DocumentStatus.COMPLETED)
    _patch_session(monkeypatch, doc)

    async def _inner():
        prep = await ds.prepare_upload("a.pdf", b"data")
        assert prep.done is True, "已存在时必须直接给终态，不回文档行"
        assert prep.result is not None
        assert prep.result.status == DocumentStatus.ALREADY_EXISTS
        assert prep.result.document_id == doc.id
        assert prep.result.existing_document_id == doc.id
        assert prep.result.chunk_count == doc.chunk_count

    asyncio.run(_inner())


# ── 并发复用：同文件正在入库时不重复起任务 ──────────────────────────────────


def test_running_job_is_reused(monkeypatch):
    doc = FakeDoc(DocumentStatus.PARSING)
    _patch_session(monkeypatch, doc)
    monkeypatch.setattr(ds, "is_ingesting", lambda _id: True)

    async def _inner():
        prep = await ds.prepare_upload("a.pdf", b"data")
        assert prep.done is True, "已在入库中的文件不应再起一个任务"
        assert prep.result is not None
        assert prep.result.status == DocumentStatus.PARSING
        assert "正在处理" in (prep.result.message or "")

    asyncio.run(_inner())


# ── 续传：上次失败的残档复用原行，且沿用原隔离属性 ──────────────────────────


def test_failed_document_resumes_in_place(monkeypatch):
    doc = FakeDoc(
        DocumentStatus.FAILED,
        tenant_id="company_a",
        department_id="dept-1",
        access_level="department",
    )
    _patch_session(monkeypatch, doc)
    monkeypatch.setattr(ds, "is_ingesting", lambda _id: False)

    # 记录续传时对文档行做的 UPDATE
    updates: list[dict] = []

    async def _record(_doc_id, **kwargs):
        updates.append(kwargs)

    monkeypatch.setattr(ds, "_update_document", _record)

    async def _inner():
        prep = await ds.prepare_upload(
            "a.pdf", b"data",
            owner_id=uuid.uuid4(),
            tenant_id="company_b",            # 本次上传者来自另一租户
            access_level="private",
        )
        assert prep.done is False
        assert prep.is_resume is True
        assert prep.doc is doc, "续传必须复用原行，不能新建"
        # 防越权接管：续传不随本次上传者改变隔离属性
        assert prep.tenant_id == "company_a"
        assert prep.department_id == "dept-1"
        assert prep.access_level == "department"

        # 受理的那一刻就必须翻回 PENDING 并清掉上一轮的残留计数/报错。
        # 否则接口刚回 202"正在后台建立索引"，用户刷新却看到"失败 + 旧报错"。
        assert updates, "续传必须在请求线程里就把行翻成 PENDING"
        patch = updates[0]
        assert patch.get("status") == DocumentStatus.PENDING
        assert patch.get("current_stage") == "pending"
        assert patch.get("failed_chunks") == 0
        assert patch.get("embedded_chunks") == 0
        assert patch.get("total_chunks") == 0
        assert patch.get("error_message") is None, "旧报错必须清掉"

    asyncio.run(_inner())


# ── 新建：无同名文件时落 PENDING 行 ─────────────────────────────────────────


def test_new_document_is_created_as_pending(monkeypatch):
    _patch_session(monkeypatch, None)
    created = {}

    async def _fake_create(filename, file_size, file_hash, owner_id=None,
                           collection_id=None, tenant_id="default",
                           department_id=None, access_level="private"):
        created.update(
            filename=filename, file_size=file_size, tenant_id=tenant_id,
            access_level=access_level,
        )
        return FakeDoc(DocumentStatus.PENDING, tenant_id=tenant_id)

    monkeypatch.setattr(ds, "_create_document_record", _fake_create)

    async def _inner():
        prep = await ds.prepare_upload(
            "new.pdf", b"x" * 42, tenant_id="default", access_level="private",
        )
        assert prep.done is False
        assert prep.doc is not None
        assert prep.file_size == 42
        # 指纹要基于内容，判重才有意义
        assert prep.file_hash == __import__("hashlib").sha256(b"x" * 42).hexdigest()
        assert created["filename"] == "new.pdf"

    asyncio.run(_inner())


# ── 核心契约：受理立即返回，不等管线 ────────────────────────────────────────


def test_schedule_ingestion_returns_pending_without_waiting(monkeypatch):
    gate_started = asyncio.Event()
    gate_release = asyncio.Event()
    finished = {"value": False}

    async def _slow_run(prep, filename, content, settings,
                        owner_id=None, collection_id=None):
        gate_started.set()
        await gate_release.wait()
        finished["value"] = True
        return ds.DocumentResult(
            document_id=prep.doc.id, filename=filename,
            status=DocumentStatus.COMPLETED,
        )

    monkeypatch.setattr(ds, "_run_ingestion", _slow_run)

    async def _inner():
        doc = FakeDoc(DocumentStatus.PENDING)
        prep = ds.PreparedUpload(doc=doc, tenant_id="default", file_size=7)

        # 关键断言：管线还被 gate 卡着，受理就已经返回了 PENDING
        result = ds.schedule_ingestion(prep, "a.pdf", b"1234567")
        assert result.status == DocumentStatus.PENDING
        assert result.document_id == doc.id
        assert "后台" in (result.message or "")
        assert finished["value"] is False, "受理不得等待管线跑完"

        await asyncio.wait_for(gate_started.wait(), timeout=3)
        assert ds.is_ingesting(doc.id) is True

        gate_release.set()
        await asyncio.wait_for(_wait_until_idle(doc.id), timeout=3)
        assert finished["value"] is True
        assert ds.is_ingesting(doc.id) is False, "任务结束后必须从注册表摘除"

    asyncio.run(_inner())


def test_sync_path_still_runs_pipeline_inline(monkeypatch):
    """`process_uploads` / `_process_single_file` 的同步语义不能被破坏（脚本在用）."""
    calls = []

    async def _fake_run(prep, filename, content, settings,
                        owner_id=None, collection_id=None):
        calls.append(filename)
        return ds.DocumentResult(
            document_id=prep.doc.id, filename=filename,
            status=DocumentStatus.COMPLETED,
        )

    monkeypatch.setattr(ds, "_run_ingestion", _fake_run)
    monkeypatch.setattr(ds, "is_ingesting", lambda _id: False)
    _patch_session(monkeypatch, None)

    async def _fake_create(filename, file_size, file_hash, owner_id=None,
                           collection_id=None, tenant_id="default",
                           department_id=None, access_level="private"):
        return FakeDoc(DocumentStatus.PENDING, tenant_id=tenant_id)

    monkeypatch.setattr(ds, "_create_document_record", _fake_create)

    async def _inner():
        resp = await ds.process_uploads([("a.txt", b"aa"), ("b.txt", b"bb")])
        assert [r.status for r in resp.documents] == [DocumentStatus.COMPLETED] * 2
        assert resp.succeeded == 2 and resp.failed == 0
        assert calls == ["a.txt", "b.txt"], "同步路径应逐份跑完管线后才返回"

    asyncio.run(_inner())


if __name__ == "__main__":
    # 允许 `python tests/test_async_ingestion.py` 直接跑（不依赖 pytest）
    import traceback

    class _MiniPatch:
        """最小 monkeypatch：逐个用例撤销，避免桩在用例之间串味。"""

        def __init__(self):
            self._undo: list[tuple[object, str, object]] = []

        def setattr(self, obj, name, value):
            self._undo.append((obj, name, getattr(obj, name, None)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, old in reversed(self._undo):
                setattr(obj, name, old)
            self._undo.clear()

    failures = 0
    for _name in sorted(n for n in list(globals()) if n.startswith("test_")):
        _fn = globals()[_name]
        _mp = _MiniPatch()
        try:
            _fn(_mp)
            print(f"PASS {_name}")
        except Exception:                                     # noqa: BLE001
            failures += 1
            print(f"FAIL {_name}")
            traceback.print_exc()
        finally:
            _mp.undo()
    print("\n" + ("ALL PASSED" if failures == 0 else f"{failures} FAILED"))
    sys.exit(1 if failures else 0)
