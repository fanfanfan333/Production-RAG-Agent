"""
上传链路冒烟测试（需要运行中的服务 + test.pdf 夹具）.

这是一条**集成冒烟**用例：它真的走一次 HTTP 上传并检查落库结果，因此依赖
两样东西 —— 监听中的后端、以及 CWD 下存在的 test.pdf。

两者缺一就 skip，而不是让整套件在**收集阶段**就 FileNotFoundError 中断。
历史上它在模块顶层直接 open()，一旦夹具不在场，pytest 会在收集时炸掉，
导致同目录下的所有用例一条都跑不了 —— 一个"环境没准备好"的问题被放大成
"整个测试套件不可用"，这是集成用例最不该有的性质。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

PDF_PATH = "test.pdf"
URL = os.getenv("UPLOAD_TEST_URL", "http://localhost:8000/upload")
BOUNDARY = "----FormBoundary7MA4YWxkTrZu0gW"


def _service_up(url: str) -> bool:
    try:
        base = url.rsplit("/upload", 1)[0] + "/health"
        with urllib.request.urlopen(base, timeout=5) as resp:
            return resp.status == 200
    except Exception:      # noqa: BLE001
        return False


@pytest.mark.skipif(
    not Path(PDF_PATH).is_file(),
    reason=f"fixture {PDF_PATH} not present in CWD — run from the backend root",
)
def test_upload_pipeline_end_to_end() -> None:
    """上传 PDF → 受理 → 校验返回的受理结果.

    契约说明：`POST /upload` 是**异步受理**接口（202），只做校验 + 判重 + 落
    一条 PENDING 行，随后解析/OCR/向量化在服务端后台跑。所以这里校验的是
    "受理成功"，而不是"入库完成" —— 后者要去 `GET /documents` 看
    status/current_stage（该端点需要鉴权，本冒烟用例不带令牌）。
    """
    if not _service_up(URL):
        pytest.skip(f"backend not reachable at {URL}")

    pdf_bytes = Path(PDF_PATH).read_bytes()
    boundary_bytes = BOUNDARY.encode()
    body = (
        b"--" + boundary_bytes + b"\r\n"
        b'Content-Disposition: form-data; name="files"; filename="test_upload.pdf"\r\n'
        b"Content-Type: application/pdf\r\n"
        b"\r\n"
        + pdf_bytes
        + b"\r\n"
        b"--" + boundary_bytes + b"--\r\n"
    )

    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
        method="POST",
    )

    print(f"Uploading {len(pdf_bytes):,} byte PDF to {URL} ...")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        pytest.fail(f"HTTP {exc.code}: {exc.read().decode()}")
    except Exception as exc:      # noqa: BLE001
        pytest.fail(f"upload request failed: {exc}")

    documents = result.get("documents") or []
    assert documents, f"no document returned: {result}"
    doc = documents[0]

    print("\n=== Accept Verification ===")
    for field in ("status", "document_id", "message"):
        print(f"  {field:12s}: {doc.get(field)}")

    assert doc.get("document_id"), f"no document_id in accept response: {doc}"
    assert doc.get("status") in {"pending", "already_exists"}, (
        f"unexpected accept status {doc.get('status')!r}, error: {doc.get('error')}"
    )
    print("\nALL CHECKS PASSED - upload accept path is working correctly")
