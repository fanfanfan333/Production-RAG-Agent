"""
上传链路冒烟测试（需要运行中的服务 + test.pdf 夹具）.

这是一条**集成冒烟**用例：它真的走一次 HTTP 上传并检查受理结果，因此依赖
三样东西 —— 监听中的后端、CWD 下存在的 test.pdf、以及一个可用的访问令牌。

任何一样缺了就 skip，而不是让整套件在**收集阶段**就 FileNotFoundError 中断。
历史上它在模块顶层直接 open()，一旦夹具不在场，pytest 会在收集时炸掉，
导致同目录下的所有用例一条都跑不了 —— 一个"环境没准备好"的问题被放大成
"整个测试套件不可用"，这是集成用例最不该有的性质。

⚠️ 鉴权（2026-09 修正）
----------------------
本用例原先假定 ``POST /upload`` 是公开端点，不带令牌直接上传。但 upload 路由
带 ``Depends(require_permission("document.write"))``（见 ``app/api/documents.py``，
``app/main.py`` 亦标注 "Phase 2 — POST /upload (auth)"）—— 鉴权落地之后这个假定
就过期了，实测拿到 ``HTTP 401 未登录或缺少访问令牌``。

修法不是"忽略 401"（那会把真实的鉴权回归一起吞掉），而是**把令牌补上**：

    优先级 1  UPLOAD_TEST_TOKEN                  显式令牌
    优先级 2  UPLOAD_TEST_USER + _PASSWORD       自动 POST /auth/login 换令牌
    都没有                                       skip（环境没配好，不是缺陷）

反方向的边界同样守住：**已经带上令牌仍被拒** ⇒ 判**失败**并点明这是鉴权/权限
缺陷，绝不静默跳过。
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

#: 显式令牌（最高优先级）
ENV_TOKEN = os.getenv("UPLOAD_TEST_TOKEN", "").strip()
#: 备选：用账号密码自动换令牌
ENV_USER = os.getenv("UPLOAD_TEST_USER", "").strip()
ENV_PASSWORD = os.getenv("UPLOAD_TEST_PASSWORD", "")


def _service_up(url: str) -> bool:
    try:
        base = url.rsplit("/upload", 1)[0] + "/health"
        with urllib.request.urlopen(base, timeout=5) as resp:
            return resp.status == 200
    except Exception:      # noqa: BLE001
        return False


def _login(base: str) -> str:
    """用 UPLOAD_TEST_USER / UPLOAD_TEST_PASSWORD 换一个 JWT.

    登录失败**不吞**：配了凭据却登不上属于环境/账号问题，需要人看，
    不该被静默跳过（调用方会把它当失败报出来）。
    """
    body = json.dumps({"username": ENV_USER, "password": ENV_PASSWORD}).encode()
    req = urllib.request.Request(
        f"{base}/auth/login",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["access_token"]


def _resolve_token(base: str) -> tuple[str | None, str]:
    """返回 (令牌, 来源)。令牌为 None ⇒ 没配置任何凭据，调用方据此 skip。"""
    if ENV_TOKEN:
        return ENV_TOKEN, "UPLOAD_TEST_TOKEN"
    if ENV_USER and ENV_PASSWORD:
        return _login(base), f"POST /auth/login (user={ENV_USER})"
    return None, ""


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

    base = URL.rsplit("/upload", 1)[0]
    token, token_source = _resolve_token(base)
    if not token:
        pytest.skip(
            "POST /upload 要求鉴权（require_permission('document.write')），"
            "本冒烟用例必须带令牌 —— 请设置 UPLOAD_TEST_TOKEN，"
            "或同时设置 UPLOAD_TEST_USER / UPLOAD_TEST_PASSWORD 后重跑"
        )

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
        headers={
            "Content-Type": f"multipart/form-data; boundary={BOUNDARY}",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )

    print(f"Uploading {len(pdf_bytes):,} byte PDF to {URL} (token from {token_source}) ...")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        if exc.code in (401, 403):
            # 已经带上令牌仍被拒 —— 这是鉴权/权限缺陷，不是"环境没配好"。
            pytest.fail(
                f"已带令牌仍被拒（HTTP {exc.code}）：{detail}"
                f"［令牌来源：{token_source}］—— 鉴权/权限缺陷，需人工核查"
            )
        pytest.fail(f"HTTP {exc.code}: {detail}")
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
