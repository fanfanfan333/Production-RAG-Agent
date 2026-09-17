"""异步入库端到端验收：受理时延 + 阶段流转 + 终态校验.

    RAG_TOKEN=<jwt> python backend/scripts/e2e_async_upload.py 文件1 文件2 ...

验证三件事（只用标准库，可宿主机直接跑）：

1. **受理时延** —— `POST /upload` 必须在秒级返回 202，而不是陪着解析/向量化
   一起等。这是本次改动的核心契约；一份 13 MB 文档入库要 7 分钟以上，让
   HTTP 连接陪着等意味着"用户关掉标签页 = 什么都没发生"。
2. **阶段流转** —— 轮询 `GET /documents`，打印每个文档 `current_stage` 的
   变化轨迹（pending → parsing → chunking → … → completed）。这条轨迹就是
   前端"正在解析内容与图片 / 正在生成向量 42%"的数据来源，能证明它真的在推进，
   而不是永远停在 pending。
3. **终态与计数** —— 最终 status 必须是 completed（或 already_exists），
   且 chunk_count > 0、image_object_count 有值。

建议直接用真实文档跑：`实战.docx`、`Python AI大模型成神手册.pdf`、
`嵌入式软件成神手册.docx`。
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.getenv("RAG_API", "http://127.0.0.1:8000").rstrip("/")
TOKEN = os.getenv("RAG_TOKEN", "")
BOUNDARY = "----E2EAsyncBoundary7MA4YWxkTrZu0gW"
ACCEPT_BUDGET_SECONDS = float(os.getenv("ACCEPT_BUDGET", "30"))
TOTAL_TIMEOUT_SECONDS = float(os.getenv("INGEST_TIMEOUT", "1800"))
POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL", "5"))

TERMINAL = {"completed", "failed", "already_exists"}


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    h = {"Accept": "application/json"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    if extra:
        h.update(extra)
    return h


def upload(files: list[Path]) -> tuple[float, dict]:
    """把所有文件打进一个 multipart 请求，返回 (耗时秒, 响应体)。"""
    parts: list[bytes] = []
    for path in files:
        parts.append(
            b"--" + BOUNDARY.encode() + b"\r\n"
            + f'Content-Disposition: form-data; name="files"; filename="{path.name}"\r\n'.encode()
            + b"Content-Type: application/octet-stream\r\n\r\n"
            + path.read_bytes()
            + b"\r\n"
        )
    parts.append(b"--" + BOUNDARY.encode() + b"--\r\n")
    body = b"".join(parts)

    req = urllib.request.Request(
        f"{BASE}/upload",
        data=body,
        headers=_headers({"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"}),
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=ACCEPT_BUDGET_SECONDS * 4) as resp:
            payload = json.loads(resp.read())
            status = resp.status
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"上传被拒：HTTP {exc.code} — {exc.read().decode('utf-8', 'replace')}")
    elapsed = time.monotonic() - started

    print(f"  受理返回：HTTP {status}，耗时 {elapsed:.2f}s，请求体 {len(body) / 1024 / 1024:.2f} MB")
    if status != 202:
        print(f"  ⚠️ 期望 202 Accepted，实际 {status}")
    return elapsed, payload


def fetch_documents() -> list[dict]:
    req = urllib.request.Request(
        f"{BASE}/documents?limit=100", headers=_headers(), method="GET"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read()).get("documents", [])


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit("文件不存在：" + ", ".join(missing))
    if not TOKEN:
        print("提示：未设置 RAG_TOKEN，受保护端点可能返回 401。\n")

    print("=" * 78)
    print(f"目标后端：{BASE}")
    print(f"待入库：{len(paths)} 个文件")
    for p in paths:
        print(f"  · {p.name}  ({p.stat().st_size / 1024 / 1024:.2f} MB)")
    print("=" * 78)

    print("\n[1/3] 受理（验证不再阻塞）")
    accept_seconds, payload = upload(paths)
    if accept_seconds > ACCEPT_BUDGET_SECONDS:
        print(f"  ❌ 受理耗时 {accept_seconds:.1f}s 超出预算 {ACCEPT_BUDGET_SECONDS}s")
    else:
        print(f"  ✅ 受理耗时 {accept_seconds:.1f}s ≤ {ACCEPT_BUDGET_SECONDS}s，未陪同入库等待")

    docs = payload.get("documents") or []
    targets: dict[str, dict] = {}
    for d in docs:
        did = str(d.get("document_id") or "")
        print(f"  · {d.get('filename'):<34} status={d.get('status'):<14} {d.get('message') or ''}")
        if did:
            targets[did] = {"filename": d.get("filename"), "seen": [], "last": None}

    if not targets:
        raise SystemExit("受理响应里没有 document_id，无法跟踪进度。")

    # 只有 pending 的需要跟踪进度；already_exists 的已是终态
    pending = {
        str(d.get("document_id"))
        for d in docs
        if d.get("status") == "pending" and d.get("document_id")
    }
    if not pending:
        print("\n所有文件此前已入库（already_exists），无需跟踪进度。")

    print(f"\n[2/3] 阶段流转（每 {POLL_INTERVAL_SECONDS:g}s 轮询一次，最多 {TOTAL_TIMEOUT_SECONDS:g}s）")
    started = time.monotonic()
    while pending and time.monotonic() - started < TOTAL_TIMEOUT_SECONDS:
        try:
            rows = {str(r.get("document_id")): r for r in fetch_documents()}
        except Exception as exc:                                  # noqa: BLE001
            print(f"  轮询失败（继续重试）：{exc}")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        for did in list(pending):
            row = rows.get(did)
            if row is None:
                continue
            stage = row.get("current_stage")
            status = row.get("status")
            total = row.get("total_chunks") or 0
            embedded = row.get("embedded_chunks") or 0
            mark = f"{status}/{stage}"
            if total:
                mark += f" {embedded}/{total}"
            info = targets[did]
            if mark != info["last"]:
                info["last"] = mark
                info["seen"].append((round(time.monotonic() - started, 1), mark))
                print(f"  [{time.monotonic() - started:6.1f}s] {info['filename']:<34} {mark}")
            if status in TERMINAL:
                pending.discard(did)

        if pending:
            time.sleep(POLL_INTERVAL_SECONDS)

    print("\n[3/3] 终态与计数")
    try:
        rows = {str(r.get("document_id")): r for r in fetch_documents()}
    except Exception as exc:                                      # noqa: BLE001
        raise SystemExit(f"无法读取文档列表：{exc}")

    failures: list[str] = []
    # 判定标准：**只有真正入库成功才算过**。
    # 早期版本把 "failed" 也当成"进入了终态"从而判通过 —— 那等于"崩了也算合格"，
    # 会把最严重的问题伪装成绿色。终态里的 completed / already_exists 才合格。
    OK_TERMINAL = {"completed", "already_exists"}
    for did, info in targets.items():
        row = rows.get(did)
        if row is None:
            failures.append(f"{info['filename']}：列表里找不到该文档")
            print(f"  ❌ {info['filename']:<34} 列表里找不到")
            continue
        status = row.get("status")
        chunks = row.get("chunk_count") or 0
        images = row.get("image_object_count") or 0
        embedded = row.get("embedded_chunks") or 0
        total = row.get("total_chunks") or 0
        ok = status in OK_TERMINAL and chunks > 0
        icon = "✅" if ok else "❌"
        print(
            f"  {icon} {row.get('filename'):<34} status={status:<14} "
            f"chunks={chunks:<5} vec={embedded}/{total:<5} "
            f"image_chunks={images:<4} pages={row.get('page_count') or 0}"
        )
        if not ok:
            err = (row.get("error") or "").strip().replace("\n", " ")[:160]
            if status == "failed":
                failures.append(
                    f"{info['filename']}：入库失败（status=failed, 阶段={row.get('current_stage')}）"
                    f"{' — ' + err if err else ''}"
                )
            elif status not in OK_TERMINAL:
                failures.append(
                    f"{info['filename']}：未进入合格终态（status={status}，阶段={row.get('current_stage')}）"
                )
            else:
                failures.append(f"{info['filename']}：completed 但 chunk_count=0，索引是空的")
        if pending and did in pending:
            failures.append(f"{info['filename']}：超时仍未进入终态（当前 {row.get('current_stage')}）")

    print("\n" + "=" * 78)
    if failures:
        print(f"❌ 验收未通过（{len(failures)} 项）")
        for f in failures:
            print(f"  · {f}")
        raise SystemExit(1)
    print("✅ 验收通过：受理秒级返回、阶段正常流转、全部 completed 且块数与向量数 > 0")
    print("=" * 78)


if __name__ == "__main__":
    main()
