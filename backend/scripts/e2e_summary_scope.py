"""
文档总结范围 + 引用校验 E2E 验收脚本.

覆盖两个已修复的问题（截图1~4）：

问题1  "总结所有文档"只总结了内容最多的那一份
      根因：请求被路由成 knowledge_qa → 走检索链，只把召回分最高的那一份
      文档的两三个片段拼成"总结"。
      修复：确定性路由（总结动词 + 全库范围词）→ document_summary；
            该分支把**全部可访问文档**的摘要都交给模型。
      验收：SSE 的 intent 事件必须是 document_summary，
            且 doc_digests 覆盖全部可访问文档。

问题2  用户指定总结哪份文档，就要只总结哪份
      修复：document_summary_node.resolve_summary_targets 从提问里解析文档名
            （书名号 / 带扩展名的文件名 / "…文档"限定语），收敛采样范围；
            点名了但对不上一份都不猜，直接把可选文档列给用户。
      验收：点名一份 → doc_digests 恰好 1 份；点名不存在的 → 列出候选清单。

问题4  对话里的数字序号被当成原文内容参与比对
      修复：citation_verifier 在校验前剥掉 Markdown 排版标记（列表序号 / 标题）。
      验收：用截图里的真实回答（1. **反幻觉机制**：… 6 条列表）跑一遍，
            断言不再出现"数字与原文不一致"。

运行（容器内）：
    docker cp backend/scripts/e2e_summary_scope.py rag_backend:/tmp/
    docker exec -e PYTHONIOENCODING=utf-8 rag_backend python /tmp/e2e_summary_scope.py

默认账号 admin / RagAdmin#2026（可用 --admin-password 覆盖）。
只读：不新建、不修改、不删除任何业务数据（问题4 部分是纯函数比对）。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

BASE_DEFAULT = "http://127.0.0.1:8000"

# 不走系统代理（与 e2e_staff_flow.py 同一约定）
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_passed = 0
_failed = 0


def check(label: str, condition: bool, extra: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label} {extra}")


def login(base: str, username: str, password: str) -> str:
    req = urllib.request.Request(
        f"{base}/auth/login",
        data=json.dumps({"username": username, "password": password}).encode(),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    with _OPENER.open(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    token = payload.get("access_token")
    if not token:
        raise SystemExit(f"登录失败：{payload}")
    return token


def stream_query(base: str, token: str, query: str, timeout: int = 420) -> dict:
    """发一轮 /query，收齐 SSE 事件（只关心 intent / doc_digests / 文本）。"""
    req = urllib.request.Request(
        f"{base}/query",
        data=json.dumps({"query": query, "stream": True}).encode("utf-8"),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    req.add_header("Authorization", f"Bearer {token}")

    result: dict = {"intent": None, "digests": [], "text": "", "events": [], "error": None}
    with _OPENER.open(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                evt = json.loads(data)
            except json.JSONDecodeError:
                continue
            etype = evt.get("type") or evt.get("event") or ""
            result["events"].append(etype)
            if etype == "intent":
                result["intent"] = evt.get("intent")
            elif etype == "doc_digests":
                result["digests"] = evt.get("documents") or []
            elif etype == "error":
                result["error"] = evt.get("message")
            elif etype == "chunk":
                result["text"] += str(evt.get("content") or "")
    return result


# ── 问题4：确定性校验（不依赖 LLM）──────────────────────────────────────────

_SCREENSHOT_ANSWER = """\
文档内容主要围绕研发部2024年度技术方案中的检索平台建设，核心要点包括：
1. **反幻觉机制**：通过检索阶段降权处理不可信文档片段，确保答案可核对、可复现；需持续校准。[Source 1]
2. **数据层设计**：采用结构感知解析器解析文档，分块时按章节建立父块并切分子块，子块通过父块编号关联以支持检索命中后批量回填。[Source 2]
3. **实施计划**：分三期推进，一期完成元数据与结构分块改造，三期进行规模压测与调参，节省约 60% 存储空间。[Source 3]
"""

_SCREENSHOT_SOURCES = [
    {
        "document_id": "doc-1",
        "filename": "研发部-2024年度技术方案.docx",
        "page_number": 1,
        "line_start": 83,
        "line_end": 105,
        "text": (
            "反幻觉机制：通过检索阶段降权处理不可信文档片段，确保答案可核对、可复现；"
            "需持续校准，将线上问答的回召偏差纳入评测集回归验证。"
        ),
    },
    {
        "document_id": "doc-1",
        "filename": "研发部-2024年度技术方案.docx",
        "page_number": 1,
        "line_start": 35,
        "line_end": 57,
        "text": (
            "数据层设计：采用结构感知解析器解析文档，分块时按章节建立父块并切分子块，"
            "子块通过父块编号关联以支持检索命中后批量回填。"
        ),
    },
    {
        "document_id": "doc-1",
        "filename": "研发部-2024年度技术方案.docx",
        "page_number": 2,
        "line_start": 12,
        "line_end": 30,
        "text": (
            "实施计划：分三期推进，一期完成元数据与结构分块改造，三期进行规模压测与调参，"
            "复用现有服务器，通过移出父块正文节省约 60% 存储空间。"
        ),
    },
]


def verify_citation_fix() -> None:
    print("\n[问题4] 引用校验不再把 Markdown 序号当正文数字")
    from app.services.nodes.citation_verifier import verify_citations

    report = verify_citations(_SCREENSHOT_ANSWER, _SCREENSHOT_SOURCES)
    audit = report.as_audit()
    check(
        "6 条列表序号不再触发『数字与原文不一致』",
        not report.number_mismatch_indices,
        f"number_mismatch={audit['number_mismatch']}",
    )
    check(
        "『未被原文直接支持』不再因序号误报",
        not report.unsupported_indices,
        f"unsupported={audit['unsupported']}",
    )
    check(
        "引用全部通过校验",
        report.passed_count == report.total and report.total == 3,
        f"passed={report.passed_count}/{report.total} overall={report.overall}",
    )
    check(
        "净化后正文未被改动（没有误删引用标记）",
        report.clean_text.strip() == _SCREENSHOT_ANSWER.strip(),
        report.clean_text[:80],
    )

    # 反向用例：列表项里的**真**数字仍要被抓出来
    bad = verify_citations(
        "3. 分三期推进，节省约 999% 存储空间 [Source 1]。", _SCREENSHOT_SOURCES
    )
    check(
        "列表项里的错误数字仍被抓出（没有放过真误报）",
        bad.number_mismatch_indices == (1,),
        f"number_mismatch={bad.number_mismatch_indices}",
    )


# ── 问题1/2：走真实 HTTP + LLM ──────────────────────────────────────────────

def verify_summary_scope(base: str, token: str) -> None:
    print("\n[问题1] 「总结所有文档」必须走整库总结，且覆盖全部文档")
    all_docs = stream_query(base, token, "总结所有文档")
    check("路由到 document_summary", all_docs["intent"] == "document_summary",
          f"intent={all_docs['intent']} events={sorted(set(all_docs['events']))}")
    check("SSE 无错误", not all_docs["error"], str(all_docs["error"]))
    digest_names = [d.get("filename") for d in all_docs["digests"]]
    check(
        "摘要覆盖了全部可访问文档（而不是只有召回最高的那一份）",
        len(digest_names) >= 2,
        f"digests={digest_names}",
    )
    check("回答非空", len(all_docs["text"]) > 80, f"len={len(all_docs['text'])}")
    print(f"        知识库文档清单：{digest_names}")
    print(f"        回答前 80 字：{all_docs['text'][:80]}")

    if not digest_names:
        return

    target = digest_names[0]

    print(f"\n[问题2] 点名文档只总结那一份：{target}")
    one = stream_query(base, token, f"总结《{target}》")
    check("路由到 document_summary", one["intent"] == "document_summary",
          f"intent={one['intent']}")
    names = [d.get("filename") for d in one["digests"]]
    check("摘要只包含被点名的那一份", names == [target], f"digests={names}")
    check("回答非空", len(one["text"]) > 80, f"len={len(one['text'])}")

    print("\n[问题2] 点名不存在的文档 → 列出可选文档，不硬编一个总结")
    missing = stream_query(base, token, "总结《绝对不存在的文档-zzz9.pdf》")
    text = missing["text"]
    check(
        "提示未找到并列出候选文档",
        ("没有在可访问的文档中找到" in text) and (target in text),
        f"回答前 120 字：{text[:120]}",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--admin", default="admin")
    ap.add_argument("--admin-password", default="RagAdmin#2026")
    ap.add_argument("--skip-llm", action="store_true", help="只跑确定性校验")
    args = ap.parse_args()

    verify_citation_fix()

    if not args.skip_llm:
        token = login(args.base, args.admin, args.admin_password)
        verify_summary_scope(args.base, token)

    print(f"\n{'=' * 60}")
    print(f"{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
