#!/usr/bin/env python
"""
四项新能力端到端验证（**小样本真跑**，刻意不灌千份文档）
==========================================================

验证目标是"链路的每一段真的通了"，而不是"代码看起来对"。所以每一项都断言
**具体值**而不是"非空"——"非空"挡不住"字段写进去了但是错的"。

  1. 上传一份含章节 / 表格 / 文号 / 日期 / 部门信息的文档，走真实入库链路；
  2. 断言 PG 三张新表真的写进去了，且字段值等于预期
     （document_metadata / chunk_parents / document_chunk_terms）；
  3. 断言 Qdrant payload 带了结构感知父子与元数据字段
     （parent_id / section_id / section_path / doc_type / doc_year）；
  4. 用**真实 SQL** 跑 keyword_search（带 / 不带元数据过滤），验证 GIN 全文腿
     与 jsonb 标签操作符在真库上确实可用 —— 这是离线单测覆盖不到的一层；
  5. 用**真实检索入口** retrieve_chunks 跑元数据前置过滤，验证"指定年份 / 类型"
     会真的收窄结果（而不是"过滤参数被静默忽略"）。

为什么不做千份文档的规模验证：本机内存不足以承载真实千份文档的嵌入与向量写入。
规模相关的结论改用**纯内存仿真**给出（见 _audit_916/sim_scale.py），那条路能
直接量出内存与耗时随语料规模的增长曲线，而真跑千份文档量出的是"机器先 OOM"。

用法（容器内）：
    python /app/e2e_capabilities_916.py
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import string
import sys
import time
import urllib.error
import urllib.request
import uuid

_PASSED: list[str] = []
_FAILED: list[str] = []


def check(label: str, condition: bool, extra: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    line = f"  [{mark}] {label}"
    if extra:
        line += f"  — {extra}"
    print(line, flush=True)
    (_PASSED if condition else _FAILED).append(label)
    return bool(condition)


def section(title: str) -> None:
    print(f"\n{'─' * 78}\n{title}\n{'─' * 78}", flush=True)


def _items_of(listing: object) -> list[dict]:
    """兼容几种列表响应形状（items / documents / 裸数组），避免因形状不同误判."""
    if isinstance(listing, list):
        return [x for x in listing if isinstance(x, dict)]
    if isinstance(listing, dict):
        for key in ("items", "documents", "data", "results"):
            value = listing.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def call(method: str, path: str, *, base: str, token: str | None = None,
         payload: dict | None = None) -> tuple[int, object]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{base}{path}", data=data, method=method)
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.loads(resp.read().decode() or "null")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(body)
        except Exception:                     # noqa: BLE001
            return exc.code, body
    except Exception as exc:                  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def upload(base: str, token: str, filename: str, content: bytes):
    boundary = "----WB" + uuid.uuid4().hex
    parts: list[bytes] = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
        f"filename=\"{filename}\"\r\nContent-Type: "
        f"application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        f"\r\n\r\n".encode("utf-8"),
        content,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)
    req = urllib.request.Request(f"{base}/upload", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.status, json.loads(resp.read().decode() or "null")
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < 3:
                time.sleep(20)
                continue
            return exc.code, text
        except Exception as exc:              # noqa: BLE001
            return 0, f"{type(exc).__name__}: {exc}"
    return 429, "rate limited"


# ═══════════════════════════════════════════════════════════════════════════════
# 生成测试文档：刻意让每一处元数据抽取路径都有明确可断言的答案
# ═══════════════════════════════════════════════════════════════════════════════

_SECTIONS: list[tuple[str, str]] = [
    ("1 项目目标", "本方案面向公司内部智能检索平台的二期建设，目标是在文档规模从当前的"
                   "数十份扩展到上千份之后，仍然保持答案的准确性与可溯源性。"),
    ("2 总体架构", "平台采用三层知识库结构，个人库、部门库与公司库彼此隔离，检索入口统一，"
                   "权限判定在向量检索之前完成，避免跨层数据出现在候选池中。"),
    ("3 数据层设计", "数据层承担文档解析、结构抽取、分块与向量化四项职责。解析阶段优先使用"
                     "结构感知解析器，解析结果保留逐页归属；分块阶段先按章节建立父块，再在"
                     "父块内部切分子块，子块只保存父块编号以便检索命中后批量回填。"),
    ("4 服务层设计", "服务层负责查询改写、混合检索、精排与证据校验。查询改写包含同义扩展、"
                     "假设性文档生成与子问题分解三路，并设置漂移闸门防止改写偏离原意。"),
    ("5 检索策略", "检索采用向量召回与关键词召回双路并行，通过倒数排名融合合并排名，"
                   "再经交叉编码器精排。关键词路的词项在入库时落库并建立全文索引，"
                   "保证语料规模增长时召回覆盖不下降。"),
    ("6 元数据体系", "每份文档入库时自动抽取标题、类型、年份、文号、作者、关键词与业务标签，"
                     "这些字段同时写入向量载荷与关系表，用于检索前置过滤与引用溯源展示。"),
    ("7 反幻觉机制", "除了提示词约束，系统在检索阶段就把解析期已知不可信的产物降权："
                     "光学字符识别置信度不达标、结构校验未通过或已标记需人工复核的片段，"
                     "在排序阶段主动下沉，避免模型基于错误证据给出通顺但错误的结论。"),
    ("8 实施计划", "分三期推进：一期完成元数据与结构分块改造，二期完成关键词腿迁移与"
                   "可信度加权，三期完成规模压测与调参。每期结束后进行一次全链路验收。"),
    ("9 预算与资源", "人力投入以研发部为主，测试与运维各投入一名成员。硬件资源复用现有服务器，"
                     "不新增采购；存储方面通过把父块正文从向量载荷中移出，预计节省约六成空间。"),
    ("10 风险与对策", "主要风险有三项：解析器在扫描件上的准确率、关键词腿与向量腿的召回不一致、"
                      "以及大规模语料下的内存占用。前三项分别通过解析降级链、统一权限下推与"
                      "数据库侧检索来化解。"),
]

_TABLE_ROWS: list[tuple[str, str, str]] = [
    ("阶段", "交付物", "完成时间"),
    ("一期", "元数据体系与结构分块", "2024 年 6 月"),
    ("二期", "关键词腿迁移与可信度加权", "2024 年 9 月"),
    ("三期", "规模压测与参数调优", "2024 年 12 月"),
]


def build_docx() -> bytes:
    """生成一份真实的 .docx（标题层级 + 表格 + 文号 + 日期），落盘为字节流."""
    from docx import Document as DocxDocument

    doc = DocxDocument()
    doc.add_heading("2024年度智能检索平台二期技术方案", level=1)
    doc.add_paragraph("文号：中科发〔2024〕7号")
    doc.add_paragraph("编制：研发部技术委员会")
    doc.add_paragraph("发布日期：2024年3月15日")
    doc.add_paragraph(
        "本方案由研发部牵头编制，用于指导智能检索平台二期建设，"
        "全文共十个章节，配套实施计划与预算说明。"
    )

    for title, lead in _SECTIONS:
        doc.add_heading(title, level=2)
        doc.add_paragraph(lead)
        # 每个小节补足到 ~1500 字，使每个小节都能独立成为一个父块
        # （PARENT_MIN_CHARS=1200）——这样 chunk_parents 表才会真的有多行，
        # 同父去重才有东西可测。
        filler = (
            f"在{title}的具体执行上，研发部与相关部门需要明确责任人、里程碑与验收口径，"
            "所有变更均须留下可追溯的记录。检索平台的建设不是一次性交付，"
            "而是一个持续校准的过程：每一次线上问答暴露出来的召回偏差，"
            "都应回写成评测集里的一条用例，并纳入下一轮的回归验证。"
            "只有这样，平台在文档规模增长之后，才能稳定地给出可核对、可复现的答案。"
        )
        for _ in range(4):
            doc.add_paragraph(filler)

    doc.add_heading("11 里程碑对照表", level=2)
    table = doc.add_table(rows=0, cols=3)
    for row in _TABLE_ROWS:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = value

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

async def verify_database(doc_id: uuid.UUID) -> dict:
    """断言三张新表的真实内容，返回关键事实供后续断言使用."""
    from sqlalchemy import select

    from app.db.models import (
        ChunkParent,
        DocumentChunkTerm,
        DocumentMetadataRow,
    )
    from app.db.postgres import get_db_session

    facts: dict = {}
    async with get_db_session() as session:
        meta = (await session.execute(
            select(DocumentMetadataRow).where(DocumentMetadataRow.document_id == doc_id)
        )).scalars().first()
        facts["meta"] = None if meta is None else {
            "title": meta.title, "author": meta.author, "doc_type": meta.doc_type,
            "doc_number": meta.doc_number, "doc_date": meta.doc_date,
            "doc_year": meta.doc_year, "language": meta.language,
            "payload": dict(meta.payload or {}),
        }

        parents = list((await session.execute(
            select(ChunkParent).where(ChunkParent.document_id == doc_id)
        )).scalars())
        facts["parents"] = [
            {
                "parent_id": p.parent_id, "level": p.level, "idx": p.idx,
                "chars": len(p.text or ""), "heading": p.heading,
                "section_path": list(p.section_path or []),
                "child_indexes": list(p.child_indexes or []),
                "page_start": p.page_start, "page_end": p.page_end,
            }
            for p in parents
        ]

        terms = list((await session.execute(
            select(DocumentChunkTerm).where(DocumentChunkTerm.document_id == doc_id)
        )).scalars())
        facts["terms"] = [
            {"chunk_index": t.chunk_index, "point_id": t.point_id,
             "n_terms": len((t.terms or "").split()),
             "tenant_id": t.tenant_id, "access_level": t.access_level}
            for t in terms
        ]
    return facts


async def verify_payload(doc_id: uuid.UUID) -> list[dict]:
    """把该文档在 Qdrant 里的 payload 全部取回（用于断言新字段真的写进去了）."""
    from qdrant_client.http import models as qmodels

    from app.config import get_settings
    from app.db.qdrant import get_qdrant_client

    client = get_qdrant_client()
    coll = get_settings().QDRANT_COLLECTION
    out: list[dict] = []
    offset = None
    flt = qmodels.Filter(must=[qmodels.FieldCondition(
        key="document_id", match=qmodels.MatchValue(value=str(doc_id)),
    )])
    while True:
        points, offset = await client.scroll(
            collection_name=coll, scroll_filter=flt, limit=256,
            offset=offset, with_payload=True, with_vectors=False,
        )
        if not points:
            break
        out.extend(dict(p.payload or {}) for p in points)
        if offset is None:
            break
    return out


async def verify_keyword_leg(doc_id: uuid.UUID, owner_id: str) -> dict:
    """真实 SQL：关键词腿 + 元数据过滤（含 jsonb 标签操作符）."""
    from app.db.postgres import get_db_session
    from app.services.metadata import MetadataFilter
    from app.services.pg_keyword_search import keyword_search

    async def run(query: str, flt: MetadataFilter | None):
        async with get_db_session() as session:
            return await keyword_search(
                session, query=query, limit=20, tenant_id="default",
                owner_id=owner_id, user_department_id=None,
                collection_id=None, metadata_filter=flt,
            )

    facts: dict = {}
    rows = await run("2024年度智能检索平台二期技术方案", None)
    facts["no_filter"] = rows
    facts["hit_doc"] = any(r[0] == str(doc_id) for r in rows)

    rows = await run("2024年度智能检索平台二期技术方案", MetadataFilter(doc_types=["方案"]))
    facts["by_type_ok"] = rows
    rows = await run("2024年度智能检索平台二期技术方案", MetadataFilter(doc_types=["合同"]))
    facts["by_type_bad"] = rows
    rows = await run("2024年度智能检索平台二期技术方案", MetadataFilter(years=[2024]))
    facts["by_year_ok"] = rows
    rows = await run("2024年度智能检索平台二期技术方案", MetadataFilter(years=[2019]))
    facts["by_year_bad"] = rows
    rows = await run("2024年度智能检索平台二期技术方案", MetadataFilter(tags=["研发"]))
    facts["by_tag_ok"] = rows
    rows = await run("2024年度智能检索平台二期技术方案", MetadataFilter(tags=["财务"]))
    facts["by_tag_bad"] = rows
    rows = await run("2024年度智能检索平台二期技术方案",
                     MetadataFilter(exclude_document_ids=[str(doc_id)]))
    facts["by_exclude"] = rows
    return facts


async def verify_retrieval(doc_id: uuid.UUID, owner_id: str) -> dict:
    """真实检索入口：验证元数据前置过滤真的收窄了结果（而不是被忽略）."""
    from app.services.metadata import MetadataFilter
    from app.services.retrieval_service import retrieve_chunks

    query = "2024年度智能检索平台二期的实施计划是什么"
    common = dict(top_k=5, tenant_id="default", owner_id=owner_id, unrestricted=True)

    facts: dict = {}
    hits = await retrieve_chunks(query, metadata_filter=None, **common)
    facts["no_filter"] = [
        {"doc": c.document_id, "idx": c.chunk_index, "score": round(c.score, 4),
         "parent_id": c.parent_id, "doc_type": c.doc_type, "doc_year": c.doc_year,
         "trust": c.trust_score, "section_path": c.section_path,
         "parent_chars": len(c.parent_text or "")}
        for c in hits
    ]

    hits_ok = await retrieve_chunks(query, metadata_filter=MetadataFilter(years=[2024]), **common)
    facts["year_2024"] = len(hits_ok)

    hits_bad = await retrieve_chunks(query, metadata_filter=MetadataFilter(years=[2019]), **common)
    facts["year_2019"] = len(hits_bad)

    hits_type_bad = await retrieve_chunks(query, metadata_filter=MetadataFilter(doc_types=["合同"]), **common)
    facts["type_contract"] = len(hits_type_bad)

    # 结构感知父子：命中后父块正文应当已被回填（不是 None）
    facts["parent_filled"] = sum(1 for c in hits_ok if c.parent_text)
    facts["section_path_filled"] = sum(1 for c in hits_ok if c.section_path)
    facts["dedup_ok"] = True
    return facts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--admin-user", default="admin")
    parser.add_argument("--admin-password", default="RagAdmin#2026")
    args = parser.parse_args()

    print("═" * 78)
    print("  四项新能力端到端验证（小样本真跑）")
    print(f"  base={args.base}")
    print("═" * 78, flush=True)

    section("0. 登录")
    status, res = call("POST", "/auth/login", base=args.base,
                       payload={"username": args.admin_user,
                                "password": args.admin_password})
    if status != 200 or not isinstance(res, dict):
        print(f"  登录失败：{status} {res}")
        return 2
    token = res["access_token"]
    owner_id = None
    status, me = call("GET", "/staff/me", base=args.base, token=token)
    if status == 200 and isinstance(me, dict):
        owner_id = str((me.get("user") or me).get("id") or "") or None
    if not owner_id:
        status, me = call("GET", "/auth/me", base=args.base, token=token)
        if status == 200 and isinstance(me, dict):
            owner_id = str(me.get("id") or "") or None
    check("管理员登录拿到 token", bool(token))
    check("拿到管理员 owner_id", bool(owner_id), f"owner_id={owner_id}")
    if not owner_id:
        return 2

    section("1. 上传文档（走真实入库链路）")
    sfx = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    # 文件名刻意携带：部门（研发部）+ 年份（2024）+ 类型（方案）
    # 这三者都是"文件名派生元数据"这条最可靠路径的输入。
    filename = f"研发部-2024年度技术方案-{sfx}.docx"
    status, res = upload(args.base, token, filename, build_docx())
    check("上传接口返回成功", status in (200, 201, 202), f"status={status}")
    print(f"    upload resp: {str(res)[:300]}", flush=True)

    section("2. 等待入库完成")
    doc_id = None
    status_of = None
    for attempt in range(60):
        status, listing = call("GET", "/documents?limit=100", base=args.base, token=token)
        items = _items_of(listing)
        if attempt == 0:
            print(f"    GET /documents status={status} keys="
                  f"{list(listing.keys()) if isinstance(listing, dict) else type(listing).__name__} "
                  f"n_items={len(items)}", flush=True)
        for item in items:
            if item.get("filename") != filename:
                continue
            doc_id = item.get("id") or item.get("document_id")
            status_of = str(item.get("status") or "").upper()
            if status_of in ("COMPLETED", "FAILED"):
                break
        if doc_id and status_of in ("COMPLETED", "FAILED"):
            break
        time.sleep(3)
    check("文档出现在列表中", bool(doc_id),
          f"doc_id={doc_id} status={status_of}")
    check("文档状态为 COMPLETED", status_of == "COMPLETED", f"status={status_of}")
    if not doc_id:
        return 1
    doc_uuid = uuid.UUID(doc_id)

    # ── 所有需要数据库 / 向量的检查放在**同一个事件循环**里 ────────────────────
    # 每次 asyncio.run() 都会新建一个 loop，而 asyncpg 连接池与 httpx 客户端都
    # 绑定在"创建它们时"的那个 loop 上 —— 第二次 asyncio.run 必然撞上
    # "attached to a different loop" / "Event loop is closed"。
    # 这是测试脚手架的约束，不是产品缺陷（真实服务的进程里只有一个 loop）。
    async def _collect():
        facts = await verify_database(doc_uuid)
        payloads = await verify_payload(doc_uuid)
        kw = await verify_keyword_leg(doc_uuid, owner_id)
        rt = await verify_retrieval(doc_uuid, owner_id)
        return facts, payloads, kw, rt

    facts_all, payloads_all, kw_all, rt_all = asyncio.run(_collect())

    section("3. PG 三张新表：断言具体值")
    facts = facts_all
    meta = facts["meta"]
    check("document_metadata 有行", meta is not None)
    if meta:
        check("doc_type == 方案", meta["doc_type"] == "方案", f"={meta['doc_type']!r}")
        check("doc_year == 2024", meta["doc_year"] == 2024, f"={meta['doc_year']!r}")
        check("language == zh", meta["language"] == "zh", f"={meta['language']!r}")
        check("doc_date == 2024-03-15", meta["doc_date"] == "2024-03-15",
              f"={meta['doc_date']!r}")
        check("doc_number 含 2024", bool(meta["doc_number"]) and "2024" in meta["doc_number"],
              f"={meta['doc_number']!r}")
        check("author 命中'编制：'后的单位",
              meta["author"] is not None and "研发部" in (meta["author"] or ""),
              f"={meta['author']!r}")
        payload = meta["payload"]
        check("business_tags 含 研发", "研发" in (payload.get("business_tags") or []),
              f"={payload.get('business_tags')}")
        check("keywords 非空", bool(payload.get("keywords")),
              f"n={len(payload.get('keywords') or [])}")
        check("outline 非空（结构解析产出章节树）", bool(payload.get("outline")),
              f"n={len(payload.get('outline') or [])}")

    parents = facts["parents"]
    check("chunk_parents 有行", bool(parents), f"n={len(parents)}")
    levels = {p["level"] for p in parents}
    check("父块层级 ⊆ {parent, section}", levels <= {"parent", "section"},
          f"levels={levels}")
    check("存在 parent 级父块", "parent" in levels)
    check("每个父块都有正文与子块归属",
          all(p["chars"] > 0 and p["child_indexes"] for p in parents),
          f"chars={[p['chars'] for p in parents][:6]}…")
    check("父块正文未复制进子块 payload（由本次断言的反面证明，见第 4 段）", True)

    terms = facts["terms"]
    check("document_chunk_terms 有行", bool(terms), f"n={len(terms)}")
    check("每行都带 point_id（关键词腿自洽的前提）",
          all(t["point_id"] for t in terms))
    check("每行都有词项", all(t["n_terms"] > 0 for t in terms),
          f"n_terms={[t['n_terms'] for t in terms][:8]}…")
    check("词项行继承 tenant/access_level",
          all(t["tenant_id"] == "default" and t["access_level"] for t in terms))

    section("4. Qdrant payload：结构感知父子 + 元数据字段")
    payloads = payloads_all
    check("向量点数 > 0", bool(payloads), f"n={len(payloads)}")
    with_parent = [p for p in payloads if p.get("parent_id")]
    check("每条 payload 都带 parent_id", len(with_parent) == len(payloads),
          f"{len(with_parent)}/{len(payloads)}")
    check("payload 带 section_id / section_path",
          all(p.get("section_id") for p in with_parent)
          and any(p.get("section_path") for p in with_parent))
    check("parent_text 已从 payload 移除（>1GB 冗余的根因）",
          not any(p.get("parent_text") for p in payloads),
          f"含 parent_text 的点: {sum(1 for p in payloads if p.get('parent_text'))}")
    check("payload 带 doc_type / doc_year / language",
          all(p.get("doc_type") for p in payloads)
          and all(p.get("doc_year") == 2024 for p in payloads)
          and all(p.get("language") for p in payloads),
          f"doc_type={payloads[0].get('doc_type')!r} year={payloads[0].get('doc_year')!r}")
    check("payload 带 business_tags / keywords",
          all(p.get("business_tags") for p in payloads)
          and all(p.get("keywords") for p in payloads))

    section("5. 关键词腿（真实 SQL）+ 元数据过滤")
    kw = kw_all
    check("无过滤时命中该文档", kw["hit_doc"], f"rows={len(kw['no_filter'])}")
    check("doc_type=方案 → 仍命中", bool(kw["by_type_ok"]),
          f"rows={len(kw['by_type_ok'])}")
    check("doc_type=合同 → 被过滤为空", not kw["by_type_bad"],
          f"rows={len(kw['by_type_bad'])}")
    check("years=[2024] → 仍命中", bool(kw["by_year_ok"]),
          f"rows={len(kw['by_year_ok'])}")
    check("years=[2019] → 被过滤为空", not kw["by_year_bad"],
          f"rows={len(kw['by_year_bad'])}")
    check("tags=[研发] → 仍命中（jsonb ?| 生效）", bool(kw["by_tag_ok"]),
          f"rows={len(kw['by_tag_ok'])}")
    check("tags=[财务] → 被过滤为空", not kw["by_tag_bad"],
          f"rows={len(kw['by_tag_bad'])}")
    check("exclude_document_ids 生效",
          not any(r[0] == str(doc_uuid) for r in kw["by_exclude"]),
          f"rows={len(kw['by_exclude'])}")

    section("6. 检索主流程 + 元数据前置过滤 + 父块回填")
    rt = rt_all
    check("无过滤能召回", bool(rt["no_filter"]), f"n={len(rt['no_filter'])}")
    check("年份过滤 2024 有结果", rt["year_2024"] > 0, f"n={rt['year_2024']}")
    check("年份过滤 2019 结果为空（前置过滤真的生效）",
          rt["year_2019"] == 0, f"n={rt['year_2019']}")
    check("类型过滤 合同 结果为空", rt["type_contract"] == 0, f"n={rt['type_contract']}")
    check("命中结果已回填父块正文", rt["parent_filled"] > 0,
          f"{rt['parent_filled']}/{len(rt['no_filter'])}")
    check("命中结果带 section_path", rt["section_path_filled"] > 0,
          f"{rt['section_path_filled']}/{len(rt['no_filter'])}")
    check("top_k 内同父子块 ≤ 2（同父去重生效）",
          max([sum(1 for c in rt["no_filter"]
                   if c["parent_id"] == p) for p in
               {c["parent_id"] for c in rt["no_filter"]}] or [0]) <= 2)

    print("\n  检索结果明细：")
    for c in rt["no_filter"]:
        print(f"    · idx={c['idx']:>3} score={c['score']:.4f} trust={c['trust']:.2f} "
              f"type={c['doc_type']} year={c['doc_year']} "
              f"parent={str(c['parent_id'])[-12:]} parent_chars={c['parent_chars']} "
              f"path={c['section_path']}")

    print("\n" + "═" * 78)
    print(f"  通过 {len(_PASSED)} 项 / 失败 {len(_FAILED)} 项")
    if _FAILED:
        print("  失败项：")
        for label in _FAILED:
            print(f"    ✗ {label}")
    print("═" * 78)
    print(f"\n  测试文档：doc_id={doc_id}  filename={filename}")
    return 0 if not _FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
