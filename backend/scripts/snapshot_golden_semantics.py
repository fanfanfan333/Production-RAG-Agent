"""金标「语义坐标」快照器（漂移判据资产，只读）.

背景
────
``golden_v1.json`` 的金标用 ``(filename, chunk_index)`` **逻辑坐标**标注。
harness（``run_eval_baseline.resolve_labels``）只在**文件名不存在**时报错，
因此一旦重新解析文档、分块边界变化，``chunk_index`` 指到别的段落，评测仍照跑、
分数照出，但结论已无意义且**无任何报错** —— 这正是本脚本要堵的盲区。

本脚本对金标集里出现的**每一个** ``(filename, chunk_index)``，落盘：
  * chunk 实际文本（前 300 字）+ content_type / page / heading / section / parent_id
  * 该文档 id / chunk_count / parser_used / extraction_method / image_count / …
  * 该文档 ``original.*`` 的 md5（磁盘原文件）
  * 该 chunk 所属父块（``chunk_parents`` 中 ``child_indexes`` 含该 index 的行）

坐标语义（已从代码确认，非猜测）
────────────────────────────────
``RetrievedItem.key = f"{document_id}::{chunk_index}"``（``evaluation.py:117``），
``chunk_index`` 取自检索命中的 **child chunk**（``item_from_chunk`` 读
``chunk.chunk_index``）。child chunk 的**正文**不落 PG，而在 Qdrant payload
（``vector_service.upsert_vectors`` 的 ``payload["text"]`` + ``payload["chunk_index"]``）；
PG 侧 ``chunk_parents``（``level in {parent,section}``）只存**父/章节正文**，
其 ``child_indexes`` 列出属于它的子块 index（``chunker._build_parents``）。
因此：**child 正文以 Qdrant payload 为准**，``chunk_parents`` 用于交叉核对父块归属。

用法（容器内）
    docker cp backend/eval/golden_v1.json rag_backend:/tmp/golden_v1.json
    docker cp backend/scripts/snapshot_golden_semantics.py rag_backend:/tmp/
    docker exec -w /app -e PYTHONPATH=/app rag_backend \\
        python /tmp/snapshot_golden_semantics.py --golden /tmp/golden_v1.json \\
        --out /tmp/golden_semantics_before.json --md /tmp/golden_semantics_before.md
    # 再 docker cp 回工作区

不要加 ``-e HOME=/tmp``（会把 user-site 切走 → 依赖变 No module named）。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

TEXT_HEAD = 300


def _md5(path: Path) -> str | None:
    try:
        h = hashlib.md5()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


async def _collect(golden_path: str) -> dict:
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    from sqlalchemy import select, text

    from app.config import get_settings
    from app.db.models import ChunkParent, Document
    from app.db.postgres import get_db_session
    from app.db.qdrant import get_qdrant_client

    # 金标涉及的 (filename → [chunk_index...])
    wanted: dict[str, list[int]] = {}
    for case in golden["cases"]:
        for item in case["expect"]:
            wanted.setdefault(item["filename"], [])
            if item["chunk_index"] not in wanted[item["filename"]]:
                wanted[item["filename"]].append(item["chunk_index"])

    st = get_settings()
    client = get_qdrant_client()

    # 文档元数据（按 filename 精确匹配；同名可能多份 → 全部取）
    names = list(wanted)
    async with get_db_session() as s:
        rows = (
            await s.execute(
                text(
                    "select id::text, filename, filename as fn, tenant_id, chunk_count, "
                    "parser_used, extraction_method, image_count, image_object_count, "
                    "file_type, access_level, coalesce(owner_id::text,'') as owner_id "
                    "from documents where filename = any(:names)"
                ),
                {"names": names},
            )
        ).mappings().all()
        doc_rows = [dict(r) for r in rows]

    out_docs: list[dict] = []
    unresolved: list[dict] = []

    for d in doc_rows:
        doc_id = d["id"]
        fname = d["filename"]
        wanted_idx = sorted(wanted.get(fname, []))

        # Qdrant：拉该文档全部点，建 chunk_index → payload
        pts, _next = await client.scroll(
            collection_name=st.QDRANT_COLLECTION,
            scroll_filter={"must": [{"key": "document_id", "match": {"value": doc_id}}]},
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        by_idx: dict[int, dict] = {}
        for p in pts:
            pay = p.payload or {}
            ci = pay.get("chunk_index")
            if ci is None:
                continue
            by_idx[int(ci)] = {
                "chunk_index": int(ci),
                "content_type": pay.get("content_type"),
                "page_number": pay.get("page_number"),
                "heading": pay.get("heading"),
                "section": pay.get("section"),
                "parent_id": pay.get("parent_id"),
                "section_id": pay.get("section_id"),
                "char_count": pay.get("char_count"),
                "text_head": (pay.get("text") or "")[:TEXT_HEAD],
                "text_len": len(pay.get("text") or ""),
            }
        # 交叉核对：chunk_parents 中 child_indexes 含该 index 的父块
        async with get_db_session() as s:
            parents = (
                await s.execute(
                    select(ChunkParent).where(ChunkParent.document_id == doc_id)
                )
            ).scalars().all()
        parent_map: dict[int, dict] = {}
        for pr in parents:
            for ci in (pr.child_indexes or []):
                parent_map.setdefault(int(ci), {
                    "parent_id": pr.parent_id, "level": pr.level,
                    "heading": pr.heading,
                })

        # 原文件 md5
        up_root = Path(st.IMAGE_STORAGE_DIR)
        if not up_root.is_absolute():
            up_root = Path.cwd() / up_root
        orig_md5 = None
        orig_name = None
        for cand in (up_root / d["tenant_id"] / doc_id, up_root / doc_id):
            if cand.is_dir():
                for f in cand.glob("original.*"):
                    orig_name = f.name
                    orig_md5 = _md5(f)
                    break
            if orig_md5:
                break

        chunks = []
        for idx in wanted_idx:
            rec = by_idx.get(idx)
            if rec is None:
                unresolved.append({"filename": fname, "document_id": doc_id, "chunk_index": idx,
                                   "reason": "Qdrant 中无该 chunk_index 的点"})
                chunks.append({"chunk_index": idx, "resolved": False})
                continue
            entry = dict(rec)
            entry["resolved"] = True
            entry["parent_block"] = parent_map.get(idx)
            chunks.append(entry)

        out_docs.append({
            "filename": fname,
            "document_id": doc_id,
            "tenant_id": d["tenant_id"],
            "chunk_count": d["chunk_count"],
            "parser_used": d["parser_used"],
            "extraction_method": d["extraction_method"],
            "file_type": d["file_type"],
            "access_level": d["access_level"],
            "owner_id": d["owner_id"] or None,
            "image_count": d["image_count"],
            "image_object_count": d["image_object_count"],
            "qdrant_points": len(pts),
            "original_file": orig_name,
            "original_md5": orig_md5,
            "chunks": chunks,
        })

    # 文件名在库中完全不存在的金标坐标
    found_names = {d["filename"] for d in doc_rows}
    missing_names = [n for n in names if n not in found_names]

    return {
        "golden_name": golden.get("name"),
        "golden_revision": golden.get("revision"),
        "text_head_chars": TEXT_HEAD,
        "coord_semantics": "key = document_id::chunk_index（child chunk；正文以 Qdrant payload.text 为准）",
        "filenames_in_golden": names,
        "filenames_missing_in_db": missing_names,
        "documents": out_docs,
        "unresolved_coords": unresolved,
        "summary": {
            "golden_filenames": len(names),
            "resolved_documents": len({d["filename"] for d in out_docs}),
            "golden_coords": sum(len(v) for v in wanted.values()),
            "unresolved_coords": len(unresolved),
            "missing_filenames": len(missing_names),
        },
    }


def _render_md(snap: dict) -> str:
    lines = [
        f"# 金标语义坐标快照（{snap['golden_name']} rev{snap['golden_revision']}）",
        "",
        f"- 坐标语义：`{snap['coord_semantics']}`",
        f"- 涉及文件名：{snap['summary']['golden_filenames']}；解析到文档：{snap['summary']['resolved_documents']}；"
        f"金标坐标数：{snap['summary']['golden_coords']}；**未解析坐标：{snap['summary']['unresolved_coords']}**",
        f"- 库中缺失的文件名：{snap['filenames_missing_in_db'] or '（无）'}",
        "",
    ]
    for d in snap["documents"]:
        lines.append(f"## {d['filename']}")
        lines.append(
            f"- doc_id=`{d['document_id']}` tenant=`{d['tenant_id']}` access=`{d['access_level']}` "
            f"chunk_count={d['chunk_count']} qdrant_points={d['qdrant_points']}"
        )
        lines.append(
            f"- parser=`{d['parser_used']}` extraction=`{d['extraction_method']}` "
            f"image_count={d['image_count']}/{d['image_object_count']} "
            f"original=`{d['original_file']}` md5=`{d['original_md5']}`"
        )
        lines.append("")
        lines.append("| chunk_index | resolved | ct | page | heading | parent_id | 文本前 120 字 |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in d["chunks"]:
            if not c.get("resolved"):
                lines.append(f"| {c['chunk_index']} | ❌ | | | | | （未解析） |")
                continue
            head = (c.get("text_head") or "").replace("|", "\\|").replace("\n", " ")[:120]
            lines.append(
                f"| {c['chunk_index']} | ✅ | {c.get('content_type')} | {c.get('page_number')} | "
                f"{c.get('heading')} | {c.get('parent_id')} | {head} |"
            )
        lines.append("")
    if snap["unresolved_coords"]:
        lines.append("## 未解析坐标（漂移/损坏信号）")
        for u in snap["unresolved_coords"]:
            lines.append(f"- `{u['filename']}` chunk_index={u['chunk_index']} doc={u['document_id']} — {u['reason']}")
    return "\n".join(lines)


async def main() -> int:
    ap = argparse.ArgumentParser(description="金标语义坐标快照器（只读）")
    ap.add_argument("--golden", default="/tmp/golden_v1.json")
    ap.add_argument("--out", default="/tmp/golden_semantics_before.json")
    ap.add_argument("--md", default="/tmp/golden_semantics_before.md")
    args = ap.parse_args()
    snap = await _collect(args.golden)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    with open(args.md, "w", encoding="utf-8") as f:
        f.write(_render_md(snap))
    s = snap["summary"]
    print(f"snapshot -> {args.out} / {args.md}")
    print(f"filenames={s['golden_filenames']} resolved_docs={s['resolved_documents']} "
          f"coords={s['golden_coords']} unresolved={s['unresolved_coords']} missing_files={s['missing_filenames']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
