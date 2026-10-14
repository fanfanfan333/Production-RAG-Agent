"""Upload only manifest fixtures serially or resume their graph jobs; never delete business documents."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_professional import available_gb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path, help="Directory produced by make_complex_professional.py")
    parser.add_argument("--phase", choices=("upload", "graph"), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    gold_path = args.fixtures / "gold.json"
    golden = json.loads(gold_path.read_text(encoding="utf-8"))
    for name in golden["documents"]:
        if Path(name).name != name or not (args.fixtures / "documents" / name).is_file():
            parser.error("夹具文件名越界或文件缺失")
    token = os.environ.get("RAG_EVAL_TOKEN")
    if not token: parser.error("需要通过 RAG_EVAL_TOKEN 提供当前账号令牌")
    state_path = args.fixtures / "run-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    def checkpoint():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(state_path)
    with httpx.Client(base_url=args.url.rstrip("/"), headers={"Authorization": f"Bearer {token}"}, timeout=60, trust_env=False) as client:
        response = client.get("/auth/me"); response.raise_for_status(); user = response.json()
        digest = hashlib.sha256(gold_path.read_bytes()).hexdigest()
        if state and (state.get("gold_sha256") != digest or state.get("owner_id") != user["id"]):
            parser.error("续跑账号或金标版本不同，拒绝复用现有验收状态")
        if not state:
            import uuid
            name = "professional-acceptance-" + uuid.uuid4().hex[:12]
            response = client.post("/kb/collections", json={"name": name, "description": "公开/构造夹具验收；不包含业务文档"})
            response.raise_for_status()
            state.update(collection_id=response.json()["id"], collection_name=name, owner_id=user["id"], gold_sha256=digest, documents={})
            checkpoint()
        response = client.get("/kb/collections"); response.raise_for_status()
        if not any(row["id"] == state["collection_id"] and row["name"] == state["collection_name"] for row in response.json()["collections"]):
            parser.error("专用验收知识库不存在或不属于当前账号")
        def document(identifier):
            page = 1
            while True:
                response = client.get("/documents", params={"collection_id": state["collection_id"], "limit": 100, "page": page})
                response.raise_for_status(); listing = response.json()
                rows = listing["documents"]
                for row in rows:
                    if row["document_id"] == identifier: return row
                if len(rows) < 100: raise RuntimeError("文档不在当前专用验收知识库中")
                page += 1
        try:
            for name in golden["documents"]:
                record = state["documents"].get(name)
                if args.phase == "graph" and not name.startswith("relation-"): continue
                if available_gb() < 5:
                    raise RuntimeError("启动下一项前宿主可用内存不足5GB；保留检查点，释放内存后可续跑")
                if args.phase == "upload":
                    if record and document(record["id"])["status"] == "completed": continue
                    with (args.fixtures / "documents" / name).open("rb") as stream:
                        response = client.post("/upload", data={"collection_id": state["collection_id"], "access_level": "private",
                            **({"company_id": user.get("tenant_id") or "default"} if user["role"] == "admin" else {})}, files={"files": (name, stream)})
                    response.raise_for_status(); uploaded = response.json()["documents"][0]
                    identifier = uploaded.get("document_id") or uploaded.get("existing_document_id")
                    if not identifier: raise RuntimeError(f"{name}: 上传未返回文档记录")
                    row = document(identifier)
                    record = dict(id=identifier, filename=name, status=row["status"])
                    state["documents"][name] = record; checkpoint()
                    started = time.monotonic(); last_stage = None
                    while row["status"] not in {"completed", "failed"}:
                        if available_gb() < 1 or time.monotonic() - started > 900:
                            client.post(f"/documents/{identifier}/cancel-ingestion")
                            raise RuntimeError(f"{name}: 当前入库已请求取消，保留检查点")
                        stage = row.get("current_stage")
                        if stage != last_stage: print(f"{name}: {stage}", flush=True); last_stage = stage
                        time.sleep(2); row = document(identifier)
                    record.update(status=row["status"], error=row.get("error"), chunk_count=row.get("chunk_count"), page_count=row.get("page_count"))
                    checkpoint()
                    if row["status"] != "completed": raise RuntimeError(f"{name}: {row.get('error') or '解析失败'}")
                else:
                    if not record or document(record["id"])["status"] != "completed": raise RuntimeError(f"{name}: 必须先完成文档入库")
                    while True:
                        if available_gb() < 5: raise RuntimeError("图抽取前可用内存不足5GB，保留检查点")
                        response = client.post(f"/documents/{record['id']}/graph-index", timeout=45)
                        response.raise_for_status(); job = response.json()
                        record["graph"] = job; checkpoint()
                        print(f"{name}: graph {job['cursor']}/{job['total']} {job['status']}", flush=True)
                        if job["status"] == "failed": raise RuntimeError(job.get("error") or "图抽取失败")
                        if job["status"] == "completed": break
                print(f"{name}: {args.phase} verified", flush=True)
            state[args.phase + "_completed"] = True; state.pop("blocked_reason", None); checkpoint()
        except (RuntimeError, httpx.HTTPError) as exc:
            state["blocked_reason"] = str(exc); checkpoint(); print(str(exc), flush=True); raise SystemExit(1)
    print(f"Completed {args.phase}; collection={state['collection_id']}")


if __name__ == "__main__": main()
