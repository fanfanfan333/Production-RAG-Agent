"""
清洗能力金标跑分（Scoring）.

与 ``tests/test_golden_cleaning.py`` 的关系：

    测试  —— 二值的：**每条金标过 / 不过**，不过就红；
    本脚本—— 连续的：**每类文档清洗前后的量化对比**，用来看"改完是变好还是变坏"
              以及"离目标还差多少"。两者共用同一份金标与同一套判定助手
              （直接 import 测试模块，避免两处判定逻辑漂移）。

指标（**均在本机/容器内实测得出，不做估算**）
────────────────────────────────────────────
    噪声残留率   清洗前后不可见/控制字符占比（BOM/零宽/软连字符/全角空格/NBSP/C0/CR）
    正文保真率   清洗后字符数 / 清洗前（<1 = 剥掉了东西；>1 = 屏蔽占位符变长）
    事实保全率   金标事实锚点（营业收入 33500、毛利率 25.0%…）清洗后仍在的比例
    页偏移自洽   清洗后 page_spans 是否仍精确框住各页（页码不漂移）
    注入屏蔽率   注入载荷是否被屏蔽且留下可见占位符
    chunk 质量   按 MIN/MAX_CHUNK_SIZE 窗口切块后的「干净 chunk 占比」与平均长度
                 —— 这是 ragas 的**代理指标**：环境未装 ragas，不冒充 ragas 分数
    LLM 通过率   清洗 → 本机 qwen3:8b → 仍答得出事实的比例（--llm 开启）

用法：
    python backend/scripts/golden_clean_eval.py                 # 全部金标，含 LLM
    python backend/scripts/golden_clean_eval.py --no-llm        # 跳过 LLM（快）
    python backend/scripts/golden_clean_eval.py --llm-limit 8
    python backend/scripts/golden_clean_eval.py --out <path>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_TESTS = _BACKEND_ROOT / "tests"


def _import_test_module():
    """复用金标测试模块：同一份金标、同一套判定助手，避免两处逻辑漂移。"""
    spec = importlib.util.spec_from_file_location(
        "golden_cleaning_tests", _TESTS / "test_golden_cleaning.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["golden_cleaning_tests"] = module
    spec.loader.exec_module(module)
    return module


T = _import_test_module()

NOISE_CHARS = "\ufeff\u200b\u00ad\u3000\u00a0\x07\r"


def _noise_rate(text: str) -> float:
    if not text:
        return 0.0
    return sum(text.count(c) for c in NOISE_CHARS) / len(text)


def _chunks(text: str, size: int = 2000, overlap: int = 200):
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, max(len(text), 1), step)] if text else []


def evaluate_case(case: dict) -> dict:
    result, pages, full_text = T._run_clean(case)
    before, after = full_text, result.full_text

    facts = list(case["must_contain"])
    facts_kept = [f for f in facts if T._present(f, after)]
    preserved = list(case.get("must_be_preserved") or [])
    preserved_kept = [p for p in preserved if p in after]

    chunks_before = _chunks(before)
    chunks_after = _chunks(after)
    clean_chunks_before = [c for c in chunks_before if _noise_rate(c) == 0]
    clean_chunks_after = [c for c in chunks_after if _noise_rate(c) == 0]

    # 页偏移自洽：每个新区间必须精确框住该页清洗后的文本
    pagemap_ok = bool(result.pagemap_intact) and len(result.page_spans) == len(pages)
    if pagemap_ok:
        for page, (start, end) in zip(pages, result.page_spans):
            if after[start:end] != T.clean_and_mask(page.text)[0]:
                pagemap_ok = False
                break

    expect_masked = int(case.get("expect_masked_paragraphs") or 0)
    masking_ok = result.masked_paragraphs == expect_masked and (
        "已屏蔽" in after if expect_masked else True
    )
    injection_leaked = any(
        tok in after for tok in case.get("must_not_contain") or []
        if tok not in NOISE_CHARS
    )

    retention = (len(after) / len(before)) if before else 1.0
    floor = float(case.get("min_retention_ratio") or 0.8)

    return {
        "id": case["id"],
        "format": case["format"],
        "noise_injected": bool(case.get("noise_injected")),
        "chars_before": len(before),
        "chars_after": len(after),
        "noise_rate_before": round(_noise_rate(before), 5),
        "noise_rate_after": round(_noise_rate(after), 5),
        "retention": round(retention, 4),
        "retention_ok": retention >= floor,
        "facts_total": len(facts),
        "facts_kept": len(facts_kept),
        "facts_rate": round(len(facts_kept) / max(len(facts), 1), 4),
        "preserved_total": len(preserved),
        "preserved_kept": len(preserved_kept),
        "chunks_before": len(chunks_before),
        "chunks_after": len(chunks_after),
        "clean_chunk_rate_before": round(
            len(clean_chunks_before) / max(len(chunks_before), 1), 4
        ),
        "clean_chunk_rate_after": round(
            len(clean_chunks_after) / max(len(chunks_after), 1), 4
        ),
        "pagemap_ok": pagemap_ok,
        "masking_ok": masking_ok,
        "injection_leaked": injection_leaked,
        "masked_paragraphs": result.masked_paragraphs,
    }


def evaluate_image_channel(case: dict) -> dict | None:
    if not case.get("images"):
        return None
    payload = case["images"][0]
    img = T.FakeImage(
        ocr_text=payload.get("ocr_text", ""),
        structured_content=payload.get("structured_content", ""),
        vision_caption=payload.get("vision_caption"),
    )
    before = (img.ocr_text or "") + (img.structured_content or "") + (img.vision_caption or "")
    res = T.clean_image_texts([img])
    after = (img.ocr_text or "") + (img.structured_content or "") + (img.vision_caption or "")
    wanted = list(case.get("image_must_contain") or [])
    kept = [w for w in wanted if T._present(w, after)]
    return {
        "id": case["id"],
        "noise_rate_before": round(_noise_rate(before), 5),
        "noise_rate_after": round(_noise_rate(after), 5),
        "facts_total": len(wanted),
        "facts_kept": len(kept),
        "facts_rate": round(len(kept) / max(len(wanted), 1), 4),
        "touched": res.touched,
        "masked_paragraphs": res.masked_paragraphs,
    }


def run_llm(case: dict, limit: int) -> dict | None:
    if not T._ollama_reachable():
        return {"skipped": True, "reason": f"Ollama 不可达（{T._OLLAMA}）"}
    spec = case.get("llm") or {}
    result, _, _ = T._run_clean(case)
    user = f"文档片段：\n{result.full_text[:4000]}\n\n问题：{spec.get('question')}"
    t0 = time.time()
    ok, value = T._with_deadline(lambda: T.call_llm(T._LLM_SYSTEM, user), 180.0, "llm")
    elapsed = round(time.time() - t0, 1)
    if not ok:
        return {"passed": False, "elapsed": elapsed, "detail": f"失败/超时 {value}"}
    content, backend = value
    content = (content or "").strip()
    missing = [t for t in (spec.get("must_mention") or []) if t not in content]
    leaked = [t for t in (spec.get("must_not_mention") or []) if t in content]
    too_short = len(content) < int(spec.get("min_content_chars") or 8)
    return {
        "backend": backend,
        "elapsed": elapsed,
        "content_chars": len(content),
        "content_head": content[:80],
        "missing": missing,
        "leaked": leaked,
        "too_short": too_short,
        "passed": not (missing or leaked or too_short),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="跳过 LLM 真链路")
    ap.add_argument("--llm-limit", type=int, default=int(os.environ.get("GOLDEN_LLM_LIMIT", "5")))
    ap.add_argument("--llm-all", action="store_true", help="对全部金标跑 LLM（慢）")
    ap.add_argument(
        "--out",
        default=str(Path(r"D:\RAG\Production-RAG-Agent\_golden_eval.txt")),
        help="跑分报告输出路径",
    )
    args = ap.parse_args()

    cases = T.CASES
    rows = [evaluate_case(c) for c in cases]
    image_rows = [r for r in (evaluate_image_channel(c) for c in cases) if r]

    lines: list[str] = []
    lines.append("=== 清洗能力金标跑分 ===")
    lines.append(f"金标条数：{len(cases)}   语料文件：{len(list((_TESTS / 'golden' / 'corpus').glob('*')))}")
    lines.append(
        "判定助手来源：tests/test_golden_cleaning.py（同一份金标、同一套判定，避免漂移）"
    )
    lines.append("")

    # ── 逐条 ──────────────────────────────────────────────────────────────
    lines.append("── 逐条金标 ─────────────────────────────────────────────")
    lines.append(
        f"{'id':<14}{'格式':<7}{'噪声前':>8}{'噪声后':>8}{'保真':>7}"
        f"{'事实':>7}{'页图':>6}{'屏蔽':>6}  判定"
    )
    failures: list[str] = []
    for r in rows:
        verdict = []
        if r["noise_rate_after"] > 0:
            verdict.append(f"噪声残留{r['noise_rate_after']:.4f}")
        if not r["retention_ok"]:
            verdict.append(f"保真{r['retention']}<阈值")
        if r["facts_kept"] != r["facts_total"]:
            verdict.append(f"丢事实{r['facts_total'] - r['facts_kept']}项")
        if r["preserved_kept"] != r["preserved_total"]:
            verdict.append(f"语义破坏{r['preserved_total'] - r['preserved_kept']}处")
        if not r["pagemap_ok"]:
            verdict.append("页偏移不自洽")
        if not r["masking_ok"]:
            verdict.append(f"屏蔽{result_masked(r)}≠期望")
        if r["injection_leaked"]:
            verdict.append("注入载荷残留")
        ok = not verdict
        if not ok:
            failures.append(f"{r['id']}: " + "；".join(verdict))
        lines.append(
            f"{r['id']:<14}{r['format']:<7}{r['noise_rate_before']:>8.4f}"
            f"{r['noise_rate_after']:>8.4f}{r['retention']:>7.3f}"
            f"{r['facts_rate']:>7.2f}{'OK' if r['pagemap_ok'] else 'BAD':>6}"
            f"{'OK' if r['masking_ok'] else 'BAD':>6}  {'PASS' if ok else 'FAIL'}"
        )

    # ── 图片通道 ──────────────────────────────────────────────────────────
    if image_rows:
        lines.append("")
        lines.append("── 图片通道（ocr_text / structured_content / vision_caption）──")
        for r in image_rows:
            lines.append(
                f"  {r['id']:<14} 噪声 {r['noise_rate_before']:.4f}→{r['noise_rate_after']:.4f}"
                f"   事实保全 {r['facts_kept']}/{r['facts_total']}"
                f"   touched={r['touched']} masked={r['masked_paragraphs']}"
            )

    # ── 按格式聚合 ────────────────────────────────────────────────────────
    lines.append("")
    lines.append("── 按格式聚合（清洗前 → 清洗后）──────────────────────────")
    lines.append(
        f"{'格式':<7}{'噪声残留率':>18}{'正文保真率':>12}{'事实保全率':>12}"
        f"{'干净chunk占比':>20}{'页偏移':>8}"
    )
    by_format: dict[str, list[dict]] = {}
    for r in rows:
        by_format.setdefault(r["format"], []).append(r)
    for fmt in sorted(by_format):
        group = by_format[fmt]
        avg = lambda key: sum(g[key] for g in group) / len(group)  # noqa: E731
        pm = "OK" if all(g["pagemap_ok"] for g in group) else "BAD"
        lines.append(
            f"{fmt:<7}"
            f"{avg('noise_rate_before'):>9.4f} →{avg('noise_rate_after'):<8.4f}"
            f"{avg('retention'):>12.4f}"
            f"{avg('facts_rate'):>12.4f}"
            f"{avg('clean_chunk_rate_before'):>10.4f} →{avg('clean_chunk_rate_after'):<8.4f}"
            f"{pm:>8}"
        )

    # ── LLM 真链路 ────────────────────────────────────────────────────────
    lines.append("")
    lines.append("── 清洗 → 本机 qwen3:8b（真链路）────────────────────────")
    if args.no_llm:
        lines.append("  （--no-llm：本次未跑）")
    else:
        selected = cases if args.llm_all else (
            [c for c in cases if c.get("noise_injected")] + cases
        )[: args.llm_limit]
        passed = 0
        for case in selected:
            r = run_llm(case, args.llm_limit)
            if r.get("skipped"):
                lines.append(f"  未验证：{r['reason']}")
                break
            if r.get("passed"):
                passed += 1
                lines.append(
                    f"  PASS {case['id']:<14} [{r['backend']}] {r['elapsed']}s "
                    f"→ {r['content_head']!r}"
                )
            else:
                detail = []
                if r.get("too_short"):
                    detail.append(f"内容过短({r.get('content_chars')}字)")
                if r.get("missing"):
                    detail.append(f"答不出{r['missing']}")
                if r.get("leaked"):
                    detail.append(f"泄漏屏蔽内容{r['leaked']}")
                detail.append(str(r.get("detail") or ""))
                lines.append(f"  FAIL {case['id']:<14} {'; '.join(x for x in detail if x)}")
        lines.append(f"  LLM 通过 {passed}/{len(selected)}")

    # ── 总评 ──────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("── 总评 ─────────────────────────────────────────────────")
    if failures:
        lines.append(f"FAILED {len(failures)}/{len(rows)}：")
        lines.extend(f"  - {f}" for f in failures)
    else:
        lines.append(f"ALL PASS {len(rows)}/{len(rows)} 条金标")
    lines.append("")
    lines.append(
        "注：chunk 指标是 MAX_CHUNK_SIZE=2000 / OVERLAP=200 固定窗口下的**代理指标**，"
        "环境未安装 ragas，本报告不含 ragas 分数。"
    )

    report = "\n".join(lines)
    out_path = Path(args.out)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 1 if failures else 0


def result_masked(r: dict) -> int:
    return int(r.get("masked_paragraphs") or 0)


if __name__ == "__main__":
    sys.exit(main())
