"""
清洗能力金标测试（golden cleaning）.

与 ``tests/test_text_cleaning.py``（单规则单测）的分工：

    单测    —— 一条规则一个用例，证明"规则写对了"；
    本套件  —— **一份文档一条金标**，证明"真实文档跑完清洗后仍可用"，
               并把最后一步真正接上本机 qwen3:8b，断言 **LLM 还能从清洗后的
               文本里答出事实**。清洗把数字洗没了 / 把代码块洗坏了，单测可能
               全绿，但 LLM 这一关会红 —— 这是金标唯一不可替代的价值。

金标文件由 ``tests/golden/make_golden_fixtures.py`` 生成（纯标准库，可复现），
期望写在 ``tests/golden/expectations.json``，是**不变量**而不是"跑一遍抄下来的
实际输出"（那样就同义反复、永远绿）。

运行：
    # 容器内（推荐，依赖齐全）
    docker exec -w /app -e PYTHONPATH=/app rag_backend \
        python -m pytest tests/test_golden_cleaning.py -q
    # 宿主机直跑（无 pytest 也能跑，见文件末尾 main()）
    python backend/tests/test_golden_cleaning.py

环境变量：
    GOLDEN_LLM=1        打开"清洗后真喂 qwen3:8b"（默认开，Ollama 不可达则跳过）
    GOLDEN_LLM_LIMIT=N  最多喂几条金标（默认 5，CPU 推理 ~33s/条）
    GOLDEN_LLM_ATTEMPTS=N  LLM 调用的退避重试次数（默认 3）

                        ⚠️ 只对「抛异常 / 超时」重试。"返回内容为空 / 过短"
                        **不重试** —— 那是 ``reasoning=True`` 吃光 ``num_predict``
                        的症状，属于本套件要抓的缺陷，重试糊过去等于拆掉防线。
                        目的：本机 Ollama 在"前一批用例刚打出大量真实请求"之后
                        会偶发返回空体错误（实测 ``responseError('')``，5 条金标
                        全中；几分钟后同一配对 3/3 全绿）。一次失败就断言
                        "清洗把事实洗没了"属于**错误归因**，会把人往错方向带。
    GOLDEN_BINARY=1     打开"二进制样张能被真实解析器打开"（默认关，见下注）
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_CORPUS = _GOLDEN_DIR / "corpus"
_EXPECT = _GOLDEN_DIR / "expectations.json"


# ─────────────────────────────────────────────────────────────────────────────
# 模块加载：绕开 app/services/__init__.py 的重依赖（sqlalchemy / fastapi / torch）
#
# 既有 tests/test_text_cleaning.py 在宿主机上跑不起来，就是因为
# `from app.services.text_controls import ...` 会触发 app.services.__init__ →
# document_service → sqlalchemy。这里预先塞入**空壳父包**，让子模块的绝对导入
# 命中缓存而不去执行 __init__ —— 于是本套件在零第三方依赖的环境里也能真跑。
# ─────────────────────────────────────────────────────────────────────────────


def _stub_package(name: str) -> types.ModuleType:
    """塞入空壳父包：子模块仍从**真实目录**加载，但不执行 ``__init__``（避开重依赖）。

    两个关键点都是实测踩出来的，缺一个就会"看着通过其实没跑"或打崩别的套件：

    1. ``__path__`` 必须指向真实目录，**不能是 ``[]``** —— 否则同会话里后续
       ``import app.services.image_understanding.*`` / ``app.utils.logging`` 这类
       绝对导入会找不到文件；而调用方普遍用 ``try/except ImportError`` 兜底，
       于是变成**静默跳过**（实测让 test_image_noise_quality / _polarity /
       _position 三个文件在组合运行下整体不收集）。
    2. 必须把空壳挂到父模块属性上（``app.services`` / ``app.utils``）—— 否则
       pytest 字符串式 monkeypatch ``app.services.x.y`` 的解析链会在
       ``getattr(app, "services")`` 处断掉（实测打断
       test_citation_open_recheck.py 的 10 个用例）。
    """
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    sub = name.split(".")[1:]
    mod.__path__ = [str(_BACKEND_ROOT.joinpath("app", *sub))]
    sys.modules[name] = mod
    if "." in name:
        parent_name, child = name.rsplit(".", 1)
        setattr(_stub_package(parent_name), child, mod)
    return mod


def _load_by_path(dotted: str, filename: str):
    if dotted in sys.modules:
        return sys.modules[dotted]
    parent = dotted.rsplit(".", 1)[0]
    _stub_package(parent)
    spec = importlib.util.spec_from_file_location(
        dotted, _BACKEND_ROOT / "app" / "services" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


for _pkg in ("app", "app.services", "app.utils"):
    _stub_package(_pkg)

# ``app.utils.logging`` 是纯标准库模块（setup_logging / get_logger），宿主机可以
# **真导入**。绝不要用桩顶替它：桩里没有 ``setup_logging``，一旦留在 sys.modules，
# 同会话后跑的 ``app/main.py`` 会在 ``from app.utils.logging import setup_logging``
# 处直接 ImportError（实测打崩 test_reliability_guards.py）。
if "app.utils.logging" not in sys.modules:
    try:
        importlib.import_module("app.utils.logging")
    except ImportError:
        _logging_stub = types.ModuleType("app.utils.logging")
        _logging_stub.get_logger = lambda name: __import__("logging").getLogger(name)
        sys.modules["app.utils.logging"] = _logging_stub

_load_by_path("app.services.text_controls", "text_controls.py")
_load_by_path("app.services.prompt_security", "prompt_security.py")
_cleaning = _load_by_path("app.services.text_cleaning", "text_cleaning.py")

clean_text = _cleaning.clean_text
clean_and_mask = _cleaning.clean_and_mask
clean_extraction = _cleaning.clean_extraction
clean_image_texts = _cleaning.clean_image_texts


# ── 跳过助手（pytest 下用 pytest.skip，直跑时落到 _SKIPPED 记录）────────────

_SKIPPED: list[tuple[str, str]] = []


class _StandaloneSkip(Exception):
    pass


def _skip(reason: str) -> None:
    try:
        import pytest

        pytest.skip(reason)
    except ImportError:
        raise _StandaloneSkip(reason) from None


# ─────────────────────────────────────────────────────────────────────────────
# 金标装载
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FakePage:
    """duck-typing 版的 ExtractedPage（clean_extraction 只认三个属性）。"""

    page_number: int
    text: str
    char_start: int
    char_end: int


@dataclass
class FakeImage:
    ocr_text: str = ""
    structured_content: str = ""
    vision_caption: str | None = None


def _load_cases() -> list[dict]:
    if not _EXPECT.exists():
        _skip(f"金标文件缺失：{_EXPECT}（先跑 golden/make_golden_fixtures.py）")
    data = json.loads(_EXPECT.read_text(encoding="utf-8"))
    return list(data.get("cases") or [])


CASES = _load_cases()


def _build_doc(page_texts: list[str]) -> tuple[list[FakePage], str]:
    """按解析器的真实做法拼文档：页间用 "\\n\\n" 分隔，偏移据此计算。"""
    pages: list[FakePage] = []
    cursor = 0
    for i, text in enumerate(page_texts, start=1):
        start, end = cursor, cursor + len(text)
        pages.append(FakePage(i, text, start, end))
        cursor = end + 2
    return pages, "\n\n".join(page_texts)


def _run_clean(case: dict):
    pages, full_text = _build_doc(case["pages"])
    return clean_extraction(pages, full_text), pages, full_text


def _norm(text: str) -> str:
    """折叠连续空白 —— 用于"内容还在不在"的判定。

    为什么需要它：清洗会把 NBSP / 全角空格**合法地**换成半角空格，于是
    "| 指标 |" 可能变成 "|␣␣指标 |"。要求逐字节相等会把这类合法替换误判成
    "内容丢了"（假红）；真正要守的是"token 与顺序都还在"。
    """
    return " ".join((text or "").split())


def _present(token: str, text: str) -> bool:
    """命中判定：原始匹配，或空白归一化后匹配."""
    return token in (text or "") or _norm(token) in _norm(text)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 语料与金标本身的自洽
# ─────────────────────────────────────────────────────────────────────────────


def test_golden_corpus_is_complete() -> None:
    """每条金标都要有对应的真实文件，且文件非空 —— 断链的金标等于没测."""
    problems = []
    for case in CASES:
        path = _CORPUS / f"{case['id']}.{case['format']}"
        if not path.exists():
            problems.append(f"{case['id']}: 缺文件 {path.name}")
        elif path.stat().st_size == 0:
            problems.append(f"{case['id']}: 空文件")
    assert not problems, "金标语料不完整:\n  " + "\n  ".join(problems)
    assert len(CASES) >= 16, f"金标条数不足（{len(CASES)}），应覆盖 8 格式 × 干净/脏"
    print(f"  ok 语料完整：{len(CASES)} 条金标 / {len(list(_CORPUS.glob('*')))} 个文件")


# ─────────────────────────────────────────────────────────────────────────────
# 2–5. 清洗不变量（逐条金标）
# ─────────────────────────────────────────────────────────────────────────────


def test_noise_is_removed_in_every_golden() -> None:
    """该清的必须清掉：BOM / 零宽 / 软连字符 / 全角空格 / NBSP / C0 / CR."""
    problems = []
    for case in CASES:
        result, _, _ = _run_clean(case)
        for token in case["must_not_contain"]:
            if token in result.full_text:
                problems.append(
                    f"{case['id']}: 残留 {token!r}（U+{ord(token[0]):04X}）"
                )
    assert not problems, "清洗未清干净:\n  " + "\n  ".join(problems)
    print(f"  ok 噪声清零：{len(CASES)} 条金标")


def test_facts_and_semantics_survive_cleaning() -> None:
    """不该动的必须还在：事实数字、表头、代码缩进、全角标点."""
    problems = []
    for case in CASES:
        result, _, _ = _run_clean(case)
        for token in case["must_contain"]:
            if not _present(token, result.full_text):
                problems.append(f"{case['id']}: 丢了 {token!r}")
        for token in case.get("must_be_preserved") or []:
            if token not in result.full_text:
                problems.append(f"{case['id']}: 语义片段被破坏 {token!r}")
        if "，" in "\n\n".join(case["pages"]):
            # 全角逗号必须原样保留（不做 NFKC 折叠 —— 那是格式损失）
            if "，" not in result.full_text:
                problems.append(f"{case['id']}: 全角标点被折叠")
    assert not problems, "清洗伤了内容:\n  " + "\n  ".join(problems)
    print(f"  ok 事实与语义保全：{len(CASES)} 条金标")


def test_cleaning_does_not_eat_the_body() -> None:
    """清洗只剥噪声，不能把正文吃掉（最低保真率由金标给定）."""
    problems = []
    ratios = {}
    for case in CASES:
        result, _, full_text = _run_clean(case)
        original = len(full_text) or 1
        ratio = len(result.full_text) / original
        ratios[case["id"]] = round(ratio, 4)
        floor = float(case.get("min_retention_ratio") or 0.8)
        if ratio < floor:
            problems.append(f"{case['id']}: 保真率 {ratio:.3f} < {floor}")
        if ratio < 0.5:
            problems.append(f"{case['id']}: 保真率 {ratio:.3f} —— 正文被大量吃掉")
    print(f"  · 保真率 {ratios}")
    assert not problems, "正文被清洗吃掉:\n  " + "\n  ".join(problems)
    print(f"  ok 正文保真：{len(CASES)} 条金标全部 ≥ 阈值")


def test_page_map_intact_and_spans_self_consistent() -> None:
    """
    页偏移必须重算且自洽.

    引用卡片上的"第 N 页"由 page_for_offset(char_start/char_end) 算出来；
    清洗改变长度却不重算 → 页码整体漂移。这里对**每条金标**做反查。
    """
    problems = []
    for case in CASES:
        result, pages, _ = _run_clean(case)
        if bool(case.get("expect_pagemap_intact")) and not result.pagemap_intact:
            problems.append(f"{case['id']}: pagemap_intact=False（页区间被判不可信）")
            continue
        if len(result.page_spans) != len(pages):
            problems.append(f"{case['id']}: page_spans 数量与页数不符")
            continue
        cleaned_pages = [clean_and_mask(p.text)[0] for p in pages]
        for page, cleaned, (start, end) in zip(pages, cleaned_pages, result.page_spans):
            if result.full_text[start:end] != cleaned:
                problems.append(f"{case['id']}: 第 {page.page_number} 页区间框不住清洗后文本")
        # 反查：每页首/末字符都必须命中本页
        for idx, (start, end) in enumerate(result.page_spans, start=1):
            hit = [i for i, (s, e) in enumerate(result.page_spans, start=1) if s <= start < e]
            if hit != [idx]:
                problems.append(f"{case['id']}: 第 {idx} 页首字符反查到 {hit}")
    assert not problems, "页偏移不自洽:\n  " + "\n  ".join(problems)
    print(f"  ok 页偏移自洽：{len(CASES)} 条金标")


def test_injection_masked_per_golden() -> None:
    """注入段落必须被屏蔽，且屏蔽占位符可见（不是静默删除）."""
    problems = []
    for case in CASES:
        result, _, _ = _run_clean(case)
        expect = int(case.get("expect_masked_paragraphs") or 0)
        if result.masked_paragraphs != expect:
            problems.append(
                f"{case['id']}: 屏蔽段数 {result.masked_paragraphs} != 期望 {expect}"
            )
        if expect and "已屏蔽" not in result.full_text:
            problems.append(f"{case['id']}: 屏蔽后没有可见占位符")
    assert not problems, "注入屏蔽不符:\n  " + "\n  ".join(problems)
    print(f"  ok 注入屏蔽：{len(CASES)} 条金标")


def test_image_channel_golden() -> None:
    """图片通道（ocr_text / structured_content / vision_caption）同强度清洗."""
    image_cases = [c for c in CASES if c.get("images")]
    assert image_cases, "金标里没有图片用例"
    problems = []
    for case in image_cases:
        for payload in case["images"]:
            img = FakeImage(
                ocr_text=payload.get("ocr_text", ""),
                structured_content=payload.get("structured_content", ""),
                vision_caption=payload.get("vision_caption"),
            )
            res = clean_image_texts([img])
            joined = (img.ocr_text or "") + (img.structured_content or "") + (img.vision_caption or "")
            for token in ("\ufeff", "\u200b", "\u00ad", "\u3000", "\u00a0", "\x07"):
                if token in joined:
                    problems.append(f"{case['id']}: 图片文本残留 {token!r}")
            for token in case.get("image_must_contain") or []:
                if not _present(token, joined):
                    problems.append(f"{case['id']}: 图片通道丢了 {token!r}")
            # 只有真被改写过的图片才计入 touched：干净用例本就不该 touched
            expect_touched = 1 if case.get("noise_injected") else 0
            if res.touched != expect_touched:
                problems.append(f"{case['id']}: touched={res.touched}（应为 {expect_touched}）")
    assert not problems, "图片通道清洗不符:\n  " + "\n  ".join(problems)
    print(f"  ok 图片通道：{len(image_cases)} 条金标")


def test_cleaning_is_idempotent_on_goldens() -> None:
    """
    同一份文档洗两次必须一致 —— 否则续传重试会产出不同的入库文本.

    ⚠️ 第二遍必须**用第一遍重算出来的页偏移**重建 pages。直接把旧 pages 配
    新 full_text 传进去是无效对照：旧 char_end 与新文本长度对不上，尾部会多切
    一段出来，差异来自测试的错配而不是清洗不幂等。
    """
    problems = []
    for case in CASES:
        first, pages, _ = _run_clean(case)
        rebuilt = [
            FakePage(
                page.page_number,
                first.full_text[start:end],
                start,
                end,
            )
            for page, (start, end) in zip(pages, first.page_spans)
        ]
        second = clean_extraction(rebuilt, first.full_text)
        if second.full_text != first.full_text:
            problems.append(
                f"{case['id']}: 二次清洗结果不同"
                f"（{len(first.full_text)} → {len(second.full_text)} 字）"
            )
    assert not problems, "清洗不幂等:\n  " + "\n  ".join(problems)
    print(f"  ok 幂等：{len(CASES)} 条金标")


# ─────────────────────────────────────────────────────────────────────────────
# 6. 高危形状：长文档里的真表头 / 正文，不得被"疑似页眉"逻辑误杀
#
#    这是"把表头当页眉删掉"的根形状：短行在多数页重复出现 → 被判页眉；
#    而表格表头恰好也是"短、成行、多列"。本用例断言**真表头与真正文都在**。
# ─────────────────────────────────────────────────────────────────────────────


def test_long_document_real_table_and_body_survive() -> None:
    """长文档（60 页）里重复出现的短行不得带走真表头 / 真正文."""
    repeated_short = "内部资料  请勿外传"           # 每页都有 → 最像页眉
    header_row = "| 指标 | 2023年 | 2024年 | 同比 |"
    body_row = "| 营业收入 | 30240 | 33500 | +10.8% |"
    pages_text: list[str] = []
    for i in range(60):
        if i == 30:                                  # 第 31 页放真表格
            pages_text.append(
                f"{repeated_short}\n\n表 3  2024 年度核心经营指标\n\n{header_row}\n"
                f"| --- | --- | --- | --- |\n{body_row}\n"
            )
        else:
            pages_text.append(f"{repeated_short}\n\n第 {i + 1} 页正文，说明第 {i + 1} 项工作的进展。")
    pages, full_text = _build_doc(pages_text)
    result = clean_extraction(pages, full_text)

    assert result.pagemap_intact, "长文档页区间应可信"
    assert header_row in result.full_text, "真表头被误删（把表头当页眉的根形状）"
    assert body_row in result.full_text, "表格数据行被误删"
    assert "表 3  2024 年度核心经营指标" in result.full_text, "题注被误删"
    assert result.full_text.count(repeated_short) == 60, (
        f"疑似页眉行数量变了（{result.full_text.count(repeated_short)} != 60）—— "
        "若将来引入页眉剥离，这里就是它的回归闸门"
    )
    assert len(result.full_text) > len(full_text) * 0.9, "长文档正文被吃掉超过 10%"
    print("  ok 长文档：60 页，真表头/数据行/题注全部幸存")


# ─────────────────────────────────────────────────────────────────────────────
# 7. 高危探针：`_remove_repeated` 的终止性 / 页眉页脚阈值一致性
#
#    ⚠️ 截至本次排查，text_cleaning.py 里**不存在** clean_pdf_text / _remove_repeated /
#       页眉页脚剥离（模块只有 clean_text / fold_inline_spaces / clean_and_mask /
#       clean_extraction / clean_image_texts 五个公开函数）。因此这两个用例是
#       **探针**：符号一旦出现就自动生效（带 deadline 保护），当前快照下
#       明确 SKIP 并注明"代码不存在，未验证" —— 不伪造绿。
# ─────────────────────────────────────────────────────────────────────────────


def _probe_symbols() -> dict:
    return {
        name: getattr(_cleaning, name, None)
        for name in ("clean_pdf_text", "_remove_repeated", "strip_header_footer",
                     "remove_repeated_lines", "_detect_repeated")
    }


def _with_deadline(fn, seconds: float, label: str):
    """
    在带 deadline 的**守护线程**里跑 fn.

    为什么不用 pytest-timeout：它在 Windows 上靠 signal 实现不可靠；而这里要防的
    正是 `while True` 型死循环——CPU 密集的死循环线程**杀不掉**，只能靠守护线程
    + join(timeout) 让主线程按时返回并判定 DEADLINE_EXCEEDED。
    """
    box: dict = {}

    def runner() -> None:
        try:
            box["ok"] = True
            box["value"] = fn()
        except Exception as exc:      # noqa: BLE001
            box["ok"] = False
            box["error"] = repr(exc)

    thread = threading.Thread(target=runner, daemon=True, name=label)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        return False, f"DEADLINE_EXCEEDED({seconds}s)"
    if not box.get("ok"):
        return False, box.get("error", "unknown")
    return True, box.get("value")


def test_repeated_line_removal_terminates_under_deadline() -> None:
    """
    `_remove_repeated` 必须在 deadline 内返回（防 `while ... and ...` 死循环）.

    靶子形状：极端模式下 `max_rounds=None` 时，"上限"只是每次 sub 的次数而不
    是循环终止条件 → 输入里重复行一多就挂死。本用例用 5 秒 deadline 证伪。
    """
    syms = _probe_symbols()
    target = syms["_remove_repeated"] or syms["remove_repeated_lines"]
    if target is None:
        _skip(
            "未验证：`_remove_repeated` / `remove_repeated_lines` 在当前 "
            "text_cleaning.py 中不存在（模块仅 5 个公开函数）。代码一旦新增，"
            "本用例自动生效。"
        )
        return

    payload = ("重复行 A\n" * 400) + "真正文：营业收入 33500 万元。\n" + ("重复行 B\n" * 400)
    ok, value = _with_deadline(lambda: target(payload), 5.0, "remove_repeated")
    assert ok, f"`_remove_repeated` 未能在 5s 内返回 → 死循环（{value}）"
    print(f"  ok `_remove_repeated` 在 deadline 内返回：{str(value)[:80]}")


def test_header_footer_threshold_consistency() -> None:
    """
    页眉/页脚判定的阈值必须只有一处真源，且多轮剥离后页序与 page_spans 一致.

    靶子形状：0.6 / 0.5 与 >=0.7 两个矛盾的阈值并存 → 默认路径走哪个不确定；
    第三轮结果被 `[:2]` 静默截断 → 页序与 page_spans 错位。
    """
    syms = _probe_symbols()
    if syms["clean_pdf_text"] is None and syms["strip_header_footer"] is None:
        _skip(
            "未验证：`clean_pdf_text` / `strip_header_footer` 在当前 "
            "text_cleaning.py 中不存在（无页眉页脚剥离逻辑）。代码一旦新增，"
            "本用例自动生效。"
        )
        return

    pages_text = [
        "页眉行 2024\n第 1 页正文，营业收入 33500 万元。\n页脚行 第 1 页",
        "页眉行 2024\n第 2 页正文，净利润 2460 万元。\n页脚行 第 2 页",
        "页眉行 2024\n第 3 页正文，毛利率 25.0%。\n页脚行 第 3 页",
    ]
    pages, full_text = _build_doc(pages_text)
    fn = syms["clean_pdf_text"] or syms["strip_header_footer"]
    ok, value = _with_deadline(lambda: fn(pages_text) if fn is not None else None,
                               5.0, "header_footer")
    assert ok, f"页眉页脚剥离未能在 5s 内返回（{value}）"
    # 页序一致性：输出页数必须与输入页数相同（不允许被 [:2] 静默截断）
    out_pages = value if isinstance(value, list) else None
    if out_pages is not None:
        assert len(out_pages) == len(pages_text), (
            f"剥离后页数 {len(out_pages)} != 输入页数 {len(pages_text)}（被静默截断）"
        )
    print("  ok 页眉页脚阈值：页数一致、在 deadline 内返回")


# ─────────────────────────────────────────────────────────────────────────────
# 8. 真链路：清洗后的文本 → 本机 qwen3:8b → 事实仍可答
# ─────────────────────────────────────────────────────────────────────────────

_OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
_NUM_GPU = os.environ.get("OLLAMA_NUM_GPU", "0")
_NUM_PREDICT = int(os.environ.get("GOLDEN_LLM_NUM_PREDICT", "512"))

#: LLM 调用退避重试次数。本机 Ollama 在"前一批用例刚打出大量真实请求"之后会偶发
#: 返回空体错误（实测 ``responseError('')``，本次 5 条金标全中；而同样配对几分钟后
#: 再跑 3/3 全绿）。一次失败就断言"清洗把事实洗没了"属于**错误归因**，故加重试。
#:
#: ⚠️ 只对「抛异常 / 超时」重试；「返回内容为空 / 过短」**绝不重试** —— 那正是
#: ``reasoning=True`` 吃光 ``num_predict`` 的症状，是本用例要抓的缺陷之一，
#: 用重试糊过去就等于自己把这条防线拆了。
_LLM_ATTEMPTS = int(os.environ.get("GOLDEN_LLM_ATTEMPTS", "3"))


def _ollama_reachable() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{_OLLAMA}/api/tags", timeout=5) as resp:
            return resp.status == 200
    except Exception:      # noqa: BLE001
        return False


def _call_llm_via_langchain(system: str, user: str) -> str:
    """首选：langchain 包着的 ChatOllama（与生产同一条调用链）。

    ``reasoning=False`` 是硬要求：本机 qwen3:8b 开着思考会把 num_predict 吃光，
    实测 content 变空串（且 92.1s vs 32.7s）。``keep_alive`` 由 langchain 默认
    带上，避免模型 5 分钟空闲后被卸载、下一张图多等约 6.7s 冷启动。
    """
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=_MODEL,
        base_url=_OLLAMA,
        reasoning=False,                    # ← 本机必关
        num_predict=_NUM_PREDICT,
        num_ctx=_NUM_CTX,
        num_gpu=(int(_NUM_GPU) if _NUM_GPU not in ("", "None") else None),
        temperature=0.2,
    )
    return str((llm.invoke(f"{system}\n\n{user}")).content or "")


def _call_llm_via_rest(system: str, user: str) -> str:
    """回退：langchain 不可用时直连 Ollama /api/chat（参数与上面等价）。"""
    import json as _json
    import urllib.request

    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,                     # ← 同上，必须关
        "keep_alive": "10m",
        "options": {
            "temperature": 0.2,
            "num_ctx": _NUM_CTX,
            "num_predict": _NUM_PREDICT,
            "num_gpu": (int(_NUM_GPU) if _NUM_GPU not in ("", "None") else 0),
        },
    }
    req = urllib.request.Request(
        f"{_OLLAMA}/api/chat",
        data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    return str((data.get("message") or {}).get("content") or "")


def call_llm(system: str, user: str) -> tuple[str, str]:
    """返回 (content, backend)；backend 用于如实标注这条证据是怎么拿到的。"""
    try:
        import langchain_ollama  # noqa: F401
    except ImportError:
        return _call_llm_via_rest(system, user), "rest(/api/chat)"
    return _call_llm_via_langchain(system, user), "langchain(ChatOllama)"


_LLM_SYSTEM = (
    "你是企业知识库助手。只能依据下面给出的文档片段作答，不要编造。"
    "用一句中文直接回答，不要任何前缀或解释。"
)


def test_llm_answers_facts_from_cleaned_text() -> None:
    """
    清洗 → qwen3:8b 的**真链路**断言.

    清洗若把数字洗没了、把表格洗成乱码，前面的单条规则断言可能仍然全绿
    （规则都是"局部正确"的），但 LLM 这一关会红 —— 这是金标不可替代的价值。
    """
    if os.environ.get("GOLDEN_LLM", "1") != "1":
        _skip("GOLDEN_LLM=0：已显式关闭 LLM 链路用例")
        return
    if not _ollama_reachable():
        _skip(f"未验证：Ollama 不可达（{_OLLAMA}）—— 无法真喂 qwen3:8b")
        return

    limit = int(os.environ.get("GOLDEN_LLM_LIMIT", "5"))
    dirty = [c for c in CASES if c.get("noise_injected")]
    selected = (dirty + [c for c in CASES if not c.get("noise_injected")])[:limit]

    problems: list[str] = []            # 内容级：清洗把事实洗没了 / 注入屏蔽失效
    transport_failures: list[str] = []  # 传输级：压根没从 LLM 拿到回答

    for case in selected:
        result, _, _ = _run_clean(case)
        spec = case.get("llm") or {}
        user = f"文档片段：\n{result.full_text[:4000]}\n\n问题：{spec.get('question')}"

        content, backend, elapsed, call_error = "", "", 0.0, ""
        for attempt in range(_LLM_ATTEMPTS):
            t0 = time.time()
            ok, value = _with_deadline(
                lambda: call_llm(_LLM_SYSTEM, user), 180.0, f"llm-{case['id']}"
            )
            elapsed = time.time() - t0
            if not ok:                   # 异常 / DEADLINE_EXCEEDED → 退避重试
                call_error = str(value)
                if attempt < _LLM_ATTEMPTS - 1:
                    time.sleep(2.0 * (attempt + 1))
                continue
            content, backend = value
            content = (content or "").strip()
            call_error = ""
            break                        # 拿到响应了（哪怕内容为空）→ 不再重试

        if call_error:
            transport_failures.append(
                f"{case['id']}: 调用失败/超时（{call_error}）— 退避重试 "
                f"{_LLM_ATTEMPTS} 次仍拿不到响应"
            )
            continue

        if len(content) < int(spec.get("min_content_chars") or 8):
            # ★ 不重试、不跳过：这正是 reasoning=True 吃光 num_predict 的症状，
            #   属于要抓的缺陷；当成"环境问题"跳过才是真的掩盖。
            problems.append(
                f"{case['id']}: LLM 返回空/过短（{len(content)} 字，backend={backend}）—— "
                "典型症状是 reasoning=True 吃光 num_predict"
            )
            continue

        for token in spec.get("must_mention") or []:
            if token not in content:
                problems.append(f"{case['id']}: LLM 答不出 {token!r}（清洗把事实洗没了？）")
        for token in spec.get("must_not_mention") or []:
            if token in content:
                problems.append(f"{case['id']}: LLM 输出里出现了已屏蔽的注入载荷")
        print(f"    · {case['id']} [{backend}] {elapsed:.1f}s → {content[:70]!r}")

    # ① 内容级问题优先报 —— 这些是真缺陷，报错文案要一眼看出是"清洗"而不是"环境"。
    assert not problems, (
        "清洗后 LLM 答不出事实（内容级 —— 真缺陷，非环境）:\n  " + "\n  ".join(problems)
    )

    # ② 传输级：一条都没问到 ⇒ 本次运行**什么都没验证到**，如实跳过而不是记成清洗缺陷；
    #    只挂了一部分 ⇒ 不像纯环境抖动，按缺陷报出来让人看。
    if transport_failures:
        if len(transport_failures) == len(selected):
            _skip(
                "未验证：本次运行 LLM 传输层全部失败，金标链路一条都没跑出来"
                "（环境原因，非清洗缺陷）—— " + "；".join(transport_failures)
            )
            return
        assert not transport_failures, (
            f"部分金标没拿到 LLM 回答（{len(transport_failures)}/{len(selected)}）"
            "—— 不像环境问题，需人工核查:\n  " + "\n  ".join(transport_failures)
        )

    print(f"  ok 清洗→qwen3:8b 真链路：{len(selected)} 条金标全部答出事实")


# ─────────────────────────────────────────────────────────────────────────────
# 8.5 归因分层的**自证**用例（传输级 vs 内容级）
#
#     为什么要有这一节：上面的 LLM 用例把"没拿到回答"分成了两类 —— 传输级
#     （环境，跳过）与内容级（真缺陷，失败）。这个分层本身必须被测试，否则
#     一旦有人把它改回"一律断言失败"，或者反过来一律跳过，都不会有人发现：
#     前者让 Ollama 打嗝长期伪装成清洗缺陷，后者把真正的缺陷静默豁免。
# ─────────────────────────────────────────────────────────────────────────────


def test_llm_case_classifies_transport_failure_as_skip(monkeypatch) -> None:
    """传输层**全部**失败 ⇒ 必须跳过（环境原因），不能报成"清洗把事实洗没了". """
    import pytest

    this = sys.modules[__name__]

    def _boom(_system, _user):          # noqa: ANN001
        raise RuntimeError("ResponseError('')")

    monkeypatch.setenv("GOLDEN_LLM", "1")
    monkeypatch.setenv("GOLDEN_LLM_LIMIT", "1")
    monkeypatch.setattr(this, "_LLM_ATTEMPTS", 1)
    monkeypatch.setattr(this, "_ollama_reachable", lambda: True)
    monkeypatch.setattr(this, "call_llm", _boom)

    with pytest.raises(pytest.skip.Exception) as ei:
        test_llm_answers_facts_from_cleaned_text()
    assert "传输层" in str(ei.value), str(ei.value)


def test_llm_case_flags_empty_content_as_defect(monkeypatch) -> None:
    """拿到 200 但内容为空 ⇒ 必须**失败**：这是 reasoning 吃光 num_predict 的症状.

    关键点：这种情况**不能**被重试或跳过糊过去 —— 那正是本用例存在的理由。
    """
    import pytest

    this = sys.modules[__name__]
    calls: list[int] = []

    def _empty(_system, _user):         # noqa: ANN001
        calls.append(1)
        return "", "langchain(ChatOllama)"

    monkeypatch.setenv("GOLDEN_LLM", "1")
    monkeypatch.setenv("GOLDEN_LLM_LIMIT", "1")
    monkeypatch.setattr(this, "_LLM_ATTEMPTS", 3)
    monkeypatch.setattr(this, "_ollama_reachable", lambda: True)
    monkeypatch.setattr(this, "call_llm", _empty)

    with pytest.raises(AssertionError) as ei:
        test_llm_answers_facts_from_cleaned_text()
    assert "内容级" in str(ei.value), str(ei.value)
    # 空内容属于内容级缺陷 → 不该触发重试（只调用 1 次）
    assert len(calls) == 1, f"空内容不应重试，实际调用了 {len(calls)} 次"


# ─────────────────────────────────────────────────────────────────────────────
# 9. 二进制样张能否被真实解析器打开（默认关闭，需 GOLDEN_BINARY=1）
#
#    默认关闭的原因：样张是本套件的生成器用**标准库**手写的 OOXML/PDF/PNG，
#    未经过 python-docx / openpyxl / python-pptx / PyMuPDF 的实际验证。
#    默认打开会把手写样张的缺陷误报成产品缺陷（假红）；显式开启后若失败，
#    需要先区分"样张问题"还是"解析器问题"。
# ─────────────────────────────────────────────────────────────────────────────


def test_binary_fixtures_parseable() -> None:
    if os.environ.get("GOLDEN_BINARY", "0") != "1":
        _skip("GOLDEN_BINARY!=1：二进制样张解析用例默认关闭（样张未经第三方库验证）")
        return
    try:
        from app.services.parsers.factory import get_parser_for_file
    except Exception as exc:      # noqa: BLE001
        _skip(f"未验证：解析器依赖不可导入（{exc}）")
        return

    problems = []
    for case in CASES:
        path = _CORPUS / f"{case['id']}.{case['format']}"
        parser = get_parser_for_file(path.name)
        try:
            res = parser.parse(path.read_bytes(), path.name)
        except Exception as exc:      # noqa: BLE001
            problems.append(f"{case['id']}: 解析抛异常 {exc!r}")
            continue
        if not (res.full_text or "").strip():
            problems.append(f"{case['id']}: 解析结果正文为空")
    assert not problems, "二进制样张解析失败:\n  " + "\n  ".join(problems)
    print(f"  ok 二进制样张解析：{len(CASES)} 条")


def main() -> None:
    """无 pytest 时直跑（宿主机零依赖可用）。"""
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    print(f"收集用例数：{len(tests)}")
    passed, failed, skipped = 0, 0, 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS {name}")
        except _StandaloneSkip as exc:
            skipped += 1
            print(f"SKIP {name} —— {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}\n     {exc}")
        except Exception as exc:      # noqa: BLE001
            failed += 1
            print(f"ERROR {name}\n     {exc!r}")
    print(f"\n通过 {passed} / 跳过 {skipped} / 失败 {failed} / 合计 {len(tests)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
