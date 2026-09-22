"""``scripts/check_eval_baseline_freshness.py``（检索质量基线输入对账门禁）的回归测试.

被锁的东西
──────────
``baseline_v1.json`` 里钉住的**输入指纹**必须与当前仓库实际值一致；一旦
「重导语料 / 换精排模型 / 动检索阈值」而基线未重钉 → 门禁必须**非 0**，并打印
可直接照做的修复步骤（重跑 ``run_eval_baseline.py`` + 按 §口径更新 baseline）。
本文件用**纯逻辑**把这条判据钉死：宿主机 / CI 可跑，不需要真实栈。

为什么这些用例不是摆设（防"恒真"）
──────────────────────────────────
每条"必须红"的用例都配一条阴性对照：不改输入 → 必绿；改一处 → 必红。
两者并排，证明判据**会区分**，而不是"永远通过"或"永远失败"。解析层另有
``_parse_dotenv`` 的行内注释用例 —— 本仓库 ``backend/.env`` 大量写
``KEY=value  # 说明``，解析错会把说明算进值里、把门禁假红。

    python -m pytest tests/test_eval_baseline_freshness.py -q
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _BACKEND_ROOT / "scripts" / "check_eval_baseline_freshness.py"
_REPO_ROOT = _BACKEND_ROOT.parent


def _load_module():
    spec = importlib.util.spec_from_file_location("_eval_freshness", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)      # 有 __main__ 守卫，import 不执行 main
    return module


freshness = _load_module()

BASELINE = freshness.DEFAULT_BASELINE
GOLDEN = freshness.DEFAULT_GOLDEN
ENV = freshness.DEFAULT_ENV


# ─────────────────────────────────────────────────────────────────────────────
# 夹具
# ─────────────────────────────────────────────────────────────────────────────


def _gather(*, golden_path: Path | None = None, env_path: Path | None = None) -> dict:
    return freshness.gather_current(
        baseline_path=BASELINE,
        golden_path=golden_path or GOLDEN,
        env_path=env_path or ENV,
    )


def _run(cur: dict):
    """按 ``gather_current`` 的返回调用 ``check_inputs``（签名是关键字专属）."""
    return freshness.check_inputs(
        baseline=cur["baseline"],
        golden=cur["golden"],
        config_defaults=cur["config_defaults"],
        config_effective=cur["config_effective"],
        env_present=cur["env_present"],
        models_defaults=cur["models_defaults"],
        models_effective=cur["models_effective"],
        documents=cur["documents"],
        thresholds_hash=cur["thresholds_hash"],
        wiring_ok=cur["wiring_ok"],
        ceiling=cur["ceiling"],
    )


def _write_golden(tmp_path: Path, mutate) -> Path:      # noqa: ANN001
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    mutate(golden)
    path = tmp_path / "golden_v1.json"
    path.write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ① 输入一致 → 通过（真仓库、真文件）
# ─────────────────────────────────────────────────────────────────────────────


def test_real_repo_inputs_are_consistent() -> None:
    """当前仓库的输入指纹必须与 baseline 相符 —— 否则基线已过期、CI 门禁应红."""
    cur = _gather()
    ok, drifts, _notes, _detail = _run(cur)
    assert ok is True, f"当前仓库输入与基线指纹不一致：{drifts}"


def test_main_returns_ok_on_consistent_repo() -> None:
    """``main([])`` 在输入一致时退出码为 0（CI 门禁契约）."""
    assert freshness.main([]) == freshness.EXIT_OK


# ─────────────────────────────────────────────────────────────────────────────
# ② 改了检索旋钮 → 必须红（默认值路径 + .env 路径）
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "field, new_value",
    [
        ("RERANK_MIN_SCORE_RATIO", 0.10),
        ("PARENT_SCORE_DECAY", 0.50),
        ("RERANK_MIN_SCORE", 0.25),
        ("HYBRID_KEYWORD_BACKEND", "memory"),
    ],
)
def test_changed_config_default_must_fail(monkeypatch, field: str, new_value) -> None:  # noqa: ANN001
    """改 ``app/config.py`` 的类默认值（而基线未重钉）→ 必须判红."""
    real = freshness.config_defaults()
    mutated = dict(real)
    mutated[field] = new_value
    monkeypatch.setattr(freshness, "config_defaults", lambda: mutated)

    ok, drifts, _notes, _detail = _run(_gather())
    assert ok is False, f"{field} 默认值已变，门禁却仍判绿"
    assert any(field in d for d in drifts), drifts


def test_env_override_must_fail(tmp_path: Path) -> None:
    """只在 ``backend/.env`` 里改（默认值不动）→ 也必须判红（防"残留旧值静默覆盖"）."""
    env_file = tmp_path / ".env"
    env_file.write_text("RERANK_MIN_SCORE_RATIO=0.10\n", encoding="utf-8")

    cur = _gather(env_path=env_file)
    assert cur["env_present"] is True
    ok, drifts, _notes, _detail = _run(cur)
    assert ok is False, ".env 改了 RERANK_MIN_SCORE_RATIO，门禁却仍判绿"
    assert any("RERANK_MIN_SCORE_RATIO" in d for d in drifts), drifts


def test_main_returns_stale_and_prints_remediation(monkeypatch, capsys) -> None:  # noqa: ANN001
    """``main`` 判红时退出码为 2，且打印可直接照做的修复步骤."""
    real = freshness.config_defaults()
    mutated = dict(real)
    mutated["RERANK_MIN_SCORE_RATIO"] = 0.10
    monkeypatch.setattr(freshness, "config_defaults", lambda: mutated)

    assert freshness.main([]) == freshness.EXIT_STALE
    out = capsys.readouterr().out
    assert "run_eval_baseline.py" in out, "修复步骤必须点名重跑基线脚本"
    assert "--pin" in out, "修复步骤必须给出重钉输入指纹的命令"


# ─────────────────────────────────────────────────────────────────────────────
# ③ 改了语料清单 / 阈值 → 必须红
# ─────────────────────────────────────────────────────────────────────────────


def test_corpus_change_must_fail(tmp_path: Path) -> None:
    """往 ``golden_v1.json`` 的语料清单里加一份文档 → 必须判红（重导语料要察觉）."""
    golden_path = _write_golden(
        tmp_path,
        lambda g: g["corpus"]["documents"].append("新文档-回归-x1.txt"),
    )

    ok, drifts, _notes, _detail = _run(_gather(golden_path=golden_path))
    assert ok is False, "语料清单变了，门禁却仍判绿"
    assert any("语料" in d for d in drifts), drifts


def test_thresholds_change_must_fail(tmp_path: Path) -> None:
    """改 ``golden_v1.json`` 的 thresholds → 必须判红（基线 metrics 是在旧阈值下量的）."""
    golden_path = _write_golden(
        tmp_path,
        lambda g: g["thresholds"].__setitem__("min_precision_at_3", 0.99),
    )

    ok, drifts, _notes, _detail = _run(_gather(golden_path=golden_path))
    assert ok is False, "阈值变了，门禁却仍判绿"
    assert any("thresholds" in d for d in drifts), drifts


def test_model_change_must_fail(monkeypatch) -> None:  # noqa: ANN001
    """换精排模型（类默认值）→ 必须判红（换模型 = 换一套打分空间）."""
    real = freshness.model_defaults()
    mutated = dict(real)
    mutated["reranker_model"] = "BAAI/bge-reranker-v2-m3"
    monkeypatch.setattr(freshness, "model_defaults", lambda: mutated)

    ok, drifts, _notes, _detail = _run(_gather())
    assert ok is False, "精排模型换名了，门禁却仍判绿"
    assert any("reranker_model" in d for d in drifts), drifts


def test_reranker_hardcoded_model_must_fail(monkeypatch) -> None:  # noqa: ANN001
    """``reranker.py`` 不再从 settings 取模型名（硬编码）→ 接线守卫必须判红."""
    monkeypatch.setattr(freshness, "reranker_uses_settings_model", lambda: False)
    ok, drifts, _notes, _detail = _run(_gather())
    assert ok is False
    assert any("RERANKER_MODEL_NAME" in d for d in drifts), drifts


# ─────────────────────────────────────────────────────────────────────────────
# ④ 阴性对照：证明判据会区分（不是恒真 / 也不是恒假）
# ─────────────────────────────────────────────────────────────────────────────


def test_detector_distinguishes_change_from_no_change(monkeypatch) -> None:  # noqa: ANN001
    """同一批输入：不改 → 绿；改一个默认值 → 红。并排证明判据**会区分**."""
    baseline_ok, _, _, _ = _run(_gather())
    assert baseline_ok is True, "不改任何东西却判红 —— 判据恒假（门禁会误报）"

    real = freshness.config_defaults()
    mutated = dict(real)
    mutated["PARENT_SCORE_DECAY"] = real["PARENT_SCORE_DECAY"] - 0.1
    monkeypatch.setattr(freshness, "config_defaults", lambda: mutated)
    changed_ok, drifts, _, _ = _run(_gather())
    assert changed_ok is False, "改了一个默认值却判绿 —— 判据恒真（门禁形同虚设）"
    assert any("PARENT_SCORE_DECAY" in d for d in drifts), drifts


def test_corpus_order_does_not_false_red(tmp_path: Path) -> None:
    """语料指纹对**顺序不敏感** —— 仅调换顺序不得误报（避免噪声红）."""
    golden_path = _write_golden(
        tmp_path,
        lambda g: g["corpus"].__setitem__("documents", list(reversed(g["corpus"]["documents"]))),
    )
    ok, drifts, _notes, _detail = _run(_gather(golden_path=golden_path))
    assert ok is True, f"仅调换语料顺序不应判红：{drifts}"


def test_missing_env_is_tolerated(tmp_path: Path) -> None:
    """``.env`` 缺席（CI / 干净检出）→ 跳过 .env 对账、仍判绿（不得因缺 .env 假红）."""
    missing = tmp_path / "no-such.env"
    cur = _gather(env_path=missing)
    assert cur["env_present"] is False
    ok, drifts, notes, _detail = _run(cur)
    assert ok is True, f".env 缺席不应判红：{drifts}"
    assert any(".env" in n for n in notes), notes


def test_missing_fingerprint_must_fail(tmp_path: Path) -> None:
    """baseline 缺 ``input_fingerprint``（尚未建立）→ 必须判红并提示先 --pin."""
    stripped = json.loads(BASELINE.read_text(encoding="utf-8"))
    stripped.pop("input_fingerprint", None)
    path = tmp_path / "baseline_v1.json"
    path.write_text(json.dumps(stripped, ensure_ascii=False, indent=2), encoding="utf-8")

    cur = freshness.gather_current(
        baseline_path=path, golden_path=GOLDEN, env_path=ENV,
    )
    ok, drifts, _notes, _detail = _run(cur)
    assert ok is False
    assert any("input_fingerprint" in d for d in drifts), drifts


# ─────────────────────────────────────────────────────────────────────────────
# 解析层 + 指纹完整性 + --pin 只增不改
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_dotenv_strips_inline_comments(tmp_path: Path) -> None:
    """``KEY=value  # 说明`` 必须只取 value —— 本仓库 .env 大量使用该写法."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "RERANK_MIN_SCORE=0.05           # 精排置信度阈值\n"
        'RERANKER_MODEL_NAME="BAAI/bge-reranker-base"   # 约 278MB\n'
        "HYBRID_SEARCH_ENABLED=true\n"
        "# 整行注释\n"
        "\n",
        encoding="utf-8",
    )
    parsed = freshness._parse_dotenv(env_file)
    assert parsed["RERANK_MIN_SCORE"] == "0.05", parsed
    assert parsed["RERANKER_MODEL_NAME"] == "BAAI/bge-reranker-base", parsed
    assert parsed["HYBRID_SEARCH_ENABLED"] == "true", parsed
    assert "#" not in parsed["RERANK_MIN_SCORE"]


def test_fingerprint_covers_all_pinned_fields() -> None:
    """已落库的 ``input_fingerprint`` 必须覆盖全部钉住字段（防"指纹不完整"）."""
    fp = freshness.load_json(BASELINE)["input_fingerprint"]
    for field in freshness.PINNED_CONFIG_FIELDS:
        assert field in fp["config_defaults"], f"指纹缺 config_defaults[{field}]"
        assert field in fp["config_effective"], f"指纹缺 config_effective[{field}]"
    for alias in freshness.MODEL_FIELDS:
        assert alias in fp["models_defaults"], f"指纹缺 models_defaults[{alias}]"
    assert fp["corpus"]["sha256"], "缺语料指纹"
    assert fp["thresholds_sha256"], "缺阈值指纹"
    assert fp["fingerprint_sha256"], "缺组合指纹"


def test_pin_only_adds_the_fingerprint_key(tmp_path: Path) -> None:
    """``--pin`` 只允许新增 ``input_fingerprint`` —— 既有键逐字节不变（自检兜底）."""
    full = json.loads(BASELINE.read_text(encoding="utf-8"))
    reference = full["input_fingerprint"]
    # 模拟"首次钉指纹"：先把 input_fingerprint 摘掉
    without = {k: v for k, v in full.items() if k != "input_fingerprint"}
    path = tmp_path / "baseline_copy.json"
    path.write_text(json.dumps(without, ensure_ascii=False, indent=2), encoding="utf-8")

    fp = freshness.build_fingerprint(
        config_defaults=reference["config_defaults"],
        config_effective=reference["config_effective"],
        env_present=True,
        models_defaults=reference["models_defaults"],
        models_effective=reference["models_effective"],
        documents=reference["corpus"]["documents"],
        thresholds_hash=reference["thresholds_sha256"],
        config_snapshot=reference_snapshot(full),
    )
    freshness.pin_fingerprint(path, fp)

    after = json.loads(path.read_text(encoding="utf-8"))
    assert set(after) - set(without) == {"input_fingerprint"}
    for key, value in without.items():
        assert after[key] == value, f"--pin 改动了既有键 {key}"
    # metrics / config_snapshot 是硬红线：数字一位都不能动
    assert after["metrics"] == full["metrics"]
    assert after["config_snapshot"] == full["config_snapshot"]


def reference_snapshot(full: dict) -> dict:
    return full.get("config_snapshot") or {}


# ─────────────────────────────────────────────────────────────────────────────
# CI 接线：门禁必须真的跑（不是被 if: false 禁掉的死代码）
# ─────────────────────────────────────────────────────────────────────────────


def test_ci_workflow_wires_the_freshness_gate() -> None:
    workflow = _REPO_ROOT / ".github" / "workflows" / "backend-ci.yml"
    if not workflow.is_file():
        pytest.skip("找不到 backend-ci.yml")
    text = workflow.read_text(encoding="utf-8")
    assert "check_eval_baseline_freshness.py" in text, (
        "CI 没有把输入对账门禁接进 gates job"
    )
    assert "if: ${{ false }}" not in text, (
        "仍有被 if: false 硬禁用的 job —— 门禁是死代码，红了自己也不知道"
    )


if __name__ == "__main__":
    import traceback

    ok = fail = 0
    for _name, _fn in sorted(globals().items()):
        if not _name.startswith("test_") or not callable(_fn):
            continue
        try:
            _fn()
            ok += 1
        except Exception:      # noqa: BLE001
            fail += 1
            print(f"  FAIL {_name}")
            traceback.print_exc()
    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)
