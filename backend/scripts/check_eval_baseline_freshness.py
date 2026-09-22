"""检索质量基线「输入指纹」对账门禁 —— **不需要真实栈**（纯逻辑）.

它解决的问题
────────────
``scripts/run_eval_baseline.py`` 是检索质量回归门禁，但它必须连真实栈
（postgres / qdrant / ollama / keycloak + ``RAG_EVAL_PASSWORD``）。托管 CI 起不动
真实栈，于是它在 ``.github/workflows/backend-ci.yml`` 里被 ``if: false`` 硬禁用 ——
结果是**门禁早就红了（recall@10 从 1.0 掉到 0.9375）却没有任何人在跑**：所谓
「假红」的土壤正在这里。红不是因为检索变差，而是因为**输入变了**（换了可见性口径 /
重导语料 / 动过阈值），而基线 ``backend/eval/baseline_v1.json`` 没有跟着重钉。

本脚本把「输入有没有变」这件事从真实栈里**解耦**出来，做成一个纯逻辑检查：
读 ``baseline_v1.json`` 里钉住的**输入指纹**，与**当前仓库实际值**逐一比对。
任何一处输入漂移（而基线未重钉）→ **退出码非 0**，并打印可直接照做的修复步骤。

比对什么（全部只读文件，零模型、零 DB）
──────────────────────────────────────
1. **配置**：``app/config.py`` 的**类默认值** + ``backend/.env`` 的**生效值**，
   对 ``baseline_v1.json.config_snapshot`` 对应的 14 个检索旋钮，外加 24 个同样
   决定检索/分块结果的字段。默认值与 .env 分别比对：
     · 只比默认值 → 漏掉「` .env` 残留旧值静默覆盖」这类历史坑；
     · 只比 .env   → 漏掉「默认值改了但被 .env 盖住」的潜伏漂移。
   两者都钉，缺一不可。
2. **模型标识**：嵌入模型（``BGE_MODEL_NAME`` + ``EMBEDDING_DIMENSION``）、
   精排模型（``RERANKER_MODEL_NAME``）、生成/视觉模型（``OLLAMA_MODEL`` /
   ``OLLAMA_VISION_MODEL``）。换任何一个是换了一套打分/向量空间 → 基线必须重跑。
   另加一条**接线守卫**：``app/services/reranker.py`` 必须仍从 ``settings`` 取
   模型名（``RERANKER_MODEL_NAME``）；若被硬编码成别的模型，本脚本照样判红。
3. **语料定义**：``golden_v1.json`` 的 ``corpus.documents`` 指纹 —— 重导语料 /
   改语料清单必须被察觉。
4. **阈值常量**：``golden_v1.json`` 的 ``thresholds`` 指纹、以及代码里的护栏常量
   ``RERANK_MIN_SCORE_RATIO_CEILING`` 必须与金标集声明一致（「两处声明同步」）。
5. **基线自洽**：``baseline_v1.json`` 内部 ``config_snapshot`` 与
   ``input_fingerprint.config_effective`` 必须一致（防「改了一处忘了另一处」）。

用法
────
对账（CI 门禁，非 0 即红）::

    cd backend && python scripts/check_eval_baseline_freshness.py

重钉输入指纹（输入**有意变更**、且已按口径重跑基线后）::

    cd backend && python scripts/check_eval_baseline_freshness.py --pin

``--pin`` **只写 ``baseline_v1.json`` 的 ``input_fingerprint`` 一个键**，
不改任何 metric 数字、不改任何 threshold、不改 ``config_snapshot``（写完会自检，
发现既有键被改动立即中止、不落盘）。

退出码
──────
    0 → 输入一致（基线可作数）
    2 → 输入漂移（基线已过期，必须先重钉再对比指标）
    3 → 环境/文件错误（缺 baseline/golden、config 字段被删等）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# ── 退出码 ───────────────────────────────────────────────────────────────────
EXIT_OK = 0
EXIT_STALE = 2
EXIT_ERROR = 3

# ── 钉住哪些字段 ─────────────────────────────────────────────────────────────

#: ``baseline_v1.json.config_snapshot`` 的键 → ``Settings`` 字段名。
#: 这是基线与金标集**共同声明**的那 14 个旋钮（baseline 的 config_snapshot 就是
#: 按它们记的），必须逐个对账。
CONFIG_SNAPSHOT_MAP: dict[str, str] = {
    "hybrid_search": "HYBRID_SEARCH_ENABLED",
    "reranker": "RERANKER_ENABLED",
    "hierarchical_rag": "HIERARCHICAL_RAG_ENABLED",
    "rerank_min_score": "RERANK_MIN_SCORE",
    "rerank_min_score_ratio": "RERANK_MIN_SCORE_RATIO",
    "rerank_min_score_filter": "RERANK_MIN_SCORE_FILTER",
    "evidence_gate_min_top_score": "EVIDENCE_GATE_MIN_TOP_SCORE",
    "retrieval_min_score": "RETRIEVAL_MIN_SCORE",
    "retrieval_max_gap": "RETRIEVAL_MAX_GAP",
    "multi_query_max_extra": "MULTI_QUERY_MAX_EXTRA",
    "query_rewrite_enabled": "QUERY_REWRITE_ENABLED",
    "acl_prefilter_enabled": "ACL_PREFILTER_ENABLED",
    "parent_score_decay": "PARENT_SCORE_DECAY",
    "hybrid_keyword_backend": "HYBRID_KEYWORD_BACKEND",
}

#: ``config_snapshot`` 未记录、但同样决定**检索结果或语料物化结果**的字段 ——
#: 动它们和动上面那 14 个一样会让基线数字失去意义，所以一并钉住。
EXTRA_CONFIG_FIELDS: tuple[str, ...] = (
    # 粗排 / 精排
    "RETRIEVAL_TOP_K",
    "ANN_OVERFETCH_FACTOR",
    "RERANKER_MAX_CANDIDATES",
    "RERANKER_MAX_TEXT_CHARS",
    "RERANKER_USE_QUERY_VARIANTS",
    "HYBRID_RRF_K",
    "EVIDENCE_GATE_MIN_COVERAGE",
    "PARENT_CHILD_MAX_PER_PARENT",
    "RETRIEVAL_CONTENT_DEDUP_ENABLED",
    "EVIDENCE_TRUST_ENABLED",
    "EVIDENCE_TRUST_RERANK_WEIGHT",
    # 查询变换
    "MULTI_QUERY_ENABLED",
    "QUERY_HYDE_ENABLED",
    "QUERY_DECOMPOSITION_ENABLED",
    # 元数据过滤
    "METADATA_FILTER_ENABLED",
    # 分块 / 父子块（决定 chunk_index 与语料物化结果）
    "MIN_CHUNK_SIZE",
    "MAX_CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "CHUNK_TOKEN_AWARE",
    "CHARS_PER_TOKEN",
    "CHUNK_PARENT_SIZE_MULT",
    "CHUNK_PARENT_OVERLAP",
    # 可见性口径（决定 admin 到底能看见哪些文档）
    "SECURITY_STRICT_MODE",
    "DEFAULT_SECURITY_LEVEL",
)

#: 钉住的配置字段全集（= 14 个 snapshot 字段 + 扩展字段）。
PINNED_CONFIG_FIELDS: tuple[str, ...] = (
    tuple(CONFIG_SNAPSHOT_MAP.values()) + EXTRA_CONFIG_FIELDS
)

#: 模型标识：embedding / reranker / chat / vision 的身份（alias → Settings 字段）。
MODEL_FIELDS: dict[str, str] = {
    "embedding_model": "BGE_MODEL_NAME",
    "embedding_dimension": "EMBEDDING_DIMENSION",
    "reranker_model": "RERANKER_MODEL_NAME",
    "ollama_chat_model": "OLLAMA_MODEL",
    "ollama_vision_model": "OLLAMA_VISION_MODEL",
}

#: reranker 源码里必须出现的接线引用（防「模型名被硬编码」）。
RERANKER_WIRING_TOKEN = "RERANKER_MODEL_NAME"

FINGERPRINT_KEY = "input_fingerprint"
FINGERPRINT_SCHEMA = 1

# ── 路径 ─────────────────────────────────────────────────────────────────────

DEFAULT_BASELINE = _BACKEND_ROOT / "eval" / "baseline_v1.json"
DEFAULT_GOLDEN = _BACKEND_ROOT / "eval" / "golden_v1.json"
DEFAULT_ENV = _BACKEND_ROOT / ".env"
RERANKER_SOURCE = _BACKEND_ROOT / "app" / "services" / "reranker.py"


# ─────────────────────────────────────────────────────────────────────────────
# 基础工具
# ─────────────────────────────────────────────────────────────────────────────


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(obj: Any) -> str:
    """稳定序列化：键排序 + 紧凑分隔符 —— 保证同一对象永远同一串。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _parse_dotenv(path: Path) -> dict[str, str]:
    """极简 .env 解析：``KEY=VALUE``，去引号、去行内注释（``#``）.

    行内注释判定与 python-dotenv 对齐：`` #`` 之后是注释。本仓库的
    ``backend/.env`` 大量使用 ``KEY=value  # 说明`` 风格，不剥掉会把说明文字
    算进值里，导致 .env 生效值对账假红。
    """
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # 先处理成对引号：``KEY="a b"  # 说明`` —— 引号优先，其余（含 # 说明）丢掉。
        if value[:1] in ("\"", "'"):
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            # 去行内注释：`` #`` 之前的部分（本仓库 .env 大量 ``KEY=value  # 说明``）
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        out[key] = value
    return out


def _coerce_like(raw: Any, reference: Any) -> Any:
    """把 ``raw`` 强制成 ``reference`` 的类型（.env 里一切皆字符串）.

    无法转换时原样返回（交给比对报差异，比静默丢字段好）。
    """
    if isinstance(reference, bool):
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
        return raw
    if isinstance(reference, int) and not isinstance(reference, bool):
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError):
            return raw
    if isinstance(reference, float):
        try:
            return float(str(raw).strip())
        except (TypeError, ValueError):
            return raw
    return raw if isinstance(raw, str) else str(raw)


def _apply_env(defaults: dict[str, Any], env_raw: dict[str, str]) -> dict[str, Any]:
    """在类默认值之上叠加 .env 生效值（键不存在则沿用默认）."""
    effective: dict[str, Any] = {}
    for field, default in defaults.items():
        if field in env_raw:
            effective[field] = _coerce_like(env_raw[field], default)
        else:
            effective[field] = default
    return effective


def _env_overrides(effective: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """只保留「与默认值不同」的生效项（即 .env 真正改动了什么）."""
    return {k: v for k, v in effective.items() if defaults.get(k) != v}


# ─────────────────────────────────────────────────────────────────────────────
# 读取当前仓库
# ─────────────────────────────────────────────────────────────────────────────


def _settings_model() -> type:
    """返回 ``app.config.Settings`` 类（不实例化 —— 实例化会读 .env 且可能报错）."""
    import importlib

    return getattr(importlib.import_module("app.config"), "Settings")


def config_defaults() -> dict[str, Any]:
    """从 ``app.config.Settings`` 读**配置字段的类默认值**（不受 .env 影响）."""
    cls = _settings_model()
    out: dict[str, Any] = {}
    missing: list[str] = []
    for field in PINNED_CONFIG_FIELDS:
        f = cls.model_fields.get(field)
        if f is None:
            missing.append(field)
            continue
        out[field] = f.default
    if missing:
        raise KeyError(
            "app/config.py 的 Settings 里找不到这些配置字段（被删/改名了？）："
            + ", ".join(missing)
        )
    return out


def model_defaults() -> dict[str, Any]:
    """从 ``app.config.Settings`` 读**模型标识字段的类默认值**."""
    cls = _settings_model()
    out: dict[str, Any] = {}
    missing: list[str] = []
    for alias, field in MODEL_FIELDS.items():
        f = cls.model_fields.get(field)
        if f is None:
            missing.append(field)
            continue
        out[alias] = f.default
    if missing:
        raise KeyError(
            "app/config.py 的 Settings 里找不到这些模型字段（被删/改名了？）："
            + ", ".join(missing)
        )
    return out


def current_corpus(golden: dict) -> list[str]:
    return list((golden.get("corpus") or {}).get("documents") or [])


def corpus_sha256(documents: list[str]) -> str:
    """语料指纹：**排序后**拼接再哈希（重导语料的顺序不应造成假红）."""
    return _sha256(_canonical(sorted(documents)))


def thresholds_sha256(golden: dict) -> str:
    return _sha256(_canonical(golden.get("thresholds") or {}))


def reranker_uses_settings_model() -> bool:
    """``app/services/reranker.py`` 是否仍从 settings 取精排模型名."""
    if not RERANKER_SOURCE.is_file():
        return False
    return RERANKER_WIRING_TOKEN in RERANKER_SOURCE.read_text(
        encoding="utf-8", errors="replace"
    )


def guard_ceiling() -> float | None:
    """读代码里的护栏常量 ``RERANK_MIN_SCORE_RATIO_CEILING``."""
    try:
        from app.config import RERANK_MIN_SCORE_RATIO_CEILING  # noqa: PLC0415

        return float(RERANK_MIN_SCORE_RATIO_CEILING)
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 指纹构建 / 读取
# ─────────────────────────────────────────────────────────────────────────────


def build_fingerprint(
    *,
    config_defaults: dict[str, Any],
    config_effective: dict[str, Any],
    env_present: bool,
    models_defaults: dict[str, Any],
    models_effective: dict[str, Any],
    documents: list[str],
    thresholds_hash: str,
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按当前仓库状态构造 ``input_fingerprint`` 对象.

    ``fingerprint_sha256`` 只覆盖**与 .env 无关**的输入（类默认值 + 模型默认值 +
    语料 + 阈值 + config_snapshot），因此它在「本地带 .env」与「CI 无 .env」两种
    环境下取值一致 —— 组合哈希不会因为 .env 的在场与否而漂。
    """
    corpus_hash = corpus_sha256(documents)
    snapshot_hash = _sha256(_canonical(config_snapshot or {}))
    core = {
        "config_defaults": config_defaults,
        "models_defaults": models_defaults,
        "corpus_sha256": corpus_hash,
        "thresholds_sha256": thresholds_hash,
        "config_snapshot_sha256": snapshot_hash,
    }
    return {
        "schema": FINGERPRINT_SCHEMA,
        "algo": "sha256",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "why": (
            "检索质量基线的**输入指纹**：这些输入没变，baseline 的 metrics 才作数。"
            "任何一项变了而本对象未重钉 → scripts/check_eval_baseline_freshness.py 判红。"
            "本对象只记录证据，不约束业务（阈值以 golden_v1.json 的 thresholds 为准）。"
        ),
        "config_defaults": config_defaults,
        "config_effective": config_effective,
        "config_effective_env_present": bool(env_present),
        "env_overrides": _env_overrides(config_effective, config_defaults),
        "models_defaults": models_defaults,
        "models_effective": models_effective,
        "corpus": {"documents": documents, "sha256": corpus_hash},
        "thresholds_sha256": thresholds_hash,
        "config_snapshot_sha256": snapshot_hash,
        "fingerprint_sha256": _sha256(_canonical(core)),
        "check_scope": (
            "纯逻辑：只读 app/config.py 类默认值 / backend/.env / "
            "app/services/reranker.py 接线 / golden_v1.json 语料与阈值；不需要真实栈。"
        ),
    }


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 对账
# ─────────────────────────────────────────────────────────────────────────────


def _diff(name: str, expected: Any, got: Any) -> str | None:
    if expected == got:
        return None
    return f"{name}: 期望 {expected!r}，实际 {got!r}"


def check_inputs(
    *,
    baseline: dict,
    golden: dict,
    config_defaults: dict[str, Any],
    config_effective: dict[str, Any],
    env_present: bool,
    models_defaults: dict[str, Any],
    models_effective: dict[str, Any],
    documents: list[str],
    thresholds_hash: str,
    wiring_ok: bool,
    ceiling: float | None,
) -> tuple[bool, list[str], list[str], dict[str, Any]]:
    """返回 ``(ok, drifts, notes, detail)``；``drifts`` 为空即通过."""
    drifts: list[str] = []
    notes: list[str] = []
    detail: dict[str, Any] = {}

    pinned = baseline.get(FINGERPRINT_KEY)
    if not isinstance(pinned, dict):
        return (
            False,
            [
                f"baseline_v1.json 缺少 {FINGERPRINT_KEY!r} 字段（输入指纹未建立）"
                " —— 无法对账，请先运行 --pin 建立指纹"
            ],
            notes,
            {"fingerprint_present": False},
        )

    detail["fingerprint_present"] = True
    detail["fingerprint_sha256_pinned"] = pinned.get("fingerprint_sha256")

    # ── 1. 配置类默认值（始终可查）─────────────────────────────────────────
    pinned_defaults = pinned.get("config_defaults") or {}
    for field in PINNED_CONFIG_FIELDS:
        if field not in pinned_defaults:
            drifts.append(f"config_defaults 缺少字段 {field}（指纹不完整，请 --pin 重建）")
            continue
        d = _diff(f"config_defaults[{field}]", pinned_defaults[field],
                  config_defaults.get(field))
        if d:
            drifts.append(d)

    # ── 2. .env 生效值（仅当 .env 在场）────────────────────────────────────
    pinned_effective = pinned.get("config_effective") or {}
    if env_present:
        for field in PINNED_CONFIG_FIELDS:
            if field not in pinned_effective:
                drifts.append(
                    f"config_effective 缺少字段 {field}（指纹不完整，请 --pin 重建）"
                )
                continue
            d = _diff(f".env 生效值[{field}]", pinned_effective[field],
                      config_effective.get(field))
            if d:
                drifts.append(d)
        d = _diff("env_overrides", pinned.get("env_overrides") or {},
                  _env_overrides(config_effective, config_defaults))
        if d:
            drifts.append(d + "（backend/.env 里的覆盖项与基线指纹不一致）")
    else:
        notes.append(
            "未发现 backend/.env（CI / 干净检出常见）→ 跳过 .env 生效值对账；"
            "仅比对了类默认值。本地开发机上跑同一命令会连 .env 一起比。"
        )
    detail["env_present"] = bool(env_present)

    # ── 3. 模型标识（默认值始终可查；生效值仅当 .env 在场）─────────────────
    pinned_models = pinned.get("models_defaults") or {}
    for alias, field in MODEL_FIELDS.items():
        if alias not in pinned_models:
            drifts.append(f"models_defaults 缺少 {alias}（{field}）（指纹不完整，请 --pin 重建）")
            continue
        d = _diff(f"models_defaults[{alias}] <- {field}", pinned_models[alias],
                  models_defaults.get(alias))
        if d:
            drifts.append(d)
    if env_present:
        pinned_models_eff = pinned.get("models_effective") or {}
        for alias, field in MODEL_FIELDS.items():
            if alias not in pinned_models_eff:
                continue
            d = _diff(f".env 生效模型[{alias}] <- {field}", pinned_models_eff[alias],
                      models_effective.get(alias))
            if d:
                drifts.append(d)
    if not wiring_ok:
        drifts.append(
            "app/services/reranker.py 不再引用 RERANKER_MODEL_NAME —— "
            "精排模型名被硬编码/改名，基线指纹无法反映真实模型"
        )

    # ── 4. 语料 / 阈值（始终可查）─────────────────────────────────────────
    pinned_corpus = (pinned.get("corpus") or {}).get("sha256")
    live_corpus = corpus_sha256(documents)
    detail["corpus_sha256"] = live_corpus
    if pinned_corpus != live_corpus:
        drifts.append(
            "golden_v1.json 语料指纹变化："
            f"钉住 {pinned_corpus}，实际 {live_corpus}"
            f"（documents={len(documents)} 份）—— 重导语料/改语料清单后必须重跑基线"
        )
    d = _diff("golden_v1.json thresholds 指纹", pinned.get("thresholds_sha256"),
              thresholds_hash)
    if d:
        drifts.append(d + "（阈值变了，基线 metrics 是在旧阈值下量的）")

    # ── 5. 基线自洽：config_snapshot 必须与指纹里的生效值一致 ────────────────
    snapshot = baseline.get("config_snapshot") or {}
    for snap_key, field in CONFIG_SNAPSHOT_MAP.items():
        if snap_key not in snapshot:
            continue
        pinned_val = pinned_effective.get(field, config_defaults.get(field))
        d = _diff(f"config_snapshot[{snap_key}] vs input_fingerprint[{field}]",
                  pinned_val, snapshot[snap_key])
        if d:
            drifts.append(d + "（baseline_v1.json 内部两处声明不同步）")

    # ── 6. 护栏常量与金标集声明同步 ───────────────────────────────────────
    declared = (golden.get("thresholds") or {}).get("max_rerank_min_score_ratio")
    detail["ceiling"] = {"code": ceiling, "golden_declared": declared}
    if ceiling is None:
        drifts.append("app/config.py 里找不到 RERANK_MIN_SCORE_RATIO_CEILING（护栏常量被删）")
    elif declared is not None and abs(float(ceiling) - float(declared)) > 1e-9:
        drifts.append(
            f"护栏常量 RERANK_MIN_SCORE_RATIO_CEILING={ceiling} 与金标集声明上界 "
            f"{declared} 不一致 —— 两处声明必须同步修改"
        )

    # ── 汇总：组合哈希只在「逐项都过」时才有资格代表输入未变 ────────────────
    live = build_fingerprint(
        config_defaults=config_defaults,
        config_effective=config_effective,
        env_present=env_present,
        models_defaults=models_defaults,
        models_effective=models_effective,
        documents=documents,
        thresholds_hash=thresholds_hash,
        config_snapshot=snapshot,
    )
    detail["fingerprint_sha256_live"] = live["fingerprint_sha256"]
    if not drifts and pinned.get("fingerprint_sha256") != live["fingerprint_sha256"]:
        drifts.append(
            "组合指纹不一致（逐项对账都过，但 fingerprint_sha256 不同）——"
            " 指纹块被手工改动过，请 --pin 重建"
        )

    return (not drifts), drifts, notes, detail


# ─────────────────────────────────────────────────────────────────────────────
# 修复步骤
# ─────────────────────────────────────────────────────────────────────────────

_REMEDIATION = """\
================================================================================
❌ 检索质量基线输入对账失败：baseline_v1.json 的输入指纹与当前仓库不一致。
   这表示「测量这份基线时的输入」已经被改动 —— 基线里的 metrics 不再作数，
   拿它跟新一轮对账会得到假结果。**不要**去改阈值或指标去迁就它。
================================================================================
修复步骤（按顺序照做，全部以 backend/ 为工作目录）：

  1) 在**能起真实栈**的发布机上重跑检索质量门禁，拿到新回执：
       docker exec -e RAG_EVAL_PASSWORD=… rag_backend \\
           python -u scripts/run_eval_baseline.py --out /tmp/eval_baseline_report.json
     （退出码：0=通过 2=检索质量回归 3=登录/配置 4=金标解析 5=评测请求）

  2) 本次输入是**有意变更**、且新回执无回归后，按同一口径重钉输入指纹：
       python scripts/check_eval_baseline_freshness.py --pin
     （只更新 baseline_v1.json 的 input_fingerprint，不动任何 metric / threshold）

  3) 按回执与 §口径 更新 backend/eval/baseline_v1.json 的
     metrics / config_snapshot / revision / revision_note（§口径见该文件 note）。

  4) 重新对账应转绿；随后提交。本门禁在 CI 的 gates job 里**真跑**，非 0 即红。
"""


# ─────────────────────────────────────────────────────────────────────────────
# --pin：把当前输入指纹写进 baseline_v1.json（只增一个键）
# ─────────────────────────────────────────────────────────────────────────────


def pin_fingerprint(baseline_path: Path, fingerprint: dict, *, dry_run: bool = False) -> dict:
    """把 ``input_fingerprint`` 写入基线文件，**保证其它键逐字节不变**.

    实现：写入后重新解析、逐键比对既有内容，任一被改动即抛错并**不落盘** ——
    「只增不改」由机器校验保证，而不是靠自觉。
    """
    old = load_json(baseline_path)
    protected = {k: v for k, v in old.items() if k != FINGERPRINT_KEY}

    new = dict(old)
    new[FINGERPRINT_KEY] = fingerprint

    if dry_run:
        return new

    text = json.dumps(new, ensure_ascii=False, indent=2) + "\n"
    parsed = json.loads(text)
    for key, value in protected.items():
        if parsed.get(key) != value:
            raise RuntimeError(
                f"写入会改动既有键 {key!r} —— 已中止（--pin 只允许新增 input_fingerprint）"
            )
    baseline_path.write_text(text, encoding="utf-8")
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# 采集当前仓库输入
# ─────────────────────────────────────────────────────────────────────────────


def gather_current(
    *,
    baseline_path: Path,
    golden_path: Path,
    env_path: Path,
) -> dict[str, Any]:
    """把「当前仓库的输入」读齐（一切只读，零模型、零 DB）."""
    baseline = load_json(baseline_path)
    golden = load_json(golden_path)

    cfg_defaults = config_defaults()
    mdl_defaults = model_defaults()

    env_present = env_path.is_file()
    env_raw = _parse_dotenv(env_path)
    pinned_keys = set(PINNED_CONFIG_FIELDS) | set(MODEL_FIELDS.values())
    env_raw = {k: v for k, v in env_raw.items() if k in pinned_keys}

    cfg_effective = _apply_env(cfg_defaults, env_raw)
    mdl_effective = {
        alias: _coerce_like(env_raw.get(field, mdl_defaults[alias]), mdl_defaults[alias])
        for alias, field in MODEL_FIELDS.items()
    }

    return {
        "baseline": baseline,
        "golden": golden,
        "config_defaults": cfg_defaults,
        "config_effective": cfg_effective,
        "env_present": env_present,
        "models_defaults": mdl_defaults,
        "models_effective": mdl_effective,
        "documents": current_corpus(golden),
        "thresholds_hash": thresholds_sha256(golden),
        "wiring_ok": reranker_uses_settings_model(),
        "ceiling": guard_ceiling(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="检索质量基线「输入指纹」对账门禁（不需要真实栈）",
    )
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    ap.add_argument("--env-file", default=str(DEFAULT_ENV))
    ap.add_argument(
        "--pin", action="store_true",
        help="把当前仓库的输入指纹重钉进 baseline_v1.json（只新增 input_fingerprint 键）",
    )
    ap.add_argument("--dry-run", action="store_true", help="配合 --pin：只打印不落盘")
    args = ap.parse_args(argv)

    baseline_path = Path(args.baseline).resolve()
    golden_path = Path(args.golden).resolve()
    env_path = Path(args.env_file).resolve()

    if not baseline_path.is_file():
        print(f"[ERROR] 找不到基线文件：{baseline_path}")
        return EXIT_ERROR
    if not golden_path.is_file():
        print(f"[ERROR] 找不到金标文件：{golden_path}")
        return EXIT_ERROR

    try:
        cur = gather_current(
            baseline_path=baseline_path, golden_path=golden_path, env_path=env_path,
        )
    except KeyError as exc:
        print(f"[ERROR] 读取 app/config.py 失败：{exc}")
        return EXIT_ERROR

    print("── 检索质量基线输入对账（纯逻辑，不需要真实栈）──────────────────────")
    print(f"baseline : {baseline_path}")
    print(f"golden   : {golden_path}")
    print(f"env file : {env_path} ({'在场' if cur['env_present'] else '缺席'})")
    print(f"语料份数 : {len(cur['documents'])}")
    print(f"护栏常量 : RERANK_MIN_SCORE_RATIO_CEILING={cur['ceiling']}")

    if args.pin:
        fp = build_fingerprint(
            config_defaults=cur["config_defaults"],
            config_effective=cur["config_effective"],
            env_present=cur["env_present"],
            models_defaults=cur["models_defaults"],
            models_effective=cur["models_effective"],
            documents=cur["documents"],
            thresholds_hash=cur["thresholds_hash"],
            config_snapshot=cur["baseline"].get("config_snapshot") or {},
        )
        try:
            pin_fingerprint(baseline_path, fp, dry_run=args.dry_run)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
            return EXIT_ERROR
        verb = "（dry-run，未落盘）" if args.dry_run else "已写入"
        print(f"\n✅ 输入指纹{verb}：{FINGERPRINT_KEY}.fingerprint_sha256="
              f"{fp['fingerprint_sha256']}")
        return EXIT_OK

    ok, drifts, notes, detail = check_inputs(
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

    for note in notes:
        print(f"  · 说明：{note}")

    if ok:
        print("\n✅ 输入一致：baseline_v1.json 的输入指纹与当前仓库相符（基线可作数）。")
        print(f"   fingerprint_sha256={detail.get('fingerprint_sha256_live')}")
        return EXIT_OK

    print(f"\n发现 {len(drifts)} 处输入漂移：")
    for i, d in enumerate(drifts, 1):
        print(f"  {i}. {d}")
    print()
    print(_REMEDIATION)
    return EXIT_STALE


if __name__ == "__main__":
    sys.exit(main())
