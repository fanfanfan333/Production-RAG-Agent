# 检索质量评测与回归门禁（backend/eval）

本目录是**检索质量回归门禁**的数据与口径说明。三件东西：

| 文件 | 角色 | 一句话 |
| --- | --- | --- |
| `golden_v1.json` | **金标用例集 + 阈值** | 定义"什么算对"：16 条正例（11 单证据 + 5 多证据）+ 3 条负例，以及各指标的**门禁下限** |
| `baseline_v1.json` | **实测基线 + 校准记录** | 记录"当前是多少"：在真实栈上实测的一套指标数字、分带扫描、以及每条结论背后的口径 |
| `scripts/run_eval_baseline.py` | **门禁执行器** | 读金标 → 走生产同源检索 → 对账 thresholds → 非 0 即阻断发布 |

> 判回归请**只看 `golden_v1.json` 的 `thresholds`**，不要拿两个文件里的绝对数字
> 跨版本比大小 —— 口径（用例数、scope、ratio）变过，绝对数字不可比。原因见下文
> "为什么不能直接比数字"。

---

## 1. `revision` 语义

两个文件都有整型 `revision` 与 `revision_note`。**它们必须一起读**：

* `revision` 每次**实质变更**（用例增删 / 阈值调整 / 口径修正 / 基线数字更新）递增；
* `revision_note` 说明"这一版改了什么、为什么、与上一版是否可比"。

当前：

* `golden_v1.json` → **rev4**：只补文档说明（`scope_note`），**用例与 thresholds 零改动**，
  因此 rev3/rev4 的评测数字逐位可比。
* `baseline_v1.json` → **rev4**：**二次在运行中的完整栈上复跑重钉**（15 例可达口径，
  PASS）。相对 rev3 只有 `precision@5`（0.7444→0.7411）与 `precision@10`
  （0.7324→0.7411）两处 run-to-run 噪声级变化；`recall@*` / `mrr` / `ndcg@*` /
  `hit_rate@*` / `map` / 多证据 `all_gold_found_rate` / 负例 / 分带**逐位一致**。
  **阈值与用例零改动**（阈值在 `golden_v1.json`，本轮未碰）。rev3 是"第一次可作数"
  的 16 例口径实测；rev1/rev2 是 11 例口径、ratio=0.10 时代的历史值，**仅作追溯**。
* `baseline_v1.json.input_fingerprint` → 由 `scripts/check_eval_baseline_freshness.py`
  钉住并核对。**改输入（语料 / 模型 / 检索旋钮 / 阈值）而不重钉它，CI 立即变红**
  （见 §7）—— 这正是"门禁红了多日却无人执行"的根治点。

> 约定：`baseline_v1.json` 里带 `_status: historical` 的段落（如 `rerank_band_sweep`）
> 是**已失效**的历史结论，当前口径以被 `decision_superseded_by` 指向的新段落为准。

---

## 2. 评测口径必须与检索链路**同源**（最重要的一条）

金标的"对/错"只有在**同一个权限口径**下才有意义。口径链路：

```
request_security_scope(user)          # 五维 scope 的**唯一签发点**
        │
        ▼
tenancy.content_scope(...)            # 展开成"可见 tenant_ids / owns_tenant_ids / …"
        │
        ▼
exclude_test_tenants(...)             # 剔除测试公司租户的内容（任何账号都看不到）
        │
        ▼
retrieve_chunks_scoped(query, top_k, scope)   # ← /eval/run 端点与 harness 都走这一条
```

**硬约束**：`run_eval_baseline.py` 里三处（`resolve_labels` / `check_negatives` /
`measure_band_headroom`）共用 `main()` 里**一次签发**的同一个 `UserScope`，且检索
一律走 `retrieve_chunks_scoped`。**绝不能**让某个辅助函数自己按旧的三维口径展开：

* 用**更宽**的口径（如 `owner_id=None` 的伪全量）→ 搜到调用者本无权看的内容，
  负例检查把"权限上看不到"误判成"库里不存在"，退化成永远通过的假测试；
* 用**更窄**的口径 → 把"权限上看不到"误判成"检索漏了"，制造假红。

> 为什么反复强调：本项目 rev3 之前的"假红"正是这么来的 —— 检索质量其实没退化
> （`single_evidence` 切片全程 1.0），是**权限范围收窄**让一条用例变得不可达。

---

## 3. 命中率类指标**只对"可达标注"有意义**

`recall@k` / `hit_rate@k` / `mrr` / `ndcg@k` 这些指标的分母是"金标里的相关块"。
如果一条标注所在的文档**调用者根本无权检索**，那么"没命中"反映的是**权限事实**，
不是**检索质量**。把它混进 `relevant` 会让一条不可达用例显示成"recall 0/2 的检索
失败"—— 与"检索真的漏了"在输出上**完全无法区分**，正是本项目最忌讳的静默失效。

因此 `resolve_labels()` 对每条标注做**可达性复核**，用的是**与检索链路同一个判定内核**：

```python
from app.services.security_policy import ObjectACLView, allows, build_predicate

pred = build_predicate(scope)                       # 整轮只构造一次、复用
view = ObjectACLView.from_row(doc_object_row)       # document_objects 里 object_type='doc' 那行
decision = allows(pred, view)                       # 与检索 ACL 预过滤逐字同源
if not decision.allowed:
    unattainable.append({... "gate": decision.gate, "reason": decision.reason})
```

规则：

* **任一**标注不可达 → 该标注记入 `unattainable`（含 `gate` / `reason`），**不进 `relevant`**；
* 一条用例的**全部**标注都不可达 → **整条用例从评分集剔除**（不留一个必然 0 召回的用例
  污染 recall），但**不判回归** —— 权限变更不是检索质量退化；
* `unattainable` 恒**可见**（stdout + `--out` 顶层键 + `verdict.unattainable`），
  否则消费方分不清"标注在范围外"与"没实现这个字段"。

当前已知不可达：用例"紫罗兰计划的年度预算是多少，由谁负责审批？"的两条标注位于
**测试公司租户**（`company_registry.test_tenant_ids()`），被 `content_scope` 剔除 →
admin 口径下 `gate='tenant' / reason='tenant_mismatch'`。要真正覆盖该场景，需按
**测试公司成员 scope** 单独评测（多 scope 评测尚未实现）。

---

## 4. `run_eval_baseline.py` 用法与产物

### 4.1 运行（容器内 / 能起真实栈的机器）

```bash
# 口令刻意不进仓库，从环境变量读
RAG_EVAL_PASSWORD='…' python -u backend/scripts/run_eval_baseline.py \
    --golden /app/eval/golden_v1.json \
    --base-url http://127.0.0.1:8000 \
    --out /tmp/eval_baseline_report.json \
    --top-k 10
```

环境变量：`RAG_EVAL_USERNAME`（默认 `admin`，需 `audit.read` 权限）、`RAG_EVAL_PASSWORD`
（**必填**）、`RAG_BASE_URL`、`RAG_EVAL_GOLDEN`、`RAG_EVAL_SKIP_BAND`。
参数：`--golden` / `--base-url` / `--out` / `--top-k` / `--skip-band`。

> `--skip-band` 只是**排障**用的加速开关，会少一道护栏（分带余量实测）；
> **CI / 发布门禁不应加它**。

### 4.2 退出码（**非 0 一律阻断**）

| 码 | 含义 |
| --- | --- |
| `0` | 通过（且本轮已落 `eval_runs` 表，跨重启可查） |
| `2` | **检索质量回归**：任一 threshold 未达 / 负例未拒答 / 分带余量不足 / 评测未落库 |
| `3` | 登录或配置失败（如缺 `RAG_EVAL_PASSWORD`、评测账号不存在） |
| `4` | 金标解析失败（`golden_v1.json` 引用的文档不在库） |
| `5` | 评测请求失败 |

### 4.3 `--out` 产物字段（顶层，只增不改）

```jsonc
{
  "golden_set": "golden-v1",          // 金标集名
  "generated_at": "…Z",               // UTC 时间戳
  "wall_seconds": 123.4,              // 整轮墙钟耗时
  "report": { … },                    // POST /eval/run 的原始回执：total_cases /
                                      //   scored_cases / recall / precision / ndcg /
                                      //   mrr / map / evidence_slices …
  "negatives": [                      // 逐条负例拒答检查（与 /eval/run 同源 scope）
    {"query": "…", "returned": 0, "top_score": 0.0, "top_document_id": null,
     "threshold": 0.05, "refused": true, "note": "…"}
  ],
  "band": {                           // 分带余量实测（态 B：关闭阈值过滤）
    "method": "…", "measured_state": "B（过滤关）",
    "cases": [ {"query","gold_count","found_without_filter",
                "best_score","min_gold_ratio","gold_scores"} ],
    "safe_ceiling": 0.062, "cases_total": 5,
    "cases_fully_found_without_filter": 5, "conservatism": "…"
  },
  "unattainable": [                   // 标注在权限范围外的事实（不参与 passed）
    {"query","filename","gate","reason"}
  ],
  "verdict": {                        // 合并结论
    "passed": true,
    "failures": [],                   // 未通过项的可读原因
    "metrics": [ {"metric","value","floor","ok"} ],   // 逐指标对账
    "band": {"current_ratio","declared_ceiling","guard_constant",
             "measured_ceiling","margin"},
    "unattainable": {"count": 0, "items": []},
    "negative_cases_total": 3, "negative_cases_refused": 3, "negative_gate_ok": true,
    "eval_persisted_to_db": true       // /eval/history 里能看到本轮
  }
}
```

---

## 5. 阈值、分带护栏与安全余量现状

`golden_v1.json` 的 `thresholds`（门禁下限，**是安全余量后的下限，不是目标值**）：

| 判据 | 下限 | 说明 |
| --- | --- | --- |
| `min_recall_at_10` | `1.0` | 可达正例必须全部召回 |
| `min_mrr` | `0.95` | 首条命中不得退化 |
| `min_precision_at_3` | `0.75` | 防"为召回堆噪声进 top-3" |
| `min_multi_evidence_all_found_rate` | `1.0` | 分带不得误杀**次要**证据（行为侧判据） |
| `max_rerank_min_score_ratio` | `0.07` | `RERANK_MIN_SCORE_RATIO` 必须**严格小于**它（数值侧判据） |
| 负例 | 全部拒答 | 已有负例必须全部 `refused` |

分带护栏由**三道**判据从松到紧共同守，`judge()` 同步对账：

1. **金标集声明上界** `max_rerank_min_score_ratio = 0.07`（静态、可评审）；
2. **代码护栏常量** `config.RERANK_MIN_SCORE_RATIO_CEILING = 0.07` —— 必须与①**逐位一致**，
   否则两处声明各自漂移，其中一处形同虚设；
3. **本轮实测上界** `band.safe_ceiling`（态 B，随模型/语料实时变化）。

### 余量现状（**薄，且在侵蚀 —— 换模型/语料后必须重测**）

| 量 | 值 | 位置 |
| --- | --- | --- |
| 相对分带当前值 | `RERANK_MIN_SCORE_RATIO = 0.05` | 配置 |
| 实测分带上界 | **0.062** | `baseline_v1.json.band_headroom_measured_rev3`（rev4 复测同值） |
| 声明/护栏上界 | `0.07`（`RERANK_MIN_SCORE_RATIO_CEILING`） | 配置 + 金标 |
| 绝对下限（精排） | `RERANK_MIN_SCORE = 0.05` | 配置 |
| **最弱合法次要证据** | **0.0541** | 实测（头名 0.8724 × ratio 0.062） |
| 最弱证据 / 下限 | **1.08 倍** | 同上 |

两点必须记住：

* **分带上界 0.062 < 声明的 0.07**：声明值偏保守（对护栏方向正确），当前 `ratio=0.05`
  仍**通过**（`0.05 < 0.062`），所以**本次不动任何阈值** —— 按实测去把 `0.07` 收到
  `0.06` 只会把仅剩约 3% 的噪声余量当门槛，反而更脆；
* **绝对下限余量只剩 1.08 倍**（rev3 校准时还是 1.16 倍）。这意味着 `PARENT_SCORE_DECAY`
  （现 `0.85`）一旦下调，同父第 2 条证据就会掉到 `RERANK_MIN_SCORE` 之下 →
  某条多证据用例 `recall@10` 立刻掉到 0.5，门禁随即变红。**动精排模型 / 语料 /
  `PARENT_SCORE_DECAY` 前后，必须重跑本项并对齐常量。**
* **rev4 复跑复现了同一组数字**：分带上界仍 **0.062**、最弱合法次要证据仍 **0.0541**
  （头名 0.8724 × 0.062），比值仍 1.08 倍 —— 与 rev3 **逐位一致**。这说明"余量变薄"
  是**语料 / 模型侧的稳定属性，不是单次测量的偶然抖动**，不能当噪声放过。两份实测
  记录分别在 `baseline_v1.json.band_headroom_measured_rev3` 与 `revalidation_rev4`。

回归哨兵（**必须长期留在金标里**）：

* 单证据哨兵：用例"各期交付物和完成时间分别是什么？"对应块精排分曾低至 **0.22**，
  是历史上被阈值误杀的那一块；
* 多证据哨兵：用例"元组能不能被修改？打开文件时怎么指定编码？"的**次要**证据过带后
  仅 **0.0582**，`ratio` 一旦回到 `0.10`，这条合法证据会被整条砍掉（实测召回 2→1）。

---

## 6. 为什么不能直接比数字（跨版本口径）

| 版本 | 用例数 | `ratio` | 备注 |
| --- | --- | --- | --- |
| rev1 / rev2 | 11 | 0.10 | 金标块**全是精排第 1 名** → 分带风险**结构性地测不出** |
| rev3 / rev4 | 16（15 可达 + 1 不可达剔除） | 0.05 | 新增 5 条多证据用例，当场测出旧 `ratio=0.10` 在误杀次要证据 |

`recall@1 = 0.8333` 不是缺陷：10 个单证据用例各得 1.0，5 个双证据用例在 k=1 时
**最多只能命中两条金标中的一条**（每例上限 0.5），`(10 + 5×0.5)/15 = 0.8333`；
k≥3 起全部为 1.0。这是**金标结构决定的**，不是检索能力。

---

## 7. 相关脚本

| 脚本 | 作用 |
| --- | --- |
| `scripts/run_eval_baseline.py` | 本目录的门禁执行器（见 §4） |
| `scripts/golden_clean_eval.py` | 金标清洗 / 快照辅助 |
| `scripts/snapshot_golden_semantics.py` | 固化金标语义快照 |
| `scripts/compare_scope_eval_15.py` | 不同 scope 下的评测对照（多 scope 评测的探路） |

CI 接入见仓库根 `.github/workflows/backend-ci.yml`：**不依赖真实栈**的门禁（全量
pytest + 静态检查 + 隔离/权限子集）在 CI 里真跑；真实栈的检索质量门禁单列为
`eval-release-gate` job 并默认禁用（依赖 postgres/qdrant/ollama/keycloak + 口令，
托管 runner 起不动），由发布机显式执行、**非 0 即阻断发布**。
