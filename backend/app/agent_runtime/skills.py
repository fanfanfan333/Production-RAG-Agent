"""内置只读 Skill 运行时（声明式技能契约）.

## 为什么不是「会执行代码的插件」

企业内网的检索问答里，Skill 的价值是**把某一类问题的作答纪律固化成可审计的
文本契约**（该引哪一步、要不要分行、什么才算冲突），而不是给模型新的执行权限。
所以这里的 Skill 是纯声明式数据：

* 没有 entrypoint / import 路径 / 命令行 / 网络地址；
* 注册表在模块导入时冻结，运行期 `register_skill()` 直接抛 `RuntimeError`；
* `version` 是内容契约的一部分 —— 改 `instruction` 必须同时改 `version`，
  否则审计日志里「用的是哪一版技能」就失去意义。

## 为什么选择过程不调 LLM

普通问答的额外成本必须是 0。因此技能选择是**确定性打分**（意图 + 关键词 +
证据文档数），不花一次模型调用；打分结果连同命中理由一起进 trace，可复核。

## 与 Anthropic「Agent Skills」开放格式的关系

开放格式的核心是「元数据 → 说明 → 资源」的渐进式披露。本模块落地的是第 1、2 层：
`SkillSpec` 全部字段即元数据，`instruction` 即注入提示的那一段说明。第 3 层
（按需加载的资源包）在纯文本 RAG 链路里没有收益，故不引入 —— 不为了对齐格式
而增加一层无用的间接。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "SkillSpec",
    "SkillResolution",
    "BUILTIN_SKILLS",
    "SKILL_REGISTRY_LOCKED",
    "list_skills",
    "get_skill",
    "register_skill",
    "resolve_skill",
]

# 注入给模型的一段技能说明最多这么长 —— 防止后续有人往 instruction 里塞长文，
# 静默吃掉证据预算。
MAX_INSTRUCTION_CHARS = 600

# 打分门槛：至少命中 1 个触发词（2 分）才认为技能适用。
# 意图匹配（+2）与多文档（+2）只是加权，不能单独把技能拉起来 —— 否则
# 「所有 knowledge_qa 都套 exact_lookup」就等于没有技能。
MIN_SCORE = 2


@dataclass(frozen=True)
class SkillSpec:
    """一个只读技能的内容契约.

    Attributes:
        name: 稳定标识（进审计日志，不要改）。
        version: 内容版本；`instruction` 变了就必须变版本。
        title: 人读的中文名。
        description: 这个技能解决什么问题、不解决什么。
        intents: 适用的路由意图；不在列表里的意图一律不选它。
        triggers: 触发词（中英混排，大小写不敏感）。
        instruction: 追加到生成提示里的作答纪律（**只能是常量**）。
        min_docs: 至少需要几份不同文档才允许触发。
    """

    name: str
    version: str
    title: str
    description: str
    intents: tuple[str, ...]
    triggers: tuple[str, ...]
    instruction: str
    min_docs: int = 1
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SkillResolution:
    """一次技能选择的结果（含打分依据，用于审计）."""

    skill: SkillSpec
    score: int
    reason: str
    matched_triggers: tuple[str, ...]

    def as_audit(self) -> dict:
        return {
            "name": self.skill.name,
            "version": self.skill.version,
            "title": self.skill.title,
            "score": self.score,
            "reason": self.reason,
            "matched_triggers": list(self.matched_triggers),
        }


# ── 内置技能（只读） ─────────────────────────────────────────────────────────
#
# 三个技能对应三类**作答纪律确实不同**的问题。刻意不做的：
#   * 不做「打分类」「闲聊类」技能 —— 它们没有可固化的纪律，只会稀释提示；
#   * 不做跨技能叠加 —— 叠加会让指令互相冲突，且无法归因是哪条起了作用。

_EXACT_LOOKUP = SkillSpec(
    name="exact_lookup",
    version="1.0.0",
    title="单点事实精确取值",
    description="问的是某个实体某个属性上的一个值（数字/日期/条款号/阈值）。",
    intents=("knowledge_qa",),
    triggers=(
        # 刻意不写裸 "几"：它会把「几份」「几乎」「几个」全判成取值题，
        # 而 exact_lookup 的纪律是"只回答那一个值"——误套到归纳题上等于让模型
        # 少答一半。触发词越短越容易误伤，这里一律给足上下文。
        "多少", "几号", "几天", "几点", "几次", "何时", "什么时候", "哪一年", "哪一天",
        "版本", "阈值", "上限", "下限", "比例", "单价", "总额", "周期",
        "how many", "how much", "when", "version", "threshold", "limit",
    ),
    instruction=(
        "[技能 exact_lookup v1.0.0：单点取值]"
        "只回答被问的那一个值，先给数值/日期，再给一句出处说明；"
        "数值必须与引用原文逐字一致，不要换算、不要四舍五入、不要补单位；"
        "原文没有明确写出这个值时，直接说明「原文未给出」，不要用相近数字顶替。"
    ),
    min_docs=1,
    tags=("deterministic", "prompt-only"),
)

_CROSS_DOC_COMPARE = SkillSpec(
    name="cross_document_compare",
    version="1.0.0",
    title="跨文档对比与冲突识别",
    description="要求比较多份文档在同一实体同一属性上的取值，或判断是否冲突。",
    intents=("knowledge_qa", "doc_relations"),
    triggers=(
        "对比", "比较", "差异", "区别", "不同", "冲突", "不一致", "分别",
        "哪个更", "谁更", "相比", "vs", "versus", "compare", "differ", "conflict",
    ),
    instruction=(
        "[技能 cross_document_compare v1.0.0：跨文档对比]"
        "每一步只陈述一条关系并紧跟支持它的引用，逐步换行；"
        "只有同一实体的同一属性出现互不相容的取值才算冲突，实体或属性不同不算；"
        "确有冲突时并列双方原文与引用，最后写「无法确定，需要核实」；"
        "一致时明确给出结论，不追加不确定性说明。"
    ),
    min_docs=2,
    tags=("deterministic", "prompt-only"),
)

_EVIDENCE_SUMMARY = SkillSpec(
    name="evidence_summary",
    version="1.0.0",
    title="多源要点归纳",
    description="要求把多份材料归纳成要点清单，而不是回答单个事实。",
    intents=("document_summary", "doc_relations", "knowledge_qa"),
    triggers=(
        "总结", "归纳", "梳理", "汇总", "要点", "概览", "清单", "列出",
        "summarize", "summary", "overview", "list",
    ),
    instruction=(
        "[技能 evidence_summary v1.0.0：多源归纳]"
        "按「要点 → 支持该要点的引用」逐条列出，每条不超过两句；"
        "只用检索片段里出现过的事实，不补充背景知识、不做因果推断；"
        "片段之间互相矛盾时单列一条并标出双方来源，不要自行调和。"
    ),
    min_docs=1,
    tags=("deterministic", "prompt-only"),
)

BUILTIN_SKILLS: tuple[SkillSpec, ...] = (
    _EXACT_LOOKUP,
    _CROSS_DOC_COMPARE,
    _EVIDENCE_SUMMARY,
)

_BY_NAME: dict[str, SkillSpec] = {skill.name: skill for skill in BUILTIN_SKILLS}

# 注册表冻结标记。运行期不允许注册新技能 —— 「在线自改能力」是明确非目标，
# 用一行显式常量把它钉死，比写在文档里更可靠。
SKILL_REGISTRY_LOCKED = True


def _validate(skill: SkillSpec) -> None:
    if not skill.name or not skill.version:
        raise ValueError("skill 必须有 name 与 version")
    if len(skill.instruction) > MAX_INSTRUCTION_CHARS:
        raise ValueError(
            f"skill {skill.name} 的 instruction 超过 {MAX_INSTRUCTION_CHARS} 字符，"
            "会挤占证据预算"
        )
    if not skill.intents or not skill.triggers:
        raise ValueError(f"skill {skill.name} 缺少 intents/triggers")


for _skill in BUILTIN_SKILLS:  # 导入即校验，坏数据不该等到线上才炸
    _validate(_skill)


def list_skills() -> list[dict]:
    """给 API / 审计用的技能清单（不含 instruction 全文之外的任何可执行信息）."""
    return [
        {
            "name": skill.name,
            "version": skill.version,
            "title": skill.title,
            "description": skill.description,
            "intents": list(skill.intents),
            "min_docs": skill.min_docs,
            "tags": list(skill.tags),
            "instruction_chars": len(skill.instruction),
        }
        for skill in BUILTIN_SKILLS
    ]


def get_skill(name: str) -> SkillSpec | None:
    return _BY_NAME.get(name)


def register_skill(skill: SkillSpec) -> None:  # pragma: no cover - 防线本身
    """运行期注册技能 —— 永远拒绝.

    保留这个函数（而不是干脆不写）是为了让「想加技能的人」撞到一堵有解释的墙：
    新增技能 = 改代码 + 升版本 + 过评审，不是运行时一句调用。
    """
    if SKILL_REGISTRY_LOCKED:
        raise RuntimeError(
            "Skill 注册表已冻结：新增技能必须改代码并升版本，不支持运行期注册"
        )
    _validate(skill)
    _BY_NAME[skill.name] = skill


def _count_matches(query_lower: str, triggers: tuple[str, ...]) -> tuple[str, ...]:
    """返回命中的触发词（去重、保序）."""
    hits: list[str] = []
    for trigger in triggers:
        if trigger.lower() in query_lower and trigger not in hits:
            hits.append(trigger)
    return tuple(hits)


def resolve_skill(
    *,
    intent: str,
    query: str,
    document_count: int,
    enabled: bool = True,
) -> SkillResolution | None:
    """确定性选择技能；不适用时返回 ``None``（调用方据此完全不加指令）.

    Args:
        intent: 路由意图（`knowledge_qa` / `doc_relations` / `document_summary` …）。
        query: 用户原始问题（**不要**传改写后的查询：触发词要贴合用户措辞）。
        document_count: 本轮证据覆盖的不同文档数。
        enabled: 总开关；关闭时恒返回 ``None``。

    Returns:
        得分最高且唯一领先的 `SkillResolution`；并列第一视为「不确定」而放弃选择 ——
        两个技能同时套上去，指令会互相打架，归因也会失效。
    """
    if not enabled or not query:
        return None
    query_lower = query.lower()
    scored: list[SkillResolution] = []

    for skill in BUILTIN_SKILLS:
        if intent not in skill.intents:
            continue
        if document_count < skill.min_docs:
            continue
        hits = _count_matches(query_lower, skill.triggers)
        if not hits:
            continue
        score = 2 * min(len(hits), 3) + 2  # 意图/文档数已满足，给基础分 2
        reason = f"intent={intent}; docs={document_count}>={skill.min_docs}; " \
                 f"triggers={'/'.join(hits)}"
        scored.append(SkillResolution(skill, score, reason, hits))

    if not scored:
        return None
    scored.sort(key=lambda item: (-item.score, item.skill.name))
    if len(scored) > 1 and scored[0].score == scored[1].score:
        return None
    return scored[0]


def render_instruction(resolution: SkillResolution | None) -> str:
    """把技能指令渲染成可加在问题前面的文本；无技能时返回空串."""
    if resolution is None:
        return ""
    return resolution.skill.instruction
