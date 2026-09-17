# -*- coding: utf-8 -*-
"""
Unicode 控制字符 / 同形字骨架 —— 单一事实源（single source of truth）。

为什么单独成模块
────────────────
`prompt_security`（判定）与 `text_cleaning`（清洗）此前**各自维护**一份"隐形
字符"字符类，两份并不相等：判定那份缺 ``\\u00ad`` 与 ``\\u2061-\\u2064``（清洗那份有），
两份都漏掉的方向隔离符、TAG 字符、变体选择符等另有 16 个码位。
2026-09-17 红队实测（37 个码位 × 3 项检查）：**查询侧判定被绕过 21 项**，
清洗侧剥离失败 16 项 —— 前者比后者多出的 5 项正是"判定比清洗还窄"的那几个。
攻击者只需要挑强度弱的那条通道。

把两份定义并成一份常量导出，"通道强度一致"才由结构保证，而不是靠人工同步。

依赖约束
────────
本模块**只依赖 re / unicodedata**，不 import 任何业务代码。这条约束是刻意的：
``text_cleaning`` 可以在模块级导入它，从而不会把 ``prompt_security`` 拖进
``tests/test_async_ingestion.py`` 的模块桩替换范围（那份单测把
``app.services.prompt_security`` 整个换成了桩）。
"""

from __future__ import annotations

import re
import unicodedata

# ── 隐形 / 控制字符（保留 \t 与 \n）──────────────────────────────────────────
# 覆盖原则：**凡是在渲染上不可见、或能被用来把一段文本藏起来 / 把匹配打断的
# 码位，一律剥掉**。历史教训是"按已知攻击手法枚举"永远漏 —— 这里改按 Unicode
# 类别 + 已知隐写块穷举。
INVISIBLE_RE = re.compile(
    "["
    "\u0000-\u0008"              # C0（不含 \t）
    "\u000b-\u001f"              # C0（不含 \n）
    "\u007f-\u009f"              # DEL + C1
    "\u00ad"                     # 软连字符（分词器看不见，正则匹配得到）
    "\u061c"                     # 阿拉伯字母标记（双向控制）
    "\u115f\u1160"               # Hangul Choseong/Jungseong Filler（渲染为空白）
    "\u17b4\u17b5"               # 高棉固有元音（零宽）
    "\u180b-\u180e"              # 蒙古自由变体选择符 1-3 + 元音分隔符
    "\u200b-\u200f"              # 零宽空格/连接符/不连字/方向标记
    "\u202a-\u202e"              # 双向嵌入 / 覆盖 / 弹出
    "\u2060-\u206f"              # 词连接符 .. 方向隔离符(LRI/RLI/FSI/PDI)
    "\u3164"                     # Hangul Filler
    "\ufe00-\ufe0f"              # 变体选择符（可承载隐写二进制）
    "\ufeff"                     # BOM / 零宽不换行空格
    "\uffa0"                     # 半宽 Hangul Filler
    "\U000e0001"                 # 语言标签
    "\U000e0020-\U000e007f"      # TAG 字符（把整句话藏进"看不见的通道"的经典手法）
    "\U000e0100-\U000e01ef"      # 变体选择符补充
    "]"
)

# ── 形似空格的空白 → 半角空格（只动空白，不动标点）───────────────────────────
# 刻意**不**含 U+3000 之外的 CJK 标点：把全角逗号折成半角是格式损失，不是清洗。
SPACE_LIKE_RE = re.compile("[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")

# ── 同形字 → ASCII 骨架 ──────────────────────────────────────────────────────
# 只收**视觉上与被替换拉丁字母几乎无法区分**的码位。不做通用 confusable 表：
# 那会把正常俄语/希腊语正文折成看似英文的串，而这份映射的消费者只有"是否命中
# 攻击规则"这一个用途，宁缺毋滥。
CONFUSABLE_MAP = str.maketrans({
    # 西里尔
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ѕ": "s", "ј": "j", "һ": "h", "ԁ": "d", "ӏ": "l", "ᴠ": "v",
    "м": "m", "т": "t", "п": "n", "н": "h", "к": "k", "в": "b", "г": "r",
    # 希腊
    "α": "a", "ε": "e", "ο": "o", "ρ": "p", "ι": "i", "ν": "v", "τ": "t",
    "κ": "k", "υ": "u", "μ": "m", "ϲ": "c", "Ϲ": "c",
})

# 紧凑骨架里**保留**的字符：ASCII 字母数字 + CJK 基本区 / 扩展 A。
# 其余（空格、标点、转义符、未映射的其它文字）全部丢弃 —— 这正是"打断插入"的
# 对抗手段：`i g n o r e` / `ig-nore` / `ign\\ore` 都会塌缩成 `ignore`。
_COMPACT_KEEP_RE = re.compile(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+")


def compact_skeleton(text: str) -> str:
    """把文本压成"只剩 ASCII 字母数字与汉字"的紧凑小写骨架。

    **仅供判定使用，绝不可用于存储或回显** —— 它丢弃空格与标点，是有损的。

    对抗的四类"换皮"绕过：
      1. 隐形字符插桩：``忽\\u2061略你的指令``
      2. 同形字替换：``Іgnore``（西里尔 І）
      3. 空白/标点拆分：``i g n o r e``、``ig-nore``、``忽 略 你 的 指 令``
      4. HTML 实体 / 转义干扰：``&#105;gnore``、``ign\\\\ore``

    第 4 类还需调用方先做 ``html.unescape``（本函数刻意不做 —— 它是纯字符级
    变换，不应悄悄改变文本的语义长度）。
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text).lower().translate(CONFUSABLE_MAP)
    return _COMPACT_KEEP_RE.sub("", folded)


# 供调用方（prompt_security）判断"这段文本是否含非 ASCII 字母"，
# 用于在**原文已经命中**时跳过骨架计算，避免给正常文档付额外代价。
NON_ASCII_LETTER_RE = re.compile(r"[^\x00-\x7f]")


__all__ = [
    "CONFUSABLE_MAP",
    "INVISIBLE_RE",
    "NON_ASCII_LETTER_RE",
    "SPACE_LIKE_RE",
    "compact_skeleton",
]
