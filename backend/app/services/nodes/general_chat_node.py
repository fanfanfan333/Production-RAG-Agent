"""
General Chat 节点（架构图 General Chat 分支）.

闲聊/创作/常识类问题走这条分支：不检索知识库，直接让 LLM 回答。

为什么必须单独一条分支
──────────────────────
以前所有非关系类问题都进 RAG 主链路，导致：
- "你好"、"帮我写首诗" 这种问题也会去检索，召回一堆不相关 chunk；
- 检索分数必然低于阈值 → 幻觉守卫直接拒答 → 用户被"抱歉，知识库里
  没有相关信息"怼回来，体验很差。

本分支的关键约束（prompt 里强制）
──────────────────────────────────
- 明确告诉模型"本轮没有知识库上下文"，杜绝它编造 [Source N] 引用；
- 禁止说"根据知识库…"、"我查到…"这类措辞 —— 这是最强的幻觉来源；
- 保持简洁，不过度展开。
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


_GENERAL_CHAT_SYSTEM_PROMPT_ZH = """\
你是一个企业私有知识库问答助手的对话模块。当前这轮对话走的是"通用聊天"分支。

关键事实（违反即为错误）：
1. 本轮**没有**检索任何知识库文档，你手上没有任何文档上下文。
2. 因此你**绝对不能**输出任何 [Source N] 引用标记。
3. 你也**绝对不能**说"根据知识库"、"我查到"、"文档中提到"这类措辞 ——
   这轮根本没查，说了就是幻觉。
4. 如果问题其实需要查知识库才能回答（例如问某个具体文档里的数据），
   直接告诉用户："这个问题需要查询知识库才能回答，请换一种明确的提问方式
   或确认相关文档已上传。" 不要凭常识硬答。

行为准则：
- 闲聊、打招呼、身份询问、创作（写诗/写代码/翻译）、通用常识问题：
  正常、友好、简洁地回答。
- 保持简洁：除非用户明确要求，不要主动长篇展开。
- 使用与用户相同的语言（默认中文）。
- 不要复述上一轮已经说过的内容作为开场。
"""

_GENERAL_CHAT_SYSTEM_PROMPT_EN = """\
You are the general-chat module of an enterprise knowledge-base assistant.

Key facts (violating any of these is an error):
1. No document was retrieved this turn — you have NO document context.
2. You must NEVER emit [Source N] citation markers.
3. You must NEVER say "according to the knowledge base", "I found in the docs",
   or similar — nothing was retrieved this turn; saying so would be a hallucination.
4. If the question actually requires the knowledge base (e.g. asking for a figure
   from a specific document), tell the user it needs a knowledge-base query
   instead of answering from general knowledge.

Guidelines:
- Casual chat, greetings, identity questions, creative writing, and general
  knowledge: answer normally, friendly and concise.
- Match the user's language.
- Do not restate previous turns as an opening.
"""


def build_general_chat_messages(
    query: str,
    history_messages: list[BaseMessage],
) -> list[BaseMessage]:
    """
    组装 General Chat 分支的 messages.

    注意：这里**不注入任何检索上下文** —— 这正是该分支的意义。
    """
    settings = get_settings()
    lang = (settings.GENERAL_CHAT_SYSTEM_PROMPT_LANG or "zh").lower()
    system_prompt = (
        _GENERAL_CHAT_SYSTEM_PROMPT_EN if lang.startswith("en")
        else _GENERAL_CHAT_SYSTEM_PROMPT_ZH
    )

    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    messages.extend(history_messages)
    messages.append(HumanMessage(content=query))
    return messages


def build_general_chat_llm():
    """构造闲聊用 LLM（temperature 稍高，回答更自然）."""
    settings = get_settings()
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.6,          # 闲聊/创作需要一点多样性
        streaming=True,
        reasoning=True,           # qwen3 思考流照常透出
    )
