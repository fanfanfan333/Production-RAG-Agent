"""受控 Agent 运行时（Agent Runtime）.

把「Skill 能力 / 自进化记忆 / 分层上下文 / 条件化多 Agent」四项能力做成**可关、
可审计、失败可回退**的运行时，而不是重写主图。

设计不变量（与 `_audit/agent_evolution/ARCHITECTURE.md` 一致）：

1. **纯逻辑**：本包里除了 `memory_store` 需要 SQLAlchemy，其余模块只依赖标准库，
   不发起网络请求、不加载模型、不写文件 —— 因此可以被纯单元测试直接覆盖。
2. **零额外成本**：普通问答不因为本包多花一次 LLM 调用；唯一可能多花的是
   `review`（条件化多 Agent），且它默认关闭、单次请求至多 1 次调用。
3. **只读 Skill**：Skill 是声明式数据，不是可执行插件；注册表导入即冻结。
4. **记忆隔离**：长期记忆的读取键是 `(tenant_id, user_id)`，没有管理员跨用户通道。
5. **可回退**：任一子能力关闭或抛异常时，主图回落到升级前的行为（见
   `master_graph._generate_node` 的 try/except）。
"""

from __future__ import annotations

__all__ = [
    "skills",
    "context_layers",
    "review",
    "trace",
    "memory_store",
]
