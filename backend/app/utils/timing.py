"""阶段计时装饰器：把"某一段用了多久"接进监控.

为什么需要一个装饰器（而不是在函数体里手写 ``t0 = time.perf_counter()``）
────────────────────────────────────────────────────────────────────────
本项目原先只有 ``record_latency("total", ...)`` 一处埋点。后果是监控面板能说
"这轮问答 87 秒"，但**说不出这 87 秒花在哪** —— 于是真正的问题（例如 Ollama
因为 num_ctx 不一致每次重载白等 11 秒）在面板上完全不可见，只能靠反读日志或
猜。延迟归因必须变成默认能力，而不是一次性的排查动作。

手写计时的问题在于"漏"：
  * 函数里有多个 ``return``（提前返回、空结果、fail-closed 分支）时，很容易只在
    最后那个 return 前记一笔，于是最有诊断价值的"为什么啥都没召回"反而没有耗时；
  * 抛异常时那一笔通常不会记 —— 而超时/失败路径恰恰是最需要计时的。

``try/finally`` 包裹整段逻辑则天然覆盖所有出口（含异常），且调用点只多一行。

用法::

    from app.utils.timing import timed_stage

    @timed_stage("retrieval")
    async def retrieve_chunks(...): ...

阶段名会以 ``latency.<stage>`` 出现在 ``/quality/stats?window=session`` 的
``latency_ms`` 里，无需改动 API 层。

惰性导入 ``record_latency``：``monitoring_service`` 会拉起 ORM 依赖，而这个工具
模块处在更底层的位置，顶层导入容易形成"监控 → ORM → 业务 → 监控"的环。
"""
from __future__ import annotations

import functools
import time
from typing import Any, Callable


def _record(stage: str, seconds: float) -> None:
    try:
        from app.services.monitoring_service import record_latency

        record_latency(stage, seconds)
    except Exception:  # noqa: BLE001
        # 监控是旁路：它坏了绝不能让被监控的主链路失败。这与项目里
        # "改写失败回退原查询""精排失败保持粗排顺序"是同一取向。
        pass


def timed_stage(stage: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """给 async 函数加上 ``latency.<stage>`` 计时的装饰器."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                _record(stage, time.perf_counter() - started)

        return wrapper

    return decorator


__all__ = ["timed_stage"]
