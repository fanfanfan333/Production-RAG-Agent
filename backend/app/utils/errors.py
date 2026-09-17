"""
统一的"异常 → 用户可读消息"转换.

背景（为什么需要这个模块）
------------------------
服务层习惯用 ``raise KeyError("文档不存在或无权访问")`` 表达"查不到"，
路由层再 ``except KeyError as exc: detail=str(exc)``。但 Python 的
``str(KeyError("x"))`` 返回的是 ``"'x'"`` —— **带一对单引号**，于是前端
弹出的提示会变成 ``'文档不存在或无权访问'``，观感像程序出错而不是业务提示。

``clean_message`` 把这类异常统一成干净的中文消息，路由层只需调用它，
不必各自记得 ``exc.args[0]`` 这种细节。
"""

from __future__ import annotations

from typing import Any

DEFAULT_MESSAGE = "操作失败，请稍后再试"


def clean_message(exc: BaseException | Any, default: str = DEFAULT_MESSAGE) -> str:
    """
    返回异常的用户可读消息（KeyError 不带引号）.

    - ``KeyError``：取 ``args[0]``，空参数时回退 *default*；
    - 其他异常：``str(exc)``，空字符串时回退 *default*。
    """
    if isinstance(exc, KeyError):
        if exc.args:
            return str(exc.args[0])
        return default
    text = str(exc).strip()
    return text or default


__all__ = ["clean_message", "DEFAULT_MESSAGE"]
