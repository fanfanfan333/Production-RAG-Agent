"""评测金标集与回归基线（版本化资产，不是脚本的副产品）.

为什么单独成目录
────────────────
金标集是**唯一**能回答"这次改动是变好还是变坏"的东西：向量库、切分策略、
精排阈值、多查询开关每动一次，都要用同一套问题追问一遍。它必须进版本管理、
必须能被 review（"这条标注对不对"是业务问题，不是代码问题），因此不能塞在
某个探针脚本里。

为什么标注里写文件名而不是 UUID
────────────────────────────────
``document_id`` 是每次上传时生成的：语料重新导入一次，UUID 全变，金标集体
就废了。这里统一用 ``{filename, chunk_index}`` 作为**逻辑坐标**，由
``scripts/run_eval_baseline.py`` 在运行时反向解析成 ``document_id::chunk_index``。
代价是要求同一份语料不改名 —— 这个约束是显式的、可检查的（解析不到会直接
报错指出是哪个文件），比"UUID 悄悄对不上、分数悄悄变低"好得多。
"""

__all__ = ["load_golden_set"]


def load_golden_set(path: str | None = None) -> dict:
    """读取金标集 JSON（默认取本目录下的 ``golden_v1.json``）."""
    import json
    from pathlib import Path

    p = Path(path) if path else Path(__file__).with_name("golden_v1.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)
