"""HuggingFace / BGE 模型缓存目录的启动自检.

────────────────────────────────────────────────────────────────────────────
坑：HF_HOME 指向不可写目录 → 嵌入全量失败，而症状离病因极远
────────────────────────────────────────────────────────────────────────────
容器以 ``appuser`` 运行，模型缓存挂在命名卷（``hf_cache`` → ``/app/.cache``）里。
一旦该目录被 root 占有 —— 典型来源是调试时用 ``docker exec -u root`` 跑过脚本，
或更早版本的镜像/配置留下的目录 —— ``huggingface_hub`` 就会：

1. 找不到 BGE 权重（缓存路径变了），转去下载；
2. 下载时往那个目录写不进去，
   ``PermissionError: [Errno 13] Permission denied: .../hub``；
3. **不把异常抛给调用方**，只在日志里刷 ``Could not cache non-existence of file``。

真正的失败发生在后面：整批 embedding failed，文档被标成 FAILED，落库的错误
信息只有一句 ``13 chunks failed to embed permanently.``。从"入库失败"一路反查
到"缓存目录属主不对"，要翻好几层日志 —— 实测就是这么耗掉半小时的。

所以启动时就把这件事查明白：不可写就直接 ERROR，并把**可执行的修复命令**写进
日志。这与 ``main.py`` 里 Vision 探测的处理方式一致：只告警、不阻塞启动。
"""

from __future__ import annotations

import os
from pathlib import Path

from app.utils.logging import get_logger

logger = get_logger(__name__)


def hf_cache_root() -> str:
    """当前生效的 HF 缓存根目录。

    ``HF_HOME`` 显式设置时以其为准；否则回落到 huggingface_hub 的默认位置
    （``~/.cache/huggingface``）。
    """
    explicit = os.environ.get("HF_HOME")
    if explicit:
        return explicit
    return str(Path("~").expanduser() / ".cache" / "huggingface")


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".hf_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:      # noqa: BLE001 — 任何失败都只意味着"不可写"
        return False


def ensure_hf_cache() -> bool:
    """探测 HF 缓存目录是否可写；不可写时打印可执行的修复指引。

    返回 True 表示可写。**不要**在这里自动改 ``HF_HOME`` 去指向别处 ——
    那会把模型缓存换到另一个目录、每次重启重新下载几 GB 权重，比直接失败
    更难察觉。
    """
    root = Path(hf_cache_root())
    if _writable(root):
        logger.info("HuggingFace cache is writable: %s", root)
        return True

    logger.error(
        "HuggingFace cache %s is NOT writable (uid=%s, gid=%s). "
        "BGE embedding will fail for EVERY document — the visible symptom is "
        "only 'N chunks failed to embed permanently'. "
        "Typical cause: the volume dir is owned by root (e.g. an earlier "
        "`docker exec -u root` created it). Fix once, then restart the backend:\n"
        "    docker exec -u root rag_backend chown -R appuser:appgroup %s\n"
        "    docker restart rag_backend",
        root, os.getuid(), os.getgid(), root,
    )
    return False


def hf_cache_health() -> dict[str, object]:
    """给 ``/health`` 用的轻量摘要（一次 stat，不做写入探测）。"""
    root = Path(hf_cache_root())
    return {
        "path": str(root),
        "exists": root.is_dir(),
        "writable": root.is_dir() and os.access(root, os.W_OK),
    }


__all__ = ["hf_cache_root", "ensure_hf_cache", "hf_cache_health"]
