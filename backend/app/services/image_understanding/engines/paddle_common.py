"""
Paddle 系引擎的公共前置（转发到 :mod:`app.utils.paddle_env`）.

真正的实现放在 ``app/utils/paddle_env.py`` —— 因为 OCR 包（``services/ocr``）
也要用同一套"导入顺序守卫 + 模型缓存目录 + PP-OCR 版本"逻辑，而这里属于
``image_understanding``，让 ``ocr`` 反向依赖它会引入不必要的耦合甚至循环导入。

本模块只做转发，保证引擎层的 import 路径读起来仍然是"引擎的公共前置"。
"""

from __future__ import annotations

from app.utils.paddle_env import (  # noqa: F401
    build_paddle_ocr,
    ensure_model_home,
    formula_class,
    import_paddleocr,
    paddle_available,
    paddle_lang,
    paddle_ocr_version,
    pp_structure_v2_class,
    pp_structure_v3_class,
    predict_lock,
)

__all__ = [
    "ensure_model_home",
    "paddle_ocr_version",
    "paddle_lang",
    "predict_lock",
    "import_paddleocr",
    "build_paddle_ocr",
    "paddle_available",
    "pp_structure_v3_class",
    "pp_structure_v2_class",
    "formula_class",
]
