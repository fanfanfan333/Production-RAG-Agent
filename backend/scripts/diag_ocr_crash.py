"""定位 PaddleOCR 原生崩溃（SIGSEGV）的触发图.

背景：入库 嵌入式软件成神手册.docx 时整个 API 进程被 Paddle 的
``FatalError: Segmentation fault`` 带走（容器重启、文档留下 stuck_recovered）。
崩溃发生在原生层，Python 侧 try/except **抓不住**，所以只能先找出哪张图触发。

用法（**必须在子进程里跑**，它会崩）：

    docker exec -e HOME=/app/.cache rag_backend \
        python /tmp/ragscripts/diag_ocr_crash.py <图片目录>

逐图打印 ``TRY <name>`` 后再跑"预处理 + PaddleOCR"；进程崩掉时，最后一行
``TRY`` 就是元凶。全部跑完会打印 ``ALL OK``。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 必须先落实 paddle_env（导入顺序守卫 + HOME 修正），再碰 paddleocr
from app.utils.paddle_env import build_paddle_ocr
from app.services.image_understanding.preprocess import pick_mode, preprocess

from PIL import Image


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    files = sorted(src.glob("*.png")) + sorted(src.glob("*.jpg"))
    print(f"共 {len(files)} 张图，目录 {src}", flush=True)

    ocr = build_paddle_ocr()
    if ocr is None:
        print("PaddleOCR 不可用")
        return 2

    for f in files:
        try:
            im = Image.open(f).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {f.name}: {exc}", flush=True)
            continue

        mode, _meta = pick_mode(im, None)
        pre = preprocess(im, mode=mode)
        target = pre.ocr_image if pre.ocr_image is not None else pre.image
        print(
            f"TRY {f.name} raw={im.size} mode={mode} -> {target.size} {target.mode}",
            flush=True,
        )

        import numpy as np

        arr = np.array(target.convert("RGB"))[:, :, ::-1].copy()
        out = ocr.ocr(arr, cls=False)
        n = len(out[0]) if out and out[0] else 0
        print(f"    ok lines={n}", flush=True)

    print("ALL OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
