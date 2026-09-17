"""复现 PaddleOCR **并发推理**导致的原生崩溃（SIGSEGV）.

假设：`PaddleOCR` 的 predictor 实例**不是线程安全的**（原生层有共享的
workspace / oneDNN primitive cache）。`asyncio.to_thread` 让多份文档的 OCR
并行跑在不同线程里，同时又共用 `paddle_engines._ocr_instance` 这一个实例，
于是原生内存被两个线程同时改 → 报

    could not create a primitive descriptor for a reorder primitive
    InvalidArgumentError: Broadcast dimension mismatch ... db_fpn.py:249
    FatalError: `Segmentation fault` is detected by the operating system.

而这个异常在最外层 try/except 里**抓不住** —— 它是 SIGSEGV，整个进程直接没了，
容器重启后文档被 recover_stuck_documents 标成 FAILED。

用法（会崩，务必在子进程里跑）：

    docker exec -e HOME=/app/.cache rag_backend \
        python /tmp/ragscripts/repro_paddle_threads.py <图片目录> [线程数]

期望：并发跑 → 几十次调用内进程崩掉（退出码 139 / Segmentation fault）。
加锁串行跑（`LOCK=1`）→ 全部跑完打印 ALL OK。
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from app.utils.paddle_env import build_paddle_ocr

import numpy as np
from PIL import Image


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    nthreads = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rounds = int(os.getenv("ROUNDS", "12"))
    use_lock = os.getenv("LOCK") == "1"

    files = (sorted(src.glob("*.png")) + sorted(src.glob("*.jpg")))[:6]
    if not files:
        print("没有图片")
        return 2

    ocr = build_paddle_ocr()
    if ocr is None:
        print("PaddleOCR 不可用")
        return 2

    arrays = [
        np.array(Image.open(f).convert("RGB"))[:, :, ::-1].copy() for f in files
    ]
    print(f"共享实例 × {nthreads} 线程 × {rounds} 轮，LOCK={use_lock}", flush=True)

    lock = threading.Lock()
    done = [0]
    counter_lock = threading.Lock()

    def worker(tid: int) -> None:
        for r in range(rounds):
            for i, arr in enumerate(arrays):
                if use_lock:
                    with lock:
                        ocr.ocr(arr, cls=False)
                else:
                    ocr.ocr(arr, cls=False)
                with counter_lock:
                    done[0] += 1
            print(f"  t{tid} round {r} ok (total {done[0]})", flush=True)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(nthreads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"ALL OK — {done[0]} 次调用无崩溃", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
