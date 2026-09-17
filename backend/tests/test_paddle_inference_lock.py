"""Paddle 原生推理必须**串行化**（防止并发 SIGSEGV 打死整个 API）.

背景（2026-09-14 实修）：入库改为"202 受理 + 后台 asyncio 任务"之后，多份文档
的 OCR 会经 ``asyncio.to_thread`` 真的并行跑在不同线程上，而它们共用同一个
PaddleOCR predictor 实例。Paddle 的 predictor **不是线程安全的**，并发推理会
写坏原生内存：

    could not create a primitive descriptor for a reorder primitive
    InvalidArgumentError: Broadcast dimension mismatch ... db_fpn.py:249
    FatalError: `Segmentation fault` is detected by the operating system.

最后一条是 SIGSEGV —— 原生崩溃，Python 侧 try/except 抓不住，**整个 API 进程
直接消失**；容器重启后文档被标成 FAILED(stuck_recovered)，日志里连 Python
异常栈都没有。复现脚本见 ``backend/scripts/repro_paddle_threads.py``。

本测试不加载真实模型：注入一个假 predictor，断言

1. 三套 Paddle 引擎用的是**同一把**进程级锁（分别上锁等于没上）；
2. 多线程并发调用 ``process()`` 时，锁确实被持有，且**任何时刻只有一个**在跑。

跑法（容器内）：

    docker exec -u root -e HOME=/tmp rag_backend \
        sh -c "cd /app && python -m pytest tests/test_paddle_inference_lock.py -q"
"""
from __future__ import annotations

import threading
import time

from PIL import Image

from app.utils import paddle_env


def _img(size: tuple[int, int] = (80, 24)) -> Image.Image:
    return Image.new("RGB", size, (255, 255, 255))


class _FakePredictor:
    """记录并发度与"调用时锁是否被持有"的假 predictor."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self.inflight = 0
        self.max_inflight = 0
        self.calls = 0
        self.lock_held_on_entry = 0
        self.saw_lock: list[bool] = []

    def ocr(self, *_args, **_kwargs):
        lock = paddle_env.predict_lock()
        held = lock.locked()
        with self._guard:
            self.calls += 1
            if held:
                self.lock_held_on_entry += 1
            self.saw_lock.append(held)
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        time.sleep(0.002)          # 放大竞争窗口
        with self._guard:
            self.inflight -= 1
        # 返回结构要能被解析：[[ [coords, (text, conf)] ]]
        return [[[[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]], ("hi", 0.9)]]


def test_paddle_engines_share_one_process_wide_lock():
    """三套引擎 + 旧 provider 必须是同一把锁."""
    from app.services.image_understanding.engines import paddle_engines
    from app.services.ocr import paddle_provider

    shared = paddle_env.predict_lock()
    assert paddle_engines._predict_lock is shared
    # 旧 provider 走的是 paddle_env.predict_lock() 函数调用，同一对象
    assert paddle_provider.predict_lock() is shared


def test_concurrent_process_calls_are_serialised():
    """多线程并发 → 任一时刻只有一个推理在跑，且锁在调用点是持有的."""
    from app.services.image_understanding.engines import paddle_engines

    fake = _FakePredictor()
    saved = paddle_engines._ocr_instance
    paddle_engines._ocr_instance = fake
    try:
        engine = paddle_engines.PaddleOCREngine()
        assert engine.is_available() is True

        errors: list[BaseException] = []

        def run() -> None:
            try:
                for _ in range(8):
                    out = engine.process(_img())
                    assert out.ok, out.error
            except BaseException as exc:      # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        paddle_engines._ocr_instance = saved

    assert not errors, errors
    assert fake.calls == 6 * 8
    # 核心断言：没有任何一次推理是"锁外"发生的，也没有两个同时在跑
    assert fake.lock_held_on_entry == fake.calls, (
        f"{fake.calls - fake.lock_held_on_entry} 次推理没持锁 —— "
        "并发调用会 SIGSEGV 打死进程"
    )
    assert fake.max_inflight == 1, f"并发度 {fake.max_inflight} > 1，Paddle 会被写坏"
