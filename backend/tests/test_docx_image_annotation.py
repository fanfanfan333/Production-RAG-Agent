"""
DOCX 图片占位符标注的回归测试（2026-09-13）.

Docling 的 Markdown 导出不携带图片本体，正文里只留一个 ``<!-- image -->``。
原样入库的后果：用户在"文档内容"里看到的是一串裸 HTML 注释 ——
**图片等于没有被标注**，既不知道是第几张，也不知道图里是什么。

覆盖：

    1. 占位符按出现顺序与图片对象一一对齐，替换成 ［图 N · 类型］说明
    2. 说明文字的优先级：图义描述 > 图内文字 > 结构化结果
    3. 占位符多于图片对象时按序号降级（不越界、不抛异常）
    4. 完全没有图片对象时退化为 ［图片］
    5. 没有占位符的正文原样返回
    6. 超长说明被截断（不能把整张 Markdown 表格塞进正文）

运行方式（容器内）：
    docker exec -e HOME=/tmp rag_backend \
        sh -c "cd /app && python -m pytest tests/test_docx_image_annotation.py -q"
"""

from __future__ import annotations

from app.services.parsers.base import ExtractedImage
from app.services.parsers.docx_parser import (
    _ANNOTATION_MAX_CHARS,
    annotate_image_placeholders,
)


def _image(index: int, **kwargs) -> ExtractedImage:
    return ExtractedImage(
        image_id=f"doc-uuid-p1-i{index}",
        page_number=1,
        position=index,
        **kwargs,
    )


def test_placeholders_replaced_in_order() -> None:
    text = "模拟CIFAR10，搭建网络：\n\n<!-- image -->\n\nimport torch\n\n<!-- image -->"
    images = [
        _image(1, image_type="table", structured_content="| 层 | 通道 |"),
        _image(2, image_type="diagram", vision_caption="风扇系统：input→Fan→output"),
    ]

    result = annotate_image_placeholders(text, images)

    assert "<!-- image -->" not in result, "占位符没有被替换"
    assert "［图 1 · 表格］" in result
    assert "［图 2 · 流程图］风扇系统：input→Fan→output" in result
    # 顺序必须与图片对象一致：图 1 的标注要出现在图 2 之前
    assert result.index("［图 1") < result.index("［图 2")
    # 正文其余内容不能被破坏
    assert "模拟CIFAR10，搭建网络：" in result
    assert "import torch" in result


def test_description_priority_prefers_vision() -> None:
    """给人看的标注要"语义优先" —— 图义描述比 OCR 片段更说明问题。"""
    img = _image(
        1,
        image_type="diagram",
        vision_caption="视觉描述",
        ocr_text="OCR 文字",
        structured_content="结构化内容",
    )
    result = annotate_image_placeholders("<!-- image -->", [img])
    assert "视觉描述" in result
    assert "OCR 文字" not in result


def test_description_falls_back_to_ocr_then_structured() -> None:
    only_ocr = _image(1, ocr_text="只有 OCR", vision_caption=None)
    assert "只有 OCR" in annotate_image_placeholders("<!-- image -->", [only_ocr])

    only_structured = _image(1, structured_content="| a | b |", vision_caption=None)
    assert "| a | b |" in annotate_image_placeholders("<!-- image -->", [only_structured])


def test_blank_description_keeps_label_only() -> None:
    img = _image(1, image_type="photo", vision_caption="   ", ocr_text="")
    result = annotate_image_placeholders("<!-- image -->", [img])
    assert result == "［图 1 · 图片］"


def test_more_placeholders_than_images_degrades() -> None:
    """包内可能有 PIL 解不开的 part，占位符会多于图片对象 —— 不能越界。"""
    text = "<!-- image -->\n<!-- image -->\n<!-- image -->"
    result = annotate_image_placeholders(text, [_image(1, vision_caption="第一张")])
    assert "［图 1 · 图片］第一张" in result
    assert "［图 2］" in result
    assert "［图 3］" in result


def test_source_ordinal_alignment_with_filtered_image() -> None:
    """
    被过滤掉的图**不占号** —— 对齐必须靠 source_ordinals，不能靠位置.

    真实案例（实战.docx）：正文 3 个占位符，但第 2 张图是 262×33 的窄条，
    被 MIN_IMAGE_DIMENSION=60 过滤掉。若按位置对齐，第 3 张图（数据流图）
    的说明会被错标到第 2 个占位符上，而第 3 个占位符只剩一个空编号。
    """
    text = "A<!-- image -->B<!-- image -->C<!-- image -->D"
    images = [
        _image(1, image_type="table", vision_caption="网络结构表"),
        _image(2, image_type="diagram", vision_caption="数据流图 input→Fan→output"),
    ]
    # 第 2 张来源图被过滤 → 收下的是第 1、3 张
    result = annotate_image_placeholders(text, images, [1, 3])

    assert "A［图 1 · 表格］网络结构表B" in result
    assert "C［图 3 · 流程图］数据流图 input→Fan→outputD" in result
    # 被过滤掉的那张：编号保留，但不编造说明
    assert "B［图 2］C" in result
    assert "网络结构表" not in result.split("C［图 3")[1]


def test_source_ordinals_length_mismatch_falls_back_to_position() -> None:
    """序号列表与图片列表不等长时不能错位，退回按位置对齐。"""
    text = "<!-- image --><!-- image -->"
    images = [_image(1, vision_caption="甲"), _image(2, vision_caption="乙")]
    result = annotate_image_placeholders(text, images, [3])  # 长度不匹配
    assert "［图 1 · 图片］甲" in result
    assert "［图 2 · 图片］乙" in result


def test_source_ordinal_none_entries_are_skipped() -> None:
    """register_page_scan 之类的条目来源序号为 None，不应被误配到占位符上。"""
    text = "<!-- image -->"
    images = [_image(1, vision_caption="整页扫描"), _image(2, vision_caption="内嵌图")]
    result = annotate_image_placeholders(text, images, [None, 1])
    assert "内嵌图" in result
    assert "整页扫描" not in result


def test_no_image_objects() -> None:
    result = annotate_image_placeholders("前<!-- image -->后", [])
    assert result == "前［图片］后"


def test_no_placeholder_is_noop() -> None:
    text = "纯正文，没有任何图片。"
    assert annotate_image_placeholders(text, [_image(1)]) == text

    # 大小写 / 空格变体也要能识别
    assert "<!-- image -->" not in annotate_image_placeholders("<!--IMAGE-->", [])
    assert "<!-- image -->" not in annotate_image_placeholders("<!--  image  -->", [])


def test_long_description_truncated() -> None:
    img = _image(1, vision_caption="很长的描述" * 200)
    result = annotate_image_placeholders("<!-- image -->", [img])
    label = "［图 1 · 图片］"
    assert result.startswith(label)
    assert len(result) - len(label) <= _ANNOTATION_MAX_CHARS + 1  # +1 = 省略号
    assert result.endswith("…")


def test_wide_short_content_image_is_not_filtered() -> None:
    """
    宽而矮的**内容**截图不能被当成图标丢掉.

    真实案例：``torch.Size([64, 10])`` 的一行终端输出截图只有 262×33，
    老规则"任一边 < 60 即丢弃"把它当成图标丢掉了 —— 这张图因此既没有被
    分析，也没能出现在正文标注里。判据改为"两边都小才算图标"。
    """
    from PIL import Image

    from app.services.parsers.image_recognition import _is_usable_image

    # 内容：一行终端输出（宽 262、高 33）
    assert _is_usable_image(Image.new("RGB", (262, 33), "black")) is True
    # 内容：宽而矮的代码截图
    assert _is_usable_image(Image.new("RGB", (600, 40), "white")) is True
    # 图标：两边都小
    assert _is_usable_image(Image.new("RGB", (32, 32), "white")) is False
    assert _is_usable_image(Image.new("RGB", (48, 40), "white")) is False
    # 退化条带：任一边过小（分隔线 / 边框）
    assert _is_usable_image(Image.new("RGB", (900, 3), "white")) is False
    assert _is_usable_image(Image.new("RGB", (4, 700), "white")) is False


def main() -> int:
    tests = [
        test_placeholders_replaced_in_order,
        test_description_priority_prefers_vision,
        test_description_falls_back_to_ocr_then_structured,
        test_blank_description_keeps_label_only,
        test_more_placeholders_than_images_degrades,
        test_source_ordinal_alignment_with_filtered_image,
        test_source_ordinals_length_mismatch_falls_back_to_position,
        test_source_ordinal_none_entries_are_skipped,
        test_no_image_objects,
        test_no_placeholder_is_noop,
        test_long_description_truncated,
        test_wide_short_content_image_is_not_filtered,
    ]
    for test in tests:
        test()
        print(f"  ok {test.__name__}")
    print(f"\nALL PASSED ({len(tests)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
