"""诊断：实战.docx 的图片块现状 + 深色流程图的原图分析（容器内运行）."""

import asyncio
import sys


async def main(img_path: str) -> None:
    from app.config import get_settings
    from app.db.qdrant import get_qdrant_client

    client = get_qdrant_client()
    coll = get_settings().QDRANT_COLLECTION

    # ── 1. 库里该文档的图片块现在长什么样 ────────────────────────────────────
    from qdrant_client.http import models as qmodels

    pts, _ = await client.scroll(
        collection_name=coll,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchValue(
                        value="6121edf2-8f8f-4e49-8f6c-07c504f118ed"
                    ),
                ),
                qmodels.FieldCondition(
                    key="content_type", match=qmodels.MatchValue(value="image")
                ),
            ]
        ),
        limit=20,
        with_payload=True,
    )
    print(f"=== 库中图片块: {len(pts)} 个 ===")
    for p in pts:
        pl = p.payload or {}
        print({
            "image_id": pl.get("image_id"),
            "image_type": pl.get("image_type"),
            "engine": pl.get("analyze_engine"),
            "path": pl.get("image_path"),
            "text": (pl.get("text") or "")[:120],
        })

    # ── 2. 用当前管线原图直测 ────────────────────────────────────────────────
    from PIL import Image

    from app.services.image_understanding import understand_image
    from app.services.image_understanding.imaging import encode_png_safe

    img = Image.open(img_path).convert("RGB")
    print(f"\n=== 原图直测: {img.size} ===")
    result = understand_image(
        img, page_number=1, filename="flow.png", png_bytes=encode_png_safe(img)
    )
    print("type:", result.image_type)
    print("analyze_engine:", result.analyze_engine)
    print("decision:", result.decision)
    print("attempts:", result.attempts)
    print("confidence:", result.confidence)
    print("manual_review:", result.manual_review)
    print("ocr_text:", repr((result.ocr_text or "")[:300]))
    print("vision:", repr((result.vision_caption or "")[:300]))
    sc = getattr(result, "structured_content", None)
    print("structured:", repr((sc or "")[:200]))


asyncio.run(main(sys.argv[1]))
