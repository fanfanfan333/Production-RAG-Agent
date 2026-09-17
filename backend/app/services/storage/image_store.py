"""
Image persistence layer（部分2：图片必须保存下来）.

Every embedded image recognised during ingestion is written to disk so it can
later be:

    1. indexed as an independent retrieval object, and
    2. **returned as the original image** — the chat citation panel renders the
       real picture instead of a text description of it.

Layout（第三层：图片按租户隔离）:

    uploads/
    └── {tenant_id}/                 # 如 tenant_A / company_A
        └── {document_id}/
            ├── original.pdf
            └── images/
                ├── page_1_image_1.png
                └── page_3_image_1.png

升级前的旧布局是 ``uploads/{document_id}/``（无租户级目录）；读取侧
``resolve_image_path`` 对新旧布局都做了回退兼容，存量图片不需要搬迁。

The path stored in the Qdrant payload is *relative to the document directory*
(``images/page_3_image_1.png``) so the vector store never contains a machine
absolute path and stays portable across deployments.
"""

from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.services.tenancy import normalize_tenant_id
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Sub-directory that holds a document's extracted images.
IMAGES_SUBDIR = "images"


def storage_root() -> Path:
    """Absolute path of the image storage root (created on demand)."""
    root = Path(get_settings().IMAGE_STORAGE_DIR)
    if not root.is_absolute():
        # Resolve relative to the backend working directory (where uvicorn runs).
        root = Path.cwd() / root
    return root


def document_dir(document_id: str, tenant_id: str | None = None) -> Path:
    """
    Absolute directory that holds one document's assets.

    带 tenant_id → 新布局 ``uploads/{tenant_id}/{document_id}/``；
    不带 → 旧布局 ``uploads/{document_id}/``（兼容存量数据）。
    tenant_id 经过 normalize（白名单字符），不会成为目录穿越向量。
    """
    if tenant_id:
        return storage_root() / normalize_tenant_id(tenant_id) / str(document_id)
    return storage_root() / str(document_id)


def images_dir(document_id: str, tenant_id: str | None = None) -> Path:
    """Absolute directory that holds one document's extracted images."""
    return document_dir(document_id, tenant_id) / IMAGES_SUBDIR


def _candidate_dirs(document_id: str, tenant_id: str | None) -> list[Path]:
    """
    读取侧的候选目录：新布局优先，旧布局回退（存量免搬迁）.

    只知道 document_id 时（tenant_id=None），额外扫描一层 ``* /{document_id}``
    —— document_id 是不可猜测的 UUID，不会因此泄漏跨租户文件。
    """
    dirs: list[Path] = []
    if tenant_id:
        dirs.append(document_dir(document_id, tenant_id))
    dirs.append(document_dir(document_id))
    if not tenant_id:
        try:
            root = storage_root()
            for child in root.iterdir():
                if child.is_dir() and child.name != str(document_id):
                    cand = child / str(document_id)
                    if cand.is_dir():
                        dirs.append(cand)
        except OSError:
            pass
    return dirs


def image_relative_path(page_number: int, index: int, ext: str = "png") -> str:
    """
    Relative path (from the document directory) for one extracted image.

    ``page_3_image_1.png``  →  ``images/page_3_image_1.png``
    """
    ext = (ext or "png").lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    return f"{IMAGES_SUBDIR}/page_{int(page_number)}_image_{int(index)}.{ext}"


def save_image(
    document_id: str,
    page_number: int,
    index: int,
    data: bytes,
    ext: str = "png",
    tenant_id: str | None = None,
) -> str | None:
    """
    Persist one image and return its document-relative path.

    Returns None when saving fails — ingestion must never fail because an
    image could not be written (the textual description still gets indexed).
    """
    if not data:
        return None
    try:
        target_dir = images_dir(document_id, tenant_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        rel = image_relative_path(page_number, index, ext)
        (document_dir(document_id, tenant_id) / rel).write_bytes(data)
        return rel
    except Exception as exc:      # noqa: BLE001 — never break ingestion on I/O
        logger.warning(
            "Failed to save image (doc=%s page=%d idx=%d): %s",
            document_id, page_number, index, exc,
        )
        return None


def resolve_image_path(
    document_id: str,
    relative_path: str,
    tenant_id: str | None = None,
) -> Path | None:
    """
    Resolve a stored relative path to an absolute path, or None if it escapes
    the document's own directory (defence against path traversal).

    依次尝试新布局（{tenant}/{doc}）与旧布局（{doc}）——存量图片免搬迁。
    """
    if not relative_path:
        return None
    for base_dir in _candidate_dirs(document_id, tenant_id):
        base = base_dir.resolve()
        try:
            candidate = (base / relative_path).resolve()
        except Exception:
            continue
        # The resolved path must stay inside the document directory.
        if base != candidate and base not in candidate.parents:
            logger.warning(
                "Rejected image path escaping document dir (doc=%s path=%r)",
                document_id, relative_path,
            )
            continue
        if candidate.is_file():
            return candidate
    return None


def save_original(
    document_id: str,
    filename: str,
    data: bytes,
    tenant_id: str | None = None,
) -> str | None:
    """
    Archive the uploaded original alongside its extracted images
    (``uploads/{tenant_id}/{document_id}/original.pdf`` style, keeping the
    real suffix).
    """
    if not data:
        return None
    try:
        suffix = Path(filename).suffix or ""
        target = document_dir(document_id, tenant_id) / f"original{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target.name
    except Exception as exc:      # noqa: BLE001
        logger.warning("Failed to archive original (doc=%s): %s", document_id, exc)
        return None


def delete_document_images(document_id: str, tenant_id: str | None = None) -> None:
    """Remove every stored asset for a document (called on delete)."""
    import shutil

    for target in _candidate_dirs(document_id, tenant_id):
        try:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
                logger.info("Removed stored assets for document_id=%s (%s)", document_id, target)
        except Exception as exc:      # noqa: BLE001
            logger.warning("Failed to remove assets for document_id=%s: %s", document_id, exc)
