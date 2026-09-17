"""
File validation utilities for upload endpoints.
"""

from fastapi import HTTPException, UploadFile, status

from app.config import get_settings

_ALLOWED_EXTENSIONS = {
    "pdf", "docx", "doc", "pptx", "xlsx", "csv", "txt", "md", "markdown",
    "png", "jpg", "jpeg", "tiff", "bmp", "webp", "json", "log"
}

# 旧版 Word 的 OLE2 复合文档魔数（D0 CF 11 E0 A1 B1 1A E1）。
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# 扩展名不被支持时，告诉用户**下一步做什么**，而不是只回一句
# "unsupported extension"。这些是格式确定不支持、但有等价可上传格式的。
_UNSUPPORTED_EXT_HINTS = {
    "rtf": "请用 Word 打开后另存为 .docx 重新上传",
    "wps": "WPS 私有格式请另存为 .docx / .xlsx / .pptx 后重新上传",
    "wpt": "WPS 私有格式请另存为 .docx 后重新上传",
    "xls": "旧版 Excel 请另存为 .xlsx 后重新上传",
    "ppt": "旧版 PowerPoint 请另存为 .pptx 后重新上传",
    "html": "请另存为 .md 或 .txt 后重新上传",
    "htm": "请另存为 .md 或 .txt 后重新上传",
}

# 魔数与扩展名不符时（多半是文件被改名或已损坏）的格式专属提示。
_MAGIC_MISMATCH_HINTS = {
    "doc": "文件内容不是有效的旧版 Word 文档（可能已损坏或被改过扩展名）",
    "docx": "文件内容不是有效的 .docx（可能已损坏或被改过扩展名）",
    "pdf": "文件内容不是有效的 PDF（可能已损坏或被改过扩展名）",
    "pptx": "文件内容不是有效的 .pptx（可能已损坏或被改过扩展名）",
    "xlsx": "文件内容不是有效的 .xlsx（可能已损坏或被改过扩展名）",
}

async def validate_document_upload(file: UploadFile, settings=None) -> bytes:
    """
    Read the uploaded file into memory and validate:
      1. File size does not exceed MAX_UPLOAD_SIZE_MB
      2. File format is supported
    
    Returns the raw bytes so the caller doesn't re-read from disk.
    Raises HTTPException on any violation.
    """
    if settings is None:
        settings = get_settings()

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # Read entire file into memory (bounded by max_bytes + 1 so we can detect oversize)
    content = await file.read(max_bytes + 1)

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File '{file.filename}' exceeds the maximum allowed size "
                f"of {settings.MAX_UPLOAD_SIZE_MB} MB."
            ),
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{file.filename}' is empty.",
        )

    filename = file.filename or ""
    ext = filename.lower().split('.')[-1] if '.' in filename else ""
    
    if ext not in _ALLOWED_EXTENSIONS:
        hint = _UNSUPPORTED_EXT_HINTS.get(ext)
        detail = f"不支持的文件格式 '.{ext}'。" if ext else "文件缺少扩展名。"
        if hint:
            detail += hint
        else:
            detail += f"支持的格式：{', '.join(sorted(_ALLOWED_EXTENSIONS))}。"
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=detail,
        )

    # 3. Magic bytes verification (prevent MIME spoofing)
    header = content[:8]
    is_valid_magic = False
    
    if ext == "pdf" and header.startswith(b"%PDF-"):
        is_valid_magic = True
    elif ext in {"docx", "pptx", "xlsx"} and header.startswith(b"PK\x03\x04"):
        is_valid_magic = True
    elif ext == "doc":
        # .doc 的唯一合法容器是 OLE2。必须在这里给出结论，不能落到末尾
        # "其它格式默认放行"的分支 —— 否则任意文件改个 .doc 后缀就能送进
        # 解析器（测试 test_upload_rejects_doc_with_wrong_magic_* 覆盖此点）。
        is_valid_magic = header.startswith(_OLE2_MAGIC)
    elif ext == "png" and header.startswith(b"\x89PNG\r\n\x1a\n"):
        is_valid_magic = True
    elif ext in {"jpg", "jpeg"} and header.startswith(b"\xff\xd8\xff"):
        is_valid_magic = True
    elif ext in {"txt", "md", "markdown", "csv", "json", "log"}:
        # Text files don't have standard magic bytes; verify they contain valid text
        # Attempt decoding with UTF-8, Windows-1252, and Latin-1 fallbacks, ensuring no null bytes
        is_valid_magic = False
        for encoding in ["utf-8", "windows-1252", "latin-1"]:
            try:
                decoded = content[:1024].decode(encoding)
                if "\x00" not in decoded:
                    is_valid_magic = True
                    break
            except UnicodeDecodeError:
                continue
    else:
        # Fallback for other image formats (tiff, bmp, webp) if not strictly checked
        is_valid_magic = True
        
    if not is_valid_magic:
        detail = _MAGIC_MISMATCH_HINTS.get(
            ext, f"文件内容与扩展名 '.{ext}' 不符，文件可能已损坏"
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{detail}。",
        )

    return content
