import asyncio
import io
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

from app.utils.file_utils import validate_document_upload


def validate(name, data):
    return asyncio.run(validate_document_upload(UploadFile(filename=name, file=io.BytesIO(data)), SimpleNamespace(MAX_UPLOAD_SIZE_MB=1)))


@pytest.mark.parametrize("extension", ["pdf", "docx", "pptx", "xlsx", "doc", "png", "jpg", "jpeg", "tiff", "bmp", "webp"])
def test_binary_extensions_reject_incorrect_signatures(extension):
    with pytest.raises(HTTPException) as error:
        validate("renamed." + extension, b"plain text, not the declared binary format")
    assert error.value.status_code == 415


@pytest.mark.parametrize("extension", ["txt", "md", "markdown", "csv", "json", "log"])
def test_text_rejects_nul_after_first_kilobyte(extension):
    with pytest.raises(HTTPException) as error:
        validate("late-nul." + extension, b"A" * 2048 + b"\x00")
    assert error.value.status_code == 415


def test_webp_checks_its_format_tag_and_plain_text_still_works():
    with pytest.raises(HTTPException):
        validate("wrong.webp", b"RIFF1234WAVE")
    text = "项目,年度,预算\n测试,2024,4321".encode()
    assert validate("actual.csv", text) == text
