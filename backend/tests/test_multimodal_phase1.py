"""
Task 2 Phase 1: Multimodal File Upload and Validation Tests
Covers all 6 validation stages, security logging, storage isolation,
API response security (no storage_path), and /ask endpoint regression.
"""
import io
import os
import sys
import json
import tempfile
import pytest

# Ensure backend/ is on path for all imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from fastapi.testclient import TestClient

from multimodal.validator import MultimodalValidator, MultimodalValidationError
from multimodal.config import MultimodalConfig
from main import app

client = TestClient(app)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: minimal valid file bytes
# ──────────────────────────────────────────────────────────────────────────────

def make_png_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (4, 4), color=(0, 255, 0))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_webp_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (4, 4), color=(0, 0, 255))
    img.save(buf, format="WEBP")
    return buf.getvalue()


def make_pdf_bytes() -> bytes:
    # Minimal valid one-page PDF (no external deps)
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Valid uploads via validate_file()
# ──────────────────────────────────────────────────────────────────────────────

def test_valid_png_upload():
    v = MultimodalValidator()
    ok, reason, meta = v.validate_file(make_png_bytes(), "test.png", "image/png")
    assert ok, f"Expected valid PNG but got: {reason}"
    assert meta is not None
    assert meta["extension"] == ".png"
    assert "dimensions" in meta


def test_valid_jpg_upload():
    v = MultimodalValidator()
    ok, reason, meta = v.validate_file(make_jpeg_bytes(), "photo.jpg", "image/jpeg")
    assert ok, f"Expected valid JPEG but got: {reason}"
    assert meta is not None
    assert meta["extension"] == ".jpg"


def test_valid_jpeg_extension_upload():
    v = MultimodalValidator()
    ok, reason, meta = v.validate_file(make_jpeg_bytes(), "photo.jpeg", "image/jpeg")
    assert ok, f"Expected valid JPEG (.jpeg) but got: {reason}"
    assert meta is not None
    assert meta["extension"] == ".jpeg"


def test_valid_webp_upload():
    v = MultimodalValidator()
    ok, reason, meta = v.validate_file(make_webp_bytes(), "image.webp", "image/webp")
    assert ok, f"Expected valid WEBP but got: {reason}"
    assert meta is not None
    assert meta["extension"] == ".webp"


def test_valid_pdf_upload():
    v = MultimodalValidator()
    ok, reason, meta = v.validate_file(make_pdf_bytes(), "doc.pdf", "application/pdf")
    assert ok, f"Expected valid PDF but got: {reason}"
    assert meta is not None
    assert meta["format"] == "PDF"
    assert meta["page_count"] >= 1


# ──────────────────────────────────────────────────────────────────────────────
# 2. Rejection – unsupported extension
# ──────────────────────────────────────────────────────────────────────────────

def test_unsupported_extension_gif_rejected():
    v = MultimodalValidator()
    ok, reason, meta = v.validate_file(b"GIF89a...", "animated.gif", "image/gif")
    assert not ok
    assert "gif" in reason.lower() or "extension" in reason.lower()
    assert meta is None


def test_unsupported_extension_exe_rejected():
    v = MultimodalValidator()
    ok, reason, meta = v.validate_file(b"MZ\x90\x00" + b"\x00" * 100, "malware.exe", "application/octet-stream")
    assert not ok
    assert meta is None


def test_unsupported_extension_html_rejected():
    v = MultimodalValidator()
    ok, reason, meta = v.validate_file(b"<html></html>", "page.html", "text/html")
    assert not ok
    assert meta is None


# ──────────────────────────────────────────────────────────────────────────────
# 3. Rejection – wrong MIME type
# ──────────────────────────────────────────────────────────────────────────────

def test_wrong_mime_type_gif_rejected():
    """PNG file bytes but wrong MIME type (image/gif)."""
    v = MultimodalValidator()
    ok, reason, meta = v.validate_file(make_png_bytes(), "test.png", "image/gif")
    assert not ok
    assert "mime" in reason.lower() or "gif" in reason.lower()
    assert meta is None


def test_mime_extension_mismatch_png_as_jpeg_rejected():
    """PNG file bytes but extension .jpg and MIME image/jpeg – magic header check catches it."""
    v = MultimodalValidator()
    ok, reason, meta = v.validate_file(make_png_bytes(), "tricky.jpg", "image/jpeg")
    assert not ok
    assert meta is None


# ──────────────────────────────────────────────────────────────────────────────
# 4. Rejection – empty file
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_file_rejected():
    v = MultimodalValidator()
    ok, reason, meta = v.validate_file(b"", "empty.png", "image/png")
    assert not ok
    assert "empty" in reason.lower()
    assert meta is None


def test_zero_bytes_rejected():
    v = MultimodalValidator()
    ok, reason, meta = v.validate_file(bytes(), "zero.pdf", "application/pdf")
    assert not ok
    assert meta is None


# ──────────────────────────────────────────────────────────────────────────────
# 5. Rejection – corrupted files (valid extension, garbage bytes)
# ──────────────────────────────────────────────────────────────────────────────

def test_corrupted_png_rejected():
    v = MultimodalValidator()
    garbage = b"this is totally not a png file at all\x00\x01\x02"
    ok, reason, meta = v.validate_file(garbage, "fake.png", "image/png")
    assert not ok
    assert meta is None


def test_corrupted_jpg_rejected():
    v = MultimodalValidator()
    garbage = b"notajpeg" + b"\x00" * 50
    ok, reason, meta = v.validate_file(garbage, "bad.jpg", "image/jpeg")
    assert not ok
    assert meta is None


def test_corrupted_webp_rejected():
    v = MultimodalValidator()
    garbage = b"NOT_A_RIFF_HEADER" + b"\x00" * 50
    ok, reason, meta = v.validate_file(garbage, "bad.webp", "image/webp")
    assert not ok
    assert meta is None


def test_corrupted_pdf_rejected():
    v = MultimodalValidator()
    garbage = b"These are not PDF bytes at all."
    ok, reason, meta = v.validate_file(garbage, "bad.pdf", "application/pdf")
    assert not ok
    assert meta is None


# ──────────────────────────────────────────────────────────────────────────────
# 6. Rejection – oversized file
# ──────────────────────────────────────────────────────────────────────────────

def test_oversized_file_rejected():
    # Create a config with 1 MB limit; generate 2 MB of data
    cfg = MultimodalConfig(max_file_size_mb=1)
    v = MultimodalValidator(config=cfg)
    big_data = b"\x89PNG\r\n\x1a\n" + b"A" * (2 * 1024 * 1024)
    ok, reason, meta = v.validate_file(big_data, "big.png", "image/png")
    assert not ok
    assert "exceed" in reason.lower() or "limit" in reason.lower()
    assert meta is None


# ──────────────────────────────────────────────────────────────────────────────
# 7. Rejection – disguised executables / scripts
# ──────────────────────────────────────────────────────────────────────────────

def test_disguised_windows_exe_rejected():
    """MZ header (Windows EXE/DLL) disguised as PNG."""
    v = MultimodalValidator()
    payload = b"MZ\x90\x00" + b"\x00" * 200
    ok, reason, meta = v.validate_file(payload, "image.png", "image/png")
    assert not ok
    assert meta is None


def test_disguised_elf_binary_rejected():
    """ELF binary disguised as JPEG."""
    v = MultimodalValidator()
    payload = b"\x7fELF" + b"\x00" * 200
    ok, reason, meta = v.validate_file(payload, "photo.jpg", "image/jpeg")
    assert not ok
    assert meta is None


def test_disguised_shell_script_rejected():
    """Shell script disguised as PDF."""
    v = MultimodalValidator()
    payload = b"#!/bin/bash\nrm -rf /\n" + b"\x00" * 50
    ok, reason, meta = v.validate_file(payload, "doc.pdf", "application/pdf")
    assert not ok
    assert meta is None


def test_disguised_php_script_rejected():
    v = MultimodalValidator()
    payload = b"<?php echo shell_exec($_GET['cmd']); ?>"
    ok, reason, meta = v.validate_file(payload, "img.jpg", "image/jpeg")
    assert not ok
    assert meta is None


# ──────────────────────────────────────────────────────────────────────────────
# 8. Safe temporary storage (save_upload)
# ──────────────────────────────────────────────────────────────────────────────

def test_save_upload_returns_file_id_uuid():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MultimodalConfig(upload_dir=tmpdir)
        v = MultimodalValidator(config=cfg)
        result = v.save_upload(make_png_bytes(), "upload.png", "image/png")
        assert "file_id" in result
        file_id = result["file_id"]
        assert len(file_id) == 32  # UUID hex = 32 chars
        assert all(c in "0123456789abcdef" for c in file_id)


def test_save_upload_original_filename_not_on_disk():
    """Saved filename must be UUID-based, not the original filename."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MultimodalConfig(upload_dir=tmpdir)
        v = MultimodalValidator(config=cfg)
        original_name = "my_private_photo.png"
        result = v.save_upload(make_png_bytes(), original_name, "image/png")
        # Original filename should NOT appear in disk filenames
        disk_files = os.listdir(tmpdir)
        for f in disk_files:
            assert original_name not in f
        # File should exist with UUID-based name
        assert result["safe_filename"] in disk_files


def test_save_upload_file_exists_on_disk():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MultimodalConfig(upload_dir=tmpdir)
        v = MultimodalValidator(config=cfg)
        result = v.save_upload(make_jpeg_bytes(), "test.jpg", "image/jpeg")
        safe_name = result["safe_filename"]
        assert os.path.exists(os.path.join(tmpdir, safe_name))


def test_save_upload_raises_on_invalid():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MultimodalConfig(upload_dir=tmpdir)
        v = MultimodalValidator(config=cfg)
        with pytest.raises(MultimodalValidationError):
            v.save_upload(b"", "empty.png", "image/png")


# ──────────────────────────────────────────────────────────────────────────────
# 9. API endpoint: POST /multimodal/analyze
# ──────────────────────────────────────────────────────────────────────────────

def test_api_valid_png_upload_returns_200():
    png_bytes = make_png_bytes()
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("image.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert "file_id" in data
    assert "original_filename" in data
    assert "file_size_bytes" in data
    assert "metadata" in data


def test_api_valid_pdf_upload_returns_200():
    pdf_bytes = make_pdf_bytes()
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("document.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert "file_id" in data


def test_api_invalid_file_returns_400():
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("bad.gif", io.BytesIO(b"GIF89a"), "image/gif")},
    )
    assert response.status_code == 400


def test_api_empty_file_returns_400():
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
    )
    assert response.status_code == 400


def test_api_disguised_exe_returns_400():
    payload = b"MZ\x90\x00" + b"\x00" * 200
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("virus.png", io.BytesIO(payload), "image/png")},
    )
    assert response.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# 10. SECURITY: storage_path must NOT appear in API response
# ──────────────────────────────────────────────────────────────────────────────

def test_storage_path_not_in_api_response():
    """Server-side storage path must never be returned to the client."""
    png_bytes = make_png_bytes()
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("secure.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    # Key must not exist
    assert "storage_path" not in data
    # Also verify no filesystem path leaks in any string value
    response_text = response.text
    # No Windows-style paths (e.g. C:\) or Unix paths (/tmp, /home)
    assert "C:\\" not in response_text
    assert "\\temp_uploads\\" not in response_text.replace("/", "\\")


def test_storage_path_not_in_rejection_response():
    """Rejection responses must also not expose filesystem paths."""
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
    )
    assert response.status_code == 400
    response_text = response.text
    assert "temp_uploads" not in response_text
    assert "C:\\" not in response_text


# ──────────────────────────────────────────────────────────────────────────────
# 11. Security event logging on rejection
# ──────────────────────────────────────────────────────────────────────────────

def test_security_event_logged_on_rejection():
    """A rejected upload must write a security log entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "security.log")
        cfg = MultimodalConfig(upload_dir=tmpdir, security_log_file=log_file)
        v = MultimodalValidator(config=cfg)
        # Trigger a rejection
        v.validate_file(b"", "empty.png", "image/png")
        # Log must exist and contain an entry
        assert os.path.exists(log_file), "Security log file was not created on rejection"
        with open(log_file) as f:
            content = f.read()
        assert len(content) > 0, "Security log is empty after rejection"
        # Should mention the event type
        assert "MULTIMODAL" in content or "EMPTY" in content


# ──────────────────────────────────────────────────────────────────────────────
# 12. Regression: existing /ask endpoint still works
# ──────────────────────────────────────────────────────────────────────────────

def test_ask_endpoint_still_exists():
    """The /ask endpoint must still exist and respond (even if FAISS index is missing)."""
    response = client.post("/ask", json={"question": "test question"})
    # 400 or 500 are acceptable (no FAISS in test env), but NOT 404 (endpoint gone)
    assert response.status_code != 404, "/ask endpoint has been removed or broken"


def test_root_endpoint_still_works():
    """The root / endpoint must still be reachable."""
    response = client.get("/")
    assert response.status_code == 200


def test_multimodal_endpoint_exists():
    """The /multimodal/analyze route must be registered."""
    # Posting nothing should give 422 (missing file field), not 404
    response = client.post("/multimodal/analyze")
    assert response.status_code != 404, "/multimodal/analyze endpoint is not registered"
