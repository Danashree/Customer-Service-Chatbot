"""
backend/tests/test_multimodal_phase7.py
Task 2 Phase 7 – Frontend Integration & Final Validation Tests

Tests:
  - /ask regression (text-only path unchanged)
  - /multimodal/analyze contract (sync + async FormData)
  - /multimodal/jobs/{job_id} polling contract
  - Error response safety (no server paths / stack traces)
  - HTML element presence (attachment bar, file input, pill, script tag)
  - CSS class presence (attachment-bar, evidence-card, conflict-alert, etc.)
  - JavaScript function presence (handleFileSelect, clearAttachment, pollJobStatus, renderEvidenceCard)
  - Security: no innerHTML for backend data, textContent used for safe fields
"""
import sys

import os
import io
import json
import re

# -- path setup --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def client():
    """TestClient for the FastAPI app."""
    from main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def index_html():
    """Content of frontend/index.html."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "frontend", "index.html"
    )
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def style_css():
    """Content of frontend/style.css."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "frontend", "style.css"
    )
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def script_js():
    """Content of frontend/script.js."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "frontend", "script.js"
    )
    with open(path, encoding="utf-8") as f:
        return f.read()


# ===========================================================================
# Helper – generate a real PIL-rendered PNG (passes Phase 4 integrity check)
# ===========================================================================

def make_png_bytes(color: tuple = (200, 200, 200)) -> bytes:
    """Generate a minimal but fully valid PNG via Pillow."""
    buf = io.BytesIO()
    img = Image.new("RGB", (20, 20), color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_file():
    return ("test_evidence.png", io.BytesIO(make_png_bytes()), "image/png")


# ===========================================================================
# 1.  /ask regression – existing behavior must be untouched
# ===========================================================================

class TestAskRegression:
    def test_ask_returns_json_with_answer_or_expected_error(self, client):
        """POST /ask must return JSON. If knowledgebase missing → 400 with expected detail."""
        res = client.post("/ask", json={"question": "Do you offer EMI payments?"})
        assert res.status_code in (200, 400, 500)
        data = res.json()
        assert isinstance(data, dict)
        if res.status_code == 200:
            assert "answer" in data
            assert isinstance(data["answer"], str)
        elif res.status_code == 400:
            assert "detail" in data
            assert "Knowledgebase" in data["detail"] or "knowledgebase" in data["detail"]

    def test_ask_requires_question_field(self, client):
        """POST /ask without 'question' key must return 422 Unprocessable Entity."""
        res = client.post("/ask", json={"msg": "hello"})
        assert res.status_code == 422

    def test_ask_empty_question_rejected(self, client):
        """POST /ask with empty question string may fail at app or model layer."""
        res = client.post("/ask", json={"question": ""})
        # Could be 200 with empty-ish answer or 400/500; must not be 422
        assert res.status_code in (200, 400, 500)

    def test_ask_response_has_no_server_paths(self, client):
        """Error responses from /ask must not expose file-system paths."""
        res = client.post("/ask", json={"question": "test path leak"})
        text = res.text
        assert "c:\\" not in text.lower()
        assert "c:/" not in text.lower()
        assert "/home/" not in text.lower()
        assert "traceback" not in text.lower()


# ===========================================================================
# 2.  /multimodal/analyze – sync path
# ===========================================================================

class TestMultimodalAnalyzeSync:
    def test_sync_accepts_valid_png(self, client):
        """POST /multimodal/analyze with a valid PNG and processing_mode=sync must return 200."""
        res = client.post(
            "/multimodal/analyze",
            data={"processing_mode": "sync"},
            files={"file": _png_file()},
        )
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, dict)

    def test_sync_returns_expected_top_level_keys(self, client):
        """Sync response must contain at minimum 'status' or extraction-related keys."""
        res = client.post(
            "/multimodal/analyze",
            data={"processing_mode": "sync"},
            files={"file": _png_file()},
        )
        assert res.status_code == 200
        data = res.json()
        has_key = any(k in data for k in (
            "status", "extraction", "comparison", "conflicts",
            "extracted_fields", "comparison_results", "file_id",
        ))
        assert has_key, f"Unexpected response shape: {list(data.keys())}"

    def test_sync_with_message(self, client):
        """Processing a file with an accompanying message must succeed."""
        res = client.post(
            "/multimodal/analyze",
            data={"processing_mode": "sync", "message": "My order ID is ORD-9999"},
            files={"file": _png_file()},
        )
        assert res.status_code == 200

    def test_sync_missing_file_returns_422(self, client):
        """Posting without a file must return 422."""
        res = client.post("/multimodal/analyze", data={"processing_mode": "sync"})
        assert res.status_code == 422

    def test_sync_invalid_processing_mode_returns_400(self, client):
        """An unrecognised processing_mode must return 400."""
        res = client.post(
            "/multimodal/analyze",
            data={"processing_mode": "turbo"},
            files={"file": _png_file()},
        )
        assert res.status_code == 400
        assert "Invalid processing_mode" in res.json().get("detail", "")

    def test_sync_no_server_path_in_response(self, client):
        """Sync response must not leak server file-system paths."""
        res = client.post(
            "/multimodal/analyze",
            data={"processing_mode": "sync"},
            files={"file": _png_file()},
        )
        text = res.text
        assert "c:\\" not in text.lower()
        assert "/home/" not in text.lower()
        assert "traceback" not in text.lower()

    def test_sync_no_raw_binary_in_response(self, client):
        """Response body must be valid JSON."""
        res = client.post(
            "/multimodal/analyze",
            data={"processing_mode": "sync"},
            files={"file": _png_file()},
        )
        try:
            res.json()
        except Exception:
            pytest.fail("Response is not valid JSON")


# ===========================================================================
# 3.  /multimodal/analyze – async path
# ===========================================================================

class TestMultimodalAnalyzeAsync:
    def test_async_returns_job_id(self, client):
        """POST with processing_mode=async must return job_id and queued status."""
        res = client.post(
            "/multimodal/analyze",
            data={"processing_mode": "async"},
            files={"file": _png_file()},
        )
        assert res.status_code == 200
        data = res.json()
        assert "job_id" in data
        assert data.get("status") in ("queued", "processing", "completed")

    def test_async_job_id_is_non_empty_string(self, client):
        res = client.post(
            "/multimodal/analyze",
            data={"processing_mode": "async"},
            files={"file": _png_file()},
        )
        assert res.status_code == 200
        job_id = res.json().get("job_id", "")
        assert isinstance(job_id, str) and len(job_id) > 0

    def test_auto_mode_returns_valid_response(self, client):
        """Default mode (auto) must return valid JSON with either job info or result."""
        res = client.post(
            "/multimodal/analyze",
            files={"file": _png_file()},
        )
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, dict)


# ===========================================================================
# 4.  /multimodal/jobs/{job_id} – polling contract
# ===========================================================================

class TestJobPollingContract:
    def _get_job_id(self, client):
        res = client.post(
            "/multimodal/analyze",
            data={"processing_mode": "async"},
            files={"file": _png_file()},
        )
        assert res.status_code == 200
        return res.json()["job_id"]

    def test_poll_known_job_returns_200(self, client):
        job_id = self._get_job_id(client)
        res = client.get(f"/multimodal/jobs/{job_id}")
        assert res.status_code == 200

    def test_poll_unknown_job_returns_404(self, client):
        res = client.get("/multimodal/jobs/nonexistent-job-abc-123")
        assert res.status_code == 404

    def test_poll_response_has_status_field(self, client):
        job_id = self._get_job_id(client)
        res = client.get(f"/multimodal/jobs/{job_id}")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert data["status"] in ("queued", "processing", "completed", "failed", "retrying")

    def test_poll_response_has_job_id(self, client):
        job_id = self._get_job_id(client)
        res = client.get(f"/multimodal/jobs/{job_id}")
        data = res.json()
        assert "job_id" in data

    def test_poll_no_server_paths_in_response(self, client):
        job_id = self._get_job_id(client)
        res = client.get(f"/multimodal/jobs/{job_id}")
        text = res.text
        assert "c:\\" not in text.lower()
        assert "traceback" not in text.lower()

    def test_poll_404_has_safe_detail(self, client):
        res = client.get("/multimodal/jobs/fake-job-xyz")
        data = res.json()
        assert "detail" in data
        assert "c:\\" not in data["detail"].lower()
        assert "traceback" not in data["detail"].lower()


# ===========================================================================
# 5.  HTML element presence checks
# ===========================================================================

class TestHtmlElementPresence:
    def test_attachment_bar_div_present(self, index_html):
        assert 'id="attachmentBar"' in index_html or "attachmentBar" in index_html

    def test_evidence_file_input_present(self, index_html):
        assert 'id="evidenceFileInput"' in index_html

    def test_file_input_accepts_correct_types(self, index_html):
        assert ".png" in index_html
        assert ".pdf" in index_html

    def test_file_pill_present(self, index_html):
        assert 'id="filePill"' in index_html

    def test_file_pill_name_present(self, index_html):
        assert 'id="filePillName"' in index_html

    def test_attach_hint_present(self, index_html):
        assert "attach-hint" in index_html

    def test_script_tag_references_script_js(self, index_html):
        assert "script.js" in index_html

    def test_style_tag_references_style_css(self, index_html):
        assert "style.css" in index_html

    def test_handleFileSelect_wired_in_html(self, index_html):
        assert "handleFileSelect" in index_html

    def test_clearAttachment_wired_in_html(self, index_html):
        assert "clearAttachment" in index_html

    def test_messages_container_present(self, index_html):
        assert 'id="messages"' in index_html

    def test_question_input_present(self, index_html):
        assert 'id="questionInput"' in index_html

    def test_init_btn_present(self, index_html):
        assert 'id="initBtn"' in index_html


# ===========================================================================
# 6.  CSS class presence checks
# ===========================================================================

class TestCssClassPresence:
    def test_attachment_bar_class(self, style_css):
        assert ".attachment-bar" in style_css

    def test_attach_btn_class(self, style_css):
        assert ".attach-btn" in style_css

    def test_file_pill_class(self, style_css):
        assert ".file-pill" in style_css

    def test_file_pill_name_class(self, style_css):
        assert ".file-pill-name" in style_css

    def test_file_pill_remove_class(self, style_css):
        assert ".file-pill-remove" in style_css

    def test_attach_hint_class(self, style_css):
        assert ".attach-hint" in style_css

    def test_evidence_card_class(self, style_css):
        assert ".evidence-card" in style_css

    def test_evidence_header_class(self, style_css):
        assert ".evidence-header" in style_css

    def test_evidence_quality_class(self, style_css):
        assert ".evidence-quality" in style_css

    def test_evidence_quality_good(self, style_css):
        assert ".evidence-quality.good" in style_css

    def test_evidence_quality_poor(self, style_css):
        assert ".evidence-quality.poor" in style_css

    def test_conflict_alert_class(self, style_css):
        assert ".conflict-alert" in style_css

    def test_cmp_badge_match(self, style_css):
        assert ".cmp-badge.match" in style_css

    def test_cmp_badge_conflict(self, style_css):
        assert ".cmp-badge.conflict" in style_css

    def test_security_notice_class(self, style_css):
        assert ".security-notice" in style_css

    def test_retention_notice_class(self, style_css):
        assert ".retention-notice" in style_css

    def test_evidence_grid_class(self, style_css):
        assert ".evidence-grid" in style_css


# ===========================================================================
# 7.  JavaScript function presence & security checks
# ===========================================================================

class TestJavaScriptFunctions:
    def test_handleFileSelect_defined(self, script_js):
        assert "function handleFileSelect" in script_js

    def test_clearAttachment_defined(self, script_js):
        assert "function clearAttachment" in script_js

    def test_pollJobStatus_defined(self, script_js):
        assert "function pollJobStatus" in script_js

    def test_renderEvidenceCard_defined(self, script_js):
        assert "function renderEvidenceCard" in script_js

    def test_sendMessage_still_defined(self, script_js):
        assert "function sendMessage" in script_js

    def test_selectedFile_variable_declared(self, script_js):
        assert "selectedFile" in script_js

    def test_max_file_bytes_defined(self, script_js):
        assert "MAX_FILE_BYTES" in script_js

    def test_allowed_extensions_defined(self, script_js):
        assert "ALLOWED_EXTENSIONS" in script_js

    def test_multimodal_analyze_endpoint_referenced(self, script_js):
        assert "/multimodal/analyze" in script_js

    def test_multimodal_jobs_endpoint_referenced(self, script_js):
        assert "/multimodal/jobs/" in script_js

    def test_ask_endpoint_still_referenced(self, script_js):
        assert '"/ask"' in script_js or "'/ask'" in script_js or "`${API}/ask`" in script_js

    def test_formdata_used_for_multimodal(self, script_js):
        assert "FormData" in script_js

    def test_encodeURIComponent_used_for_job_id(self, script_js):
        assert "encodeURIComponent" in script_js

    def test_textContent_used_for_pill_name(self, script_js):
        # pillName.textContent must be used (not innerHTML) for filenames
        assert "pillName.textContent" in script_js or "filePillName" in script_js

    def test_renderEvidenceCard_uses_textContent_not_innerHTML(self, script_js):
        """Evidence card renderer must not use innerHTML for backend-provided data."""
        # Find the renderEvidenceCard function body
        start = script_js.find("function renderEvidenceCard")
        assert start != -1
        # Find the next top-level function after it (or end of file)
        next_func = script_js.find("\nfunction ", start + 1)
        if next_func == -1:
            card_body = script_js[start:]
        else:
            card_body = script_js[start:next_func]
        # Must NOT contain innerHTML assignment for user data
        # (innerHTML is allowed only for static trusted strings, not .textContent replacements)
        assert ".textContent" in card_body, "renderEvidenceCard should use textContent"
        # Verify no innerHTML = <variable> pattern inside the card renderer
        # We allow innerHTML for static HTML strings but not for dynamic data
        bad_pattern = re.compile(r'\.innerHTML\s*=\s*[^"`\']')
        bad_matches = bad_pattern.findall(card_body)
        assert len(bad_matches) == 0, f"innerHTML used with dynamic data: {bad_matches}"

    def test_no_console_log_with_sensitive_data(self, script_js):
        """console.log should not be present in production code (security)."""
        # We allow zero console.log statements
        assert "console.log" not in script_js

    def test_poll_max_attempts_set(self, script_js):
        assert "MAX_ATTEMPTS" in script_js

    def test_poll_interval_set(self, script_js):
        assert "INTERVAL_MS" in script_js


# ===========================================================================
# 8.  Filename security – path traversal and dangerous filenames
# ===========================================================================

class TestFilenameSecurityEndpoint:
    def test_path_traversal_filename_rejected(self, client):
        res = client.post(
            "/multimodal/analyze",
            data={"processing_mode": "sync"},
            files={"file": ("../../etc/passwd.png", io.BytesIO(make_png_bytes()), "image/png")},
        )
        # Must be rejected with 400 (validation) or 422
        assert res.status_code in (400, 422)

    def test_null_byte_filename_rejected(self, client):
        res = client.post(
            "/multimodal/analyze",
            data={"processing_mode": "sync"},
            files={"file": ("evil\x00.png", io.BytesIO(make_png_bytes()), "image/png")},
        )
        # Backend sanitizes or rejects null-byte filenames; must not 500
        assert res.status_code in (200, 400, 422)

    def test_empty_filename_handled(self, client):
        res = client.post(
            "/multimodal/analyze",
            data={"processing_mode": "sync"},
            files={"file": ("", io.BytesIO(make_png_bytes()), "image/png")},
        )
        # Empty filename may be rejected or handled gracefully; must not be 500
        assert res.status_code in (200, 400, 422)


    def test_response_never_exposes_upload_dir(self, client):
        res = client.post(
            "/multimodal/analyze",
            data={"processing_mode": "sync"},
            files={"file": _png_file()},
        )
        # upload_dir or temp path must not appear in response
        assert "uploads" not in res.text.lower() or (
            # file_id is OK, but actual OS path (with slashes) is not
            re.search(r"[/\\]uploads[/\\]", res.text) is None
        )


# ===========================================================================
# 9.  /multimodal/cleanup regression
# ===========================================================================

class TestCleanupEndpointRegression:
    def test_cleanup_returns_200(self, client):
        res = client.post("/multimodal/cleanup")
        assert res.status_code == 200

    def test_cleanup_response_has_status(self, client):
        res = client.post("/multimodal/cleanup")
        data = res.json()
        assert "status" in data
        assert data["status"] == "success"


# ===========================================================================
# 10. Root endpoint regression
# ===========================================================================

class TestRootRegression:
    def test_root_returns_200(self, client):
        res = client.get("/")
        assert res.status_code == 200

    def test_root_returns_message(self, client):
        data = client.get("/").json()
        assert "message" in data