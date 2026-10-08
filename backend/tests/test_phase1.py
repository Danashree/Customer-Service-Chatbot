import os
import json
import pytest
from reportlab.pdfgen import canvas
import pypdf

from pipeline.config import PipelineConfig
from pipeline.document_processor import DocumentProcessor


def create_pdf(filepath: str, text: str = "Sample content for testing.") -> None:
    """Helper to generate a valid PDF with given text."""
    c = canvas.Canvas(filepath)
    c.drawString(100, 750, text)
    c.save()


def create_blank_pdf(filepath: str) -> None:
    """Helper to generate a valid PDF with blank/whitespace-only content."""
    c = canvas.Canvas(filepath)
    # Page with nothing drawn or only spaces
    c.drawString(100, 750, "   ")
    c.save()


@pytest.fixture
def pipeline_env(tmp_path):
    """Sets up an isolated sandbox environment for testing the pipeline."""
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    quarantine_dir = kb_dir / "quarantine"
    registry_file = tmp_path / "registry.json"
    report_file = quarantine_dir / "quarantine_report.json"

    config = PipelineConfig(
        kb_dir=str(kb_dir),
        quarantine_dir=str(quarantine_dir),
        registry_file=str(registry_file),
        quarantine_report_file=str(report_file),
        chunk_size=200,
        chunk_overlap=50
    )
    processor = DocumentProcessor(config=config)
    return processor, config, kb_dir, quarantine_dir, registry_file, report_file


def test_new_pdf_detection(pipeline_env):
    """Test 1: New PDF is discovered, validated, extracted, chunked, and registered."""
    processor, config, kb_dir, quarantine_dir, registry_file, _ = pipeline_env
    pdf_path = os.path.join(kb_dir, "doc1.pdf")
    create_pdf(pdf_path, "This is a new document with valid knowledge base content.")

    res = processor.process_knowledge_base()

    assert "doc1.pdf" in res.new_documents
    assert "doc1.pdf" in res.processed_documents
    assert len(res.chunks) > 0
    assert res.chunks[0].metadata["source"] == "doc1.pdf"
    assert res.chunks[0].metadata["page"] == 1
    assert res.chunks[0].metadata["doc_type"] == "pdf"

    # Verify registry persistence
    assert os.path.exists(registry_file)
    with open(registry_file, "r") as f:
        reg = json.load(f)
    assert "doc1.pdf" in reg["documents"]
    assert reg["documents"]["doc1.pdf"]["status"] == "processed"


def test_unchanged_pdf_is_skipped(pipeline_env):
    """Test 2: When the same PDF is processed again without changes, it is skipped."""
    processor, config, kb_dir, _, _, _ = pipeline_env
    pdf_path = os.path.join(kb_dir, "doc_stable.pdf")
    create_pdf(pdf_path, "Stable content that will not change between runs.")

    # First run: should be new
    res1 = processor.process_knowledge_base()
    assert "doc_stable.pdf" in res1.new_documents
    assert len(res1.chunks) > 0

    # Second run: should be skipped as unchanged
    res2 = processor.process_knowledge_base()
    assert "doc_stable.pdf" in res2.unchanged_documents
    assert "doc_stable.pdf" not in res2.new_documents
    assert len(res2.chunks) == 0  # No redundant chunking


def test_modified_pdf_is_detected(pipeline_env):
    """Test 3: If content of a known PDF changes, it is recognized as modified and re-processed."""
    processor, config, kb_dir, _, registry_file, _ = pipeline_env
    pdf_path = os.path.join(kb_dir, "doc_mod.pdf")
    create_pdf(pdf_path, "Initial revision version 1.0.")

    # First run
    res1 = processor.process_knowledge_base()
    assert "doc_mod.pdf" in res1.new_documents
    old_hash = processor.calculate_sha256(pdf_path)

    # Modify the PDF
    create_pdf(pdf_path, "Updated revision version 2.0 with new changes.")
    new_hash = processor.calculate_sha256(pdf_path)
    assert old_hash != new_hash

    # Second run: should detect modification
    res2 = processor.process_knowledge_base()
    assert "doc_mod.pdf" in res2.modified_documents
    assert "doc_mod.pdf" not in res2.unchanged_documents
    assert len(res2.chunks) > 0

    # Verify updated hash in registry
    with open(registry_file, "r") as f:
        reg = json.load(f)
    assert reg["documents"]["doc_mod.pdf"]["sha256"] == new_hash


def test_duplicate_detection(pipeline_env):
    """Test 4: Exact duplicate files with different names are caught and skipped."""
    import shutil
    processor, config, kb_dir, _, _, _ = pipeline_env
    file1 = os.path.join(kb_dir, "original.pdf")
    file2 = os.path.join(kb_dir, "copy_of_original.pdf")
    content = "Exact duplicate test content."
    create_pdf(file1, content)
    shutil.copyfile(file1, file2)  # Exact binary duplicate with identical SHA256

    res = processor.process_knowledge_base()

    # One file must be processed, the other flagged as duplicate
    all_seen = res.new_documents + res.duplicate_documents
    assert "original.pdf" in all_seen
    assert "copy_of_original.pdf" in all_seen
    assert len(res.duplicate_documents) == 1
    assert len(res.processed_documents) == 1



def test_invalid_pdf_quarantine(pipeline_env):
    """Test 5: Corrupted / unreadable PDF is quarantined and recorded in quarantine report."""
    processor, config, kb_dir, quarantine_dir, _, report_file = pipeline_env
    corrupt_file = os.path.join(kb_dir, "broken.pdf")
    with open(corrupt_file, "wb") as f:
        f.write(b"%PDF-1.4 THIS IS NOT A REAL VALID PDF HEADER JUNK DATA")

    res = processor.process_knowledge_base()

    assert len(res.quarantined_documents) == 1
    assert res.quarantined_documents[0]["filename"] == "broken.pdf"

    # Verify moved from kb_dir to quarantine_dir
    assert not os.path.exists(corrupt_file)
    assert os.path.exists(os.path.join(quarantine_dir, "broken.pdf"))

    # Verify report was generated
    assert os.path.exists(report_file)
    with open(report_file, "r") as f:
        reports = json.load(f)
    assert len(reports) == 1
    assert reports[0]["filename"] == "broken.pdf"
    assert "Corrupted" in reports[0]["reason"] or "Failed" in reports[0]["reason"]


def test_empty_pdf_rejection(pipeline_env):
    """Test 6: Empty or whitespace-only PDF is rejected and moved to quarantine."""
    processor, config, kb_dir, quarantine_dir, _, report_file = pipeline_env
    blank_file = os.path.join(kb_dir, "empty_doc.pdf")
    create_blank_pdf(blank_file)

    res = processor.process_knowledge_base()

    assert len(res.quarantined_documents) == 1
    assert res.quarantined_documents[0]["filename"] == "empty_doc.pdf"

    # Verify moved to quarantine
    assert not os.path.exists(blank_file)
    assert os.path.exists(os.path.join(quarantine_dir, "empty_doc.pdf"))

    # Verify report specifies empty content
    with open(report_file, "r") as f:
        reports = json.load(f)
    assert reports[0]["filename"] == "empty_doc.pdf"
    assert "Empty" in reports[0]["reason"]
