import os
import json
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

from pipeline.config import PipelineConfig
from pipeline.vector_store_manager import VectorStoreManager


@pytest.fixture
def fake_embeddings():
    """Provides fast, deterministic fake embeddings for isolated unit tests."""
    return FakeEmbeddings(size=10)


@pytest.fixture
def test_env(tmp_path, fake_embeddings):
    """
    Creates an isolated sandbox environment with a pre-built mock base FAISS index,
    so no real production data or heavy PyTorch models are touched.
    """
    base_dir = tmp_path / "base_faiss"
    base_dir.mkdir()
    versions_dir = tmp_path / "versions"
    registry_file = tmp_path / "registry.json"

    # Create a small valid FAISS index as the base snapshot
    base_vectordb = FAISS.from_texts(
        texts=["Nullclass provides data science courses.", "Nullclass offers virtual internships."],
        embedding=fake_embeddings
    )
    base_vectordb.save_local(str(base_dir))

    # Pre-populate dummy registry
    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "documents": {
                    "dataset.csv": {"sha256": "dummy_csv_hash_123"},
                    "FAQs.pdf": {"sha256": "dummy_pdf_hash_456"}
                },
                "hashes": {
                    "dummy_csv_hash_123": "dataset.csv",
                    "dummy_pdf_hash_456": "FAQs.pdf"
                }
            },
            f
        )

    config = PipelineConfig(
        versions_dir=str(versions_dir),
        versions_metadata_file=str(versions_dir / "versions.json"),
        active_version_file=str(versions_dir / "active_version.json"),
        base_faiss_dir=str(base_dir),
        registry_file=str(registry_file),
        health_check_delay_seconds=0  # Fast tests without 5-minute sleep
    )
    manager = VectorStoreManager(config=config, embeddings=fake_embeddings)
    return manager, config, tmp_path, fake_embeddings


def test_initial_v1_creation(test_env):
    """Test 1 & 4: Base version v1 is created from snapshot and active_version.json is initialized."""
    manager, config, _, _ = test_env
    v = manager.init_base_version()

    assert v == "v1"
    v1_dir = os.path.join(config.versions_dir, "v1")
    assert os.path.isdir(v1_dir)
    assert os.path.exists(os.path.join(v1_dir, "index.faiss"))
    assert os.path.exists(os.path.join(v1_dir, "index.pkl"))

    # Verify active version pointer
    assert manager.get_active_version() == "v1"
    with open(config.active_version_file, "r") as f:
        ptr = json.load(f)
    assert ptr.get("active_version") == "v1"


def test_new_candidate_version_creation(test_env):
    """Test 2: Candidate version v2 is built in separate directory and marked as candidate."""
    manager, config, _, fake_embeddings = test_env
    manager.init_base_version()

    new_docs = [
        Document(page_content="New product information guide.", metadata={"source": "product.pdf", "page": 1})
    ]
    candidate = manager.create_version(
        documents=new_docs,
        source_filenames=["product.pdf"],
        embeddings=fake_embeddings
    )

    assert candidate == "v2"
    v2_dir = os.path.join(config.versions_dir, "v2")
    assert os.path.isdir(v2_dir)
    assert os.path.exists(os.path.join(v2_dir, "index.faiss"))
    assert os.path.exists(os.path.join(v2_dir, "index.pkl"))

    # Critical: Active version must still be v1
    assert manager.get_active_version() == "v1"

    meta = manager.get_version_metadata("v2")
    assert meta["status"] == "candidate"
    assert meta["parent_version"] == "v1"


def test_version_metadata_persistence(test_env):
    """Test 3: Metadata is persisted in versions.json with all required audit fields."""
    manager, config, _, _ = test_env
    manager.init_base_version()

    assert os.path.exists(config.versions_metadata_file)
    with open(config.versions_metadata_file, "r") as f:
        meta = json.load(f)

    v1_meta = meta["versions"]["v1"]
    assert v1_meta["version"] == "v1"
    assert v1_meta["status"] == "active"
    assert "created_at" in v1_meta
    assert "source_documents" in v1_meta
    assert "document_hashes" in v1_meta
    assert "chunk_count" in v1_meta
    assert "activated_at" in v1_meta


def test_activating_v2(test_env):
    """Test 5: Activating v2 updates active_version.json and switches statuses atomically."""
    manager, config, _, fake_embeddings = test_env
    manager.init_base_version()

    new_docs = [Document(page_content="FAQ update.", metadata={"source": "faq.pdf"})]
    manager.create_version(documents=new_docs, source_filenames=["faq.pdf"], embeddings=fake_embeddings)

    success = manager.activate_version("v2")
    assert success is True

    # Check active version pointer
    assert manager.get_active_version() == "v2"
    with open(config.active_version_file, "r") as f:
        ptr = json.load(f)
    assert ptr["active_version"] == "v2"

    # Check metadata statuses
    v1_meta = manager.get_version_metadata("v1")
    v2_meta = manager.get_version_metadata("v2")
    assert v1_meta["status"] == "inactive"
    assert v2_meta["status"] == "active"
    assert v2_meta["activated_at"] is not None


def test_rollback_v2_to_v1(test_env):
    """Test 6: Rolling back from v2 to v1 restores v1 as active and logs rollback audit info."""
    manager, config, _, fake_embeddings = test_env
    manager.init_base_version()

    new_docs = [Document(page_content="FAQ update.", metadata={"source": "faq.pdf"})]
    manager.create_version(documents=new_docs, source_filenames=["faq.pdf"], embeddings=fake_embeddings)
    manager.activate_version("v2")
    assert manager.get_active_version() == "v2"

    # Perform rollback
    rb_success = manager.rollback("v1", reason="Manual quality issue rollback")
    assert rb_success is True
    assert manager.get_active_version() == "v1"

    v1_meta = manager.get_version_metadata("v1")
    v2_meta = manager.get_version_metadata("v2")
    assert v1_meta["status"] == "active"
    assert v2_meta["status"] == "rolled_back"
    assert v2_meta["rollback_info"]["reason"] == "Manual quality issue rollback"
    assert v2_meta["rollback_info"]["rolled_back_to"] == "v1"


def test_rollback_does_not_rebuild_index(test_env):
    """Test 7: Rollback is instantaneous and switches pointer without modifying index bytes."""
    manager, config, _, fake_embeddings = test_env
    manager.init_base_version()

    new_docs = [Document(page_content="FAQ update.", metadata={"source": "faq.pdf"})]
    manager.create_version(documents=new_docs, source_filenames=["faq.pdf"], embeddings=fake_embeddings)
    manager.activate_version("v2")

    v1_index_file = os.path.join(config.versions_dir, "v1", "index.faiss")
    v1_mtime_before = os.path.getmtime(v1_index_file)
    v1_size_before = os.path.getsize(v1_index_file)

    # Rollback
    manager.rollback("v1")

    # Verify v1 index was NOT rebuilt or re-written
    assert os.path.getmtime(v1_index_file) == v1_mtime_before
    assert os.path.getsize(v1_index_file) == v1_size_before


def test_previous_versions_remain_available(test_env):
    """Test 8: Both previous and new versions persist on disk after activation and rollback."""
    manager, config, _, fake_embeddings = test_env
    manager.init_base_version()

    new_docs = [Document(page_content="FAQ update.", metadata={"source": "faq.pdf"})]
    manager.create_version(documents=new_docs, source_filenames=["faq.pdf"], embeddings=fake_embeddings)
    manager.activate_version("v2")
    manager.rollback("v1")

    all_versions = manager.list_versions()
    version_ids = [v["version"] for v in all_versions]
    assert "v1" in version_ids
    assert "v2" in version_ids

    # Both directories must physically exist
    assert os.path.exists(os.path.join(config.versions_dir, "v1", "index.faiss"))
    assert os.path.exists(os.path.join(config.versions_dir, "v2", "index.faiss"))


def test_invalid_version_cannot_be_activated(test_env):
    """Test 9: Attempting to activate a nonexistent or broken version fails safely."""
    manager, config, _, _ = test_env
    manager.init_base_version()

    # Nonexistent version
    res_fake = manager.activate_version("v999")
    assert res_fake is False
    assert manager.get_active_version() == "v1"

    # Corrupt version with missing files
    broken_dir = os.path.join(config.versions_dir, "broken_v")
    os.makedirs(broken_dir, exist_ok=True)
    meta = manager.load_versions_metadata()
    meta["versions"]["broken_v"] = {"version": "broken_v"}
    manager.save_versions_metadata(meta)

    res_broken = manager.activate_version("broken_v")
    assert res_broken is False
    assert manager.get_active_version() == "v1"


def test_simulated_health_failure_automatic_rollback(test_env):
    """Test 10: Post-activation health check failure triggers automatic rollback."""
    manager, config, _, fake_embeddings = test_env
    manager.init_base_version()

    new_docs = [Document(page_content="Flawed update.", metadata={"source": "flawed.pdf"})]
    manager.create_version(documents=new_docs, source_filenames=["flawed.pdf"], embeddings=fake_embeddings)
    manager.activate_version("v2")
    assert manager.get_active_version() == "v2"

    # Simulate post-activation health check that fails
    report = manager.schedule_post_activation_health_check(
        version="v2",
        previous_version="v1",
        health_check_fn=lambda: False,  # Simulated failure
        delay_seconds=0
    )

    assert report["status"] == "rolled_back"
    assert report["health_passed"] is False
    # Active version should have automatically fallen back to v1
    assert manager.get_active_version() == "v1"


def test_simulated_health_success_keeps_version_active(test_env):
    """Test 11: Post-activation health check success keeps new version active."""
    manager, config, _, fake_embeddings = test_env
    manager.init_base_version()

    new_docs = [Document(page_content="Good update.", metadata={"source": "good.pdf"})]
    manager.create_version(documents=new_docs, source_filenames=["good.pdf"], embeddings=fake_embeddings)
    manager.activate_version("v2")
    assert manager.get_active_version() == "v2"

    # Simulate post-activation health check that passes
    report = manager.schedule_post_activation_health_check(
        version="v2",
        previous_version="v1",
        health_check_fn=lambda: True,  # Simulated success
        delay_seconds=0
    )

    assert report["status"] == "healthy"
    assert report["health_passed"] is True
    # Active version should remain v2
    assert manager.get_active_version() == "v2"
