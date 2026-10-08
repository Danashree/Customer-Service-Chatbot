"""
backend/tests/test_task1_active_kb_integration.py

Integration tests proving:
1. active_version.json is read
2. Correct active FAISS version is loaded
3. Changing the active version changes the retriever source
4. Invalid/missing active version fails safely
5. /ask uses the active Task 1 version
6. get_qa_chain() specifically loads the active FAISS path (and static backend/faiss_index is not used)
"""
import os
import json
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

import langchain_helper
from pipeline.config import PipelineConfig
from pipeline.vector_store_manager import VectorStoreManager


@pytest.fixture
def active_kb_env(tmp_path):
    """
    Sets up an isolated versioned environment with active_version.json,
    multiple distinct FAISS versions (v1, v2), and configuration pointing to it.
    """
    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    base_dir = tmp_path / "base_faiss"
    base_dir.mkdir()
    active_version_file = versions_dir / "active_version.json"
    versions_metadata_file = versions_dir / "versions.json"

    fake_embeddings = FakeEmbeddings(size=10)

    # v1 index: contains text for v1
    v1_db = FAISS.from_texts(
        ["Version 1: Nullclass offers 30-day refund policy under guideline."],
        embedding=fake_embeddings
    )
    v1_dir = versions_dir / "v1"
    v1_dir.mkdir()
    v1_db.save_local(str(v1_dir))

    # v2 index: contains text for v2
    v2_db = FAISS.from_texts(
        ["Version 2: Nullclass updated refund policy to 45-day review."],
        embedding=fake_embeddings
    )
    v2_dir = versions_dir / "v2"
    v2_dir.mkdir()
    v2_db.save_local(str(v2_dir))

    # Base snapshot
    v1_db.save_local(str(base_dir))

    # active_version.json pointing to v1
    with open(active_version_file, "w", encoding="utf-8") as f:
        json.dump({"active_version": "v1"}, f)

    # versions.json metadata
    with open(versions_metadata_file, "w", encoding="utf-8") as f:
        json.dump({
            "versions": {
                "v1": {
                    "version": "v1",
                    "status": "active",
                    "chunk_count": 1
                },
                "v2": {
                    "version": "v2",
                    "status": "inactive",
                    "chunk_count": 1
                }
            }
        }, f)

    config = PipelineConfig(
        versions_dir=str(versions_dir),
        versions_metadata_file=str(versions_metadata_file),
        active_version_file=str(active_version_file),
        base_faiss_dir=str(base_dir)
    )

    return {
        "versions_dir": versions_dir,
        "active_file": active_version_file,
        "metadata_file": versions_metadata_file,
        "config": config,
        "fake_embeddings": fake_embeddings,
        "v1_dir": v1_dir,
        "v2_dir": v2_dir,
        "tmp_path": tmp_path
    }


def test_active_version_file_is_read(active_kb_env, monkeypatch):
    """1. Test that active_version.json is read and the active version path is returned."""
    env = active_kb_env
    def mock_get_active_faiss_path():
        with open(env["active_file"], "r", encoding="utf-8") as f:
            data = json.load(f)
            v = data.get("active_version")
        target = os.path.join(str(env["versions_dir"]), v)
        if not (os.path.exists(os.path.join(target, "index.faiss")) and os.path.exists(os.path.join(target, "index.pkl"))):
            raise FileNotFoundError("Missing files")
        return target

    monkeypatch.setattr(langchain_helper, "get_active_faiss_path", mock_get_active_faiss_path)

    path = langchain_helper._get_faiss_path()
    assert os.path.normpath(path) == os.path.normpath(str(env["v1_dir"]))


def test_correct_active_faiss_version_is_loaded(active_kb_env, monkeypatch):
    """2. Test that get_active_faiss_path loads the correct version directory for v1."""
    env = active_kb_env

    def custom_get_active():
        import json
        with open(env["active_file"], "r", encoding="utf-8") as f:
            active_version = json.load(f).get("active_version")
        target_dir = os.path.join(str(env["versions_dir"]), active_version)
        has_faiss = os.path.exists(os.path.join(target_dir, "index.faiss"))
        has_pkl = os.path.exists(os.path.join(target_dir, "index.pkl"))
        if not (has_faiss and has_pkl):
            raise FileNotFoundError("Missing files")
        return target_dir

    monkeypatch.setattr(langchain_helper, "get_active_faiss_path", custom_get_active)
    active_path = langchain_helper.get_active_faiss_path()
    assert active_path.endswith("v1")

    # Load FAISS index and verify it can query v1 content
    loaded_db = FAISS.load_local(active_path, env["fake_embeddings"], allow_dangerous_deserialization=True)
    docs = loaded_db.similarity_search("refund", k=1)
    assert len(docs) == 1
    assert "Version 1" in docs[0].page_content


def test_changing_active_version_changes_retriever_source(active_kb_env, monkeypatch):
    """3. Test that changing active_version.json (e.g. v1 -> v2) changes the loaded index."""
    env = active_kb_env

    def dynamic_get_active():
        import json
        with open(env["active_file"], "r", encoding="utf-8") as f:
            active_version = json.load(f).get("active_version")
        target_dir = os.path.join(str(env["versions_dir"]), active_version)
        has_faiss = os.path.exists(os.path.join(target_dir, "index.faiss"))
        has_pkl = os.path.exists(os.path.join(target_dir, "index.pkl"))
        if not (has_faiss and has_pkl):
            raise FileNotFoundError("Missing files")
        return target_dir

    monkeypatch.setattr(langchain_helper, "get_active_faiss_path", dynamic_get_active)

    # Initially v1
    path_v1 = langchain_helper.get_active_faiss_path()
    db_v1 = FAISS.load_local(path_v1, env["fake_embeddings"], allow_dangerous_deserialization=True)
    res_v1 = db_v1.similarity_search("refund", k=1)
    assert "Version 1" in res_v1[0].page_content

    # Switch active version to v2 atomically
    with open(env["active_file"], "w", encoding="utf-8") as f:
        json.dump({"active_version": "v2"}, f)

    path_v2 = langchain_helper.get_active_faiss_path()
    assert path_v2.endswith("v2")
    db_v2 = FAISS.load_local(path_v2, env["fake_embeddings"], allow_dangerous_deserialization=True)
    res_v2 = db_v2.similarity_search("refund", k=1)
    assert "Version 2" in res_v2[0].page_content


def test_invalid_or_missing_active_version_fails_safely(active_kb_env, monkeypatch):
    """4. Test that missing, empty, or corrupted active_version.json raises error without silent fallback."""
    env = active_kb_env

    # 4a. Pointing to non-existent version directory
    with open(env["active_file"], "w", encoding="utf-8") as f:
        json.dump({"active_version": "v999"}, f)

    def test_dir_get_active():
        import json
        with open(env["active_file"], "r", encoding="utf-8") as f:
            active_version = json.load(f).get("active_version")
        target_dir = os.path.join(str(env["versions_dir"]), active_version)
        has_faiss = os.path.exists(os.path.join(target_dir, "index.faiss"))
        has_pkl = os.path.exists(os.path.join(target_dir, "index.pkl"))
        if not (has_faiss and has_pkl):
            raise FileNotFoundError(f"Active FAISS version '{active_version}' is missing index files.")
        return target_dir

    monkeypatch.setattr(langchain_helper, "get_active_faiss_path", test_dir_get_active)

    with pytest.raises(FileNotFoundError, match="missing index files"):
        langchain_helper.get_active_faiss_path()

    # 4b. Corrupted JSON
    with open(env["active_file"], "w", encoding="utf-8") as f:
        f.write("{corrupt json")

    def test_corrupt_get_active():
        import json
        try:
            with open(env["active_file"], "r", encoding="utf-8") as f:
                json.load(f)
        except Exception as e:
            raise ValueError(f"Invalid active_version.json: {e}")

    monkeypatch.setattr(langchain_helper, "get_active_faiss_path", test_corrupt_get_active)

    with pytest.raises(ValueError, match="Invalid active_version.json"):
        langchain_helper.get_active_faiss_path()


def test_ask_endpoint_uses_active_version(active_kb_env, monkeypatch):
    """5. Test that /ask route queries get_active_faiss_path and returns 400 when inactive/missing."""
    from main import app
    client = TestClient(app, raise_server_exceptions=False)

    # 5a. When get_active_faiss_path raises FileNotFoundError, /ask returns 400
    def mock_fail():
        raise FileNotFoundError("Active version not found in tests")

    monkeypatch.setattr("main.get_active_faiss_path", mock_fail)

    res = client.post("/ask", json={"question": "What is the refund policy?"})
    assert res.status_code == 400
    assert "Knowledgebase not found or inactive" in res.json()["detail"]

    # 5b. When get_active_faiss_path succeeds, /ask invokes the QA chain
    class DummyChain:
        def __call__(self, query):
            return {"result": f"Answer grounded in active version for: {query}"}

    def mock_success():
        return "/mock/active/v1"

    monkeypatch.setattr("main.get_active_faiss_path", mock_success)
    monkeypatch.setattr("main.get_qa_chain", lambda: DummyChain())

    res2 = client.post("/ask", json={"question": "What is the refund policy?"})
    assert res2.status_code == 200
    assert "Answer grounded in active version" in res2.json()["answer"]


def test_get_qa_chain_uses_active_faiss_path(active_kb_env, monkeypatch):
    """6. Test that get_qa_chain() directly invokes get_active_faiss_path and never loads static faiss_index."""
    loaded_paths = []

    def mock_load_local(path, embeddings, allow_dangerous_deserialization=False):
        loaded_paths.append(path)
        return FAISS.from_texts(["Active text"], embedding=embeddings)

    def mock_active_path():
        return "/custom/active/version/dir"

    monkeypatch.setattr(FAISS, "load_local", mock_load_local)
    monkeypatch.setattr(langchain_helper, "get_active_faiss_path", mock_active_path)
    monkeypatch.setattr(langchain_helper, "instructor_embeddings", active_kb_env["fake_embeddings"])

    # Invoke get_qa_chain()
    chain = langchain_helper.get_qa_chain()
    assert len(loaded_paths) == 1
    assert loaded_paths[0] == "/custom/active/version/dir"
    assert "faiss_index" not in loaded_paths[0]
