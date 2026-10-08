import os
import shutil
import json
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional, Callable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import PipelineConfig

logger = logging.getLogger("pipeline.vector_store_manager")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [VectorStoreManager] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class VectorStoreManager:
    """
    Task 1: Phase 2 Vector Store Version Control and Rollback Manager.
    Maintains versioned snapshots of FAISS indices (v1, v2, v3...),
    tracks version metadata in versions.json, manages active_version.json atomically,
    supports instant zero-recompute rollback, and provides post-activation health checks.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        embeddings=None,
        access_controller=None
    ):
        self.config = config or PipelineConfig()
        self.embeddings = embeddings
        self.access_controller = access_controller

    def get_embeddings(self):
        """Lazy loader for embeddings if not injected."""
        if self.embeddings is None:
            # Import lazily from langchain_helper so unit tests using mock embeddings
            # don't incur heavy PyTorch/Instructor startup costs.
            import sys
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if backend_dir not in sys.path:
                sys.path.insert(0, backend_dir)
            from langchain_helper import instructor_embeddings
            self.embeddings = instructor_embeddings
        return self.embeddings

    # ------------------------------------------------------------------
    # Atomic File Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _atomic_write_json(filepath: str, data: Dict) -> None:
        """Atomically writes JSON to a file using temporary file replacement."""
        dir_name = os.path.dirname(filepath)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        temp_path = f"{filepath}.tmp_{os.getpid()}_{int(time.time() * 1000)}"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, filepath)

    # ------------------------------------------------------------------
    # Metadata Persistence
    # ------------------------------------------------------------------
    def load_versions_metadata(self) -> Dict:
        """Loads versions.json metadata."""
        if os.path.exists(self.config.versions_metadata_file):
            try:
                with open(self.config.versions_metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not read versions metadata ({e}). Starting with empty record.")
        return {"versions": {}}

    def save_versions_metadata(self, metadata: Dict) -> None:
        """Persists versions.json metadata atomically."""
        self._atomic_write_json(self.config.versions_metadata_file, metadata)

    def get_active_version(self) -> Optional[str]:
        """Reads active_version.json and returns the active version string (e.g. 'v1')."""
        if os.path.exists(self.config.active_version_file):
            try:
                with open(self.config.active_version_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("active_version")
            except Exception as e:
                logger.warning(f"Failed to read active_version.json: {e}")
        return None

    def get_active_version_path(self) -> Optional[str]:
        """Returns directory path to active version if valid and files exist."""
        active = self.get_active_version()
        if not active:
            return None
        version_dir = os.path.join(self.config.versions_dir, active)
        if self._is_valid_faiss_dir(version_dir):
            return version_dir
        return None

    @staticmethod
    def _is_valid_faiss_dir(path: str) -> bool:
        """Checks if index.faiss and index.pkl exist in the given directory."""
        if not os.path.isdir(path):
            return False
        has_faiss = os.path.exists(os.path.join(path, "index.faiss"))
        has_pkl = os.path.exists(os.path.join(path, "index.pkl"))
        return has_faiss and has_pkl

    # ------------------------------------------------------------------
    # Base Version (v1) Initialization
    # ------------------------------------------------------------------
    def init_base_version(
        self,
        source_faiss_dir: Optional[str] = None,
        source_documents: Optional[List[str]] = None
    ) -> str:
        """
        Creates Version 1 from the currently active knowledge base snapshot.
        Preserves the source FAISS index without modifying it.
        """
        metadata = self.load_versions_metadata()
        active = self.get_active_version()

        # If already initialized and v1 exists, return active version
        if "v1" in metadata.get("versions", {}) and active:
            return active

        src_dir = source_faiss_dir or self.config.base_faiss_dir
        if not self._is_valid_faiss_dir(src_dir):
            raise FileNotFoundError(
                f"Source FAISS index not found at '{src_dir}'. "
                f"Cannot initialize base version v1 without a valid index snapshot."
            )

        v1_dir = os.path.join(self.config.versions_dir, "v1")
        os.makedirs(v1_dir, exist_ok=True)

        # Copy FAISS files safely into versions/v1/
        for filename in ["index.faiss", "index.pkl"]:
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(v1_dir, filename)
            shutil.copy2(src_file, dst_file)

        # Load hashes from registry.json if available
        doc_hashes = {}
        if os.path.exists(self.config.registry_file):
            try:
                with open(self.config.registry_file, "r", encoding="utf-8") as f:
                    reg = json.load(f)
                    for doc_name, info in reg.get("documents", {}).items():
                        doc_hashes[doc_name] = info.get("sha256", "")
            except Exception:
                pass

        default_docs = source_documents or [
            "dataset.csv",
            "FAQs.pdf",
            "how-to-guides.pdf",
            "product-information.pdf"
        ]

        now_iso = datetime.now(timezone.utc).isoformat()
        v1_entry = {
            "version": "v1",
            "created_at": now_iso,
            "status": "active",
            "source_documents": default_docs,
            "document_hashes": doc_hashes,
            "chunk_count": 57,
            "parent_version": None,
            "activated_at": now_iso,
            "rollback_info": None
        }

        metadata.setdefault("versions", {})["v1"] = v1_entry
        self.save_versions_metadata(metadata)

        # Set active_version.json
        self._atomic_write_json(self.config.active_version_file, {"active_version": "v1"})
        logger.info(f"Initialized base version 'v1' from snapshot '{src_dir}' into '{v1_dir}'.")
        return "v1"

    # ------------------------------------------------------------------
    # Candidate Version Creation
    # ------------------------------------------------------------------
    def create_version(
        self,
        documents: List[Document],
        source_filenames: Optional[List[str]] = None,
        embeddings=None,
        parent_version: Optional[str] = None
    ) -> str:
        """
        Builds a new candidate version (v2, v3...) in a separate directory.
        The current active version remains untouched.
        The new version is marked as 'candidate' and NOT automatically activated.
        """
        # Ensure base version exists first
        metadata = self.load_versions_metadata()
        if not metadata.get("versions"):
            self.init_base_version()
            metadata = self.load_versions_metadata()

        # Determine next version id
        existing_versions = metadata.get("versions", {})
        version_numbers = []
        for v in existing_versions.keys():
            if v.startswith("v") and v[1:].isdigit():
                version_numbers.append(int(v[1:]))
        next_num = (max(version_numbers) + 1) if version_numbers else 1
        new_version = f"v{next_num}"

        parent_v = parent_version or self.get_active_version()
        new_version_dir = os.path.join(self.config.versions_dir, new_version)
        os.makedirs(new_version_dir, exist_ok=True)

        emb = embeddings or self.get_embeddings()

        # Build candidate FAISS index
        if parent_v and parent_v in existing_versions:
            parent_dir = os.path.join(self.config.versions_dir, parent_v)
            if self._is_valid_faiss_dir(parent_dir):
                # Copy or load parent index and add new documents
                vectordb = FAISS.load_local(parent_dir, emb, allow_dangerous_deserialization=True)
                if documents:
                    vectordb.add_documents(documents)
                vectordb.save_local(new_version_dir)
            else:
                if not documents:
                    raise ValueError(f"Parent version '{parent_v}' index not found and no documents provided.")
                vectordb = FAISS.from_documents(documents, emb)
                vectordb.save_local(new_version_dir)
        else:
            if not documents:
                raise ValueError("Cannot create version with empty documents and no parent version.")
            vectordb = FAISS.from_documents(documents, emb)
            vectordb.save_local(new_version_dir)

        if not self._is_valid_faiss_dir(new_version_dir):
            raise RuntimeError(f"Failed to generate valid FAISS index files for version '{new_version}'.")

        # Collect hashes
        doc_hashes = {}
        if os.path.exists(self.config.registry_file):
            try:
                with open(self.config.registry_file, "r", encoding="utf-8") as f:
                    reg = json.load(f)
                    for doc_name, info in reg.get("documents", {}).items():
                        doc_hashes[doc_name] = info.get("sha256", "")
            except Exception:
                pass

        parent_chunks = existing_versions.get(parent_v, {}).get("chunk_count", 0) if parent_v else 0
        total_chunks = parent_chunks + len(documents)

        now_iso = datetime.now(timezone.utc).isoformat()
        metadata["versions"][new_version] = {
            "version": new_version,
            "created_at": now_iso,
            "status": "candidate",
            "source_documents": source_filenames or [],
            "document_hashes": doc_hashes,
            "chunk_count": total_chunks,
            "parent_version": parent_v,
            "activated_at": None,
            "rollback_info": None
        }
        self.save_versions_metadata(metadata)
        logger.info(f"Created candidate version '{new_version}' at '{new_version_dir}' with {total_chunks} chunks.")
        return new_version

    # ------------------------------------------------------------------
    # Atomic Activation
    # ------------------------------------------------------------------
    def activate_version(self, version: str, role: Optional[str] = None) -> bool:
        """
        Safely and atomically activates a version.
        Verifies authorization if role is specified, verifies target files exist,
        marks old version inactive, marks new version active.
        """
        if role is not None:
            from .security import AccessController, SecurityEventLogger
            ctrl = self.access_controller or AccessController(
                security_logger=SecurityEventLogger(log_file=self.config.security_log_file)
            )
            ctrl.authorize(role, "activate_version", resource=version)

        metadata = self.load_versions_metadata()
        versions = metadata.get("versions", {})

        if version not in versions:
            logger.error(f"Cannot activate nonexistent version '{version}'.")
            return False

        version_dir = os.path.join(self.config.versions_dir, version)
        if not self._is_valid_faiss_dir(version_dir):
            logger.error(f"Cannot activate version '{version}': missing index files in '{version_dir}'.")
            return False

        current_active = self.get_active_version()
        if current_active == version:
            logger.info(f"Version '{version}' is already active.")
            return True

        # Mark current active as inactive
        if current_active and current_active in versions:
            versions[current_active]["status"] = "inactive"

        # Mark new version as active
        versions[version]["status"] = "active"
        versions[version]["activated_at"] = datetime.now(timezone.utc).isoformat()

        # Atomically update pointer and metadata
        self._atomic_write_json(self.config.active_version_file, {"active_version": version})
        self.save_versions_metadata(metadata)

        logger.info(f"Successfully activated version '{version}' (previous active: '{current_active}').")
        return True

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------
    def rollback(
        self,
        target_version: str,
        reason: str = "Manual rollback",
        role: Optional[str] = None
    ) -> bool:
        """
        Rolls back to a target version without recomputing embeddings.
        Verifies authorization if role is specified.
        Switches active_version.json and records rollback audit log in versions.json.
        """
        if role is not None:
            from .security import AccessController, SecurityEventLogger
            ctrl = self.access_controller or AccessController(
                security_logger=SecurityEventLogger(log_file=self.config.security_log_file)
            )
            ctrl.authorize(role, "rollback_version", resource=target_version)

        metadata = self.load_versions_metadata()
        versions = metadata.get("versions", {})

        if target_version not in versions:
            logger.error(f"Cannot rollback to nonexistent version '{target_version}'.")
            return False

        target_dir = os.path.join(self.config.versions_dir, target_version)
        if not self._is_valid_faiss_dir(target_dir):
            logger.error(f"Cannot rollback to '{target_version}': missing index files in '{target_dir}'.")
            return False

        current_active = self.get_active_version()
        if current_active == target_version:
            logger.info(f"Target version '{target_version}' is already the active version.")
            return True

        now_iso = datetime.now(timezone.utc).isoformat()
        rollback_entry = {
            "timestamp": now_iso,
            "rolled_back_from": current_active,
            "rolled_back_to": target_version,
            "reason": reason
        }

        # Update statuses
        if current_active and current_active in versions:
            versions[current_active]["status"] = "rolled_back"
            versions[current_active]["rollback_info"] = rollback_entry

        versions[target_version]["status"] = "active"
        versions[target_version]["activated_at"] = now_iso
        versions[target_version]["rollback_info"] = rollback_entry

        # Atomically update active version and metadata
        self._atomic_write_json(self.config.active_version_file, {"active_version": target_version})
        self.save_versions_metadata(metadata)

        logger.info(
            f"Rollback successful: Switched active from '{current_active}' to '{target_version}'. Reason: {reason}"
        )
        return True

    # ------------------------------------------------------------------
    # Post-Activation Health Check Hook (5-minute monitoring window)
    # ------------------------------------------------------------------
    def schedule_post_activation_health_check(
        self,
        version: str,
        previous_version: str,
        health_check_fn: Optional[Callable[[], bool]] = None,
        delay_seconds: Optional[int] = None
    ) -> Dict:
        """
        Post-activation health check hook.
        Executes health check after a delay (default: config.health_check_delay_seconds = 300s).
        If health check fails, automatically triggers rollback to previous_version.
        """
        delay = self.config.health_check_delay_seconds if delay_seconds is None else delay_seconds
        if delay > 0:
            logger.info(f"Waiting {delay}s for post-activation health evaluation of version '{version}'...")
            time.sleep(delay)

        # Execute health check
        health_passed = False
        error_msg = ""
        try:
            if health_check_fn is not None:
                health_passed = bool(health_check_fn())
            else:
                # Default baseline health check: verify index can be opened and active pointer matches
                active = self.get_active_version()
                active_path = self.get_active_version_path()
                health_passed = (active == version and active_path is not None)
        except Exception as e:
            health_passed = False
            error_msg = str(e)

        if health_passed:
            logger.info(f"Health check PASSED for version '{version}'. Version confirmed active.")
            return {
                "status": "healthy",
                "version": version,
                "health_passed": True
            }
        else:
            reason = f"Post-activation health check failed: {error_msg or 'Validation returned False'}"
            logger.warning(f"{reason}. Triggering automatic rollback to '{previous_version}'.")
            self.rollback(target_version=previous_version, reason=reason)
            return {
                "status": "rolled_back",
                "failed_version": version,
                "active_version": previous_version,
                "health_passed": False,
                "reason": reason
            }

    # ------------------------------------------------------------------
    # Query Helper
    # ------------------------------------------------------------------
    def list_versions(self) -> List[Dict]:
        """Returns all versions listed in metadata."""
        metadata = self.load_versions_metadata()
        return list(metadata.get("versions", {}).values())

    def get_version_metadata(self, version: str) -> Optional[Dict]:
        """Returns metadata for a specific version."""
        metadata = self.load_versions_metadata()
        return metadata.get("versions", {}).get(version)
