"""
Customer Service Bot — Backend Directory Shortcut Runner
========================================================
Delegates to root run_pipeline.py when invoked directly from the backend/ directory.
"""
import os
import sys
import subprocess

if __name__ == "__main__":
    # Resolve project root directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(backend_dir)
    target_script = os.path.join(root_dir, "run_pipeline.py")

    cmd = [sys.executable, target_script] + sys.argv[1:]
    sys.exit(subprocess.call(cmd, cwd=root_dir))
