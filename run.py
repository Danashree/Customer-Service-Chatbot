"""
Customer Service Bot — Main Entry Point Runner
==============================================
Convenient shortcut for running the server, demo, tests, or status check.

Usage:
  python run.py           -> Interactive CLI menu (Option 1-5)
  python run.py server    -> Start FastAPI API server on port 8000
  python run.py demo      -> Run Task 1-6 practical verification demo
  python run.py status    -> Check Knowledge-Base & Pipeline health status
  python run.py test      -> Run automated test suite
  python run.py run       -> Execute knowledge-base ingestion pipeline
"""
import sys
import subprocess

if __name__ == "__main__":
    cmd = [sys.executable, "run_pipeline.py"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
