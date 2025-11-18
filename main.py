"""Main entry point to run the Parenting RAG pipeline end-to-end.

Usage (PowerShell):

    python main.py -q "How can I handle toddler tantrums?" --rebuild

Flags:
  -q / --question  The question to ask (default sample question)
  --rebuild        Force rebuilding the Chroma collection even if it exists
  -k               Number of chunks to retrieve (default config.DEFAULT_TOP_K)
"""
from __future__ import annotations
import argparse
import shutil
from config import DEFAULT_TOP_K


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--question", type=str, default="How can I deal with my child's tantrums?")
    p.add_argument("-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild of vector store")
    p.add_argument("--db-dir", type=str, default=None, help="Override Chroma DB directory (useful on Windows if locked)")
    return p.parse_args()


def maybe_rebuild(force: bool, chroma_dir_path):
    from pathlib import Path
    chroma_dir = Path(chroma_dir_path)
    if force and chroma_dir.exists():
        print("Force rebuild requested: deleting existing Chroma directory ...")
        try:
            shutil.rmtree(chroma_dir)
        except PermissionError as e:
            print("PermissionError while deleting Chroma directory (file may be in use).")
            print("- Close any running notebooks, REPLs, or apps using the DB.")
            print("- Alternatively, use --db-dir to write to a fresh directory (e.g., chroma_db2)")
            raise
    from vectorstore_build import build_if_needed
    build_if_needed()


def main():
    args = parse_args()
    # Allow overriding DB path before loading modules that read config.CHROMA_DIR
    if args.db_dir:
        import os
        os.environ["CHROMA_DIR"] = args.db_dir
    # Import after possible env override so modules pick up the new path
    from config import CHROMA_DIR
    maybe_rebuild(args.rebuild, CHROMA_DIR)
    from rag import answer
    result = answer(args.question, k=args.k)
    print("\n=== QUESTION ===\n" + result["question"])
    print("\n=== ANSWER ===\n" + result["answer"])
    print("\n=== SOURCES ===")
    for c in result["chunks"]:
        print(f"- {c['id']} (source={c['source']})")

if __name__ == "__main__":
    main()
