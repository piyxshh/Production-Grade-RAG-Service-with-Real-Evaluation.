"""CLI script to run the complete RAG evaluation suite.

Usage:
    python scripts/run_eval.py
"""
import asyncio
from scripts.evaluate import main

if __name__ == "__main__":
    asyncio.run(main())

