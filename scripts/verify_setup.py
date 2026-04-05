#!/usr/bin/env python3
"""Quick verification script to test RAG setup before running experiments.

Run this locally or on the cluster to verify:
1. ChromaDB is accessible and has data
2. Embeddings load correctly
3. Retrieval returns documents
4. LLM API is working

Usage:
    python scripts/verify_setup.py
    python scripts/verify_setup.py --chroma-path chroma_docling
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_chromadb(chroma_path: str) -> bool:
    """Verify ChromaDB is accessible and has data."""
    print("=" * 50)
    print("1. CHECKING CHROMADB")
    print("=" * 50)

    try:
        import sqlite3
        db_path = Path(chroma_path) / "chroma.sqlite3"

        if not db_path.exists():
            print(f"❌ Database not found: {db_path}")
            return False

        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM embeddings')
        chunks = c.fetchone()[0]
        conn.close()

        if chunks == 0:
            print(f"❌ ChromaDB is empty: 0 chunks")
            return False

        print(f"✅ ChromaDB: {chunks:,} chunks available")
        return True

    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return False


def check_embeddings() -> bool:
    """Verify embeddings model loads correctly."""
    print("\n" + "=" * 50)
    print("2. CHECKING EMBEDDINGS")
    print("=" * 50)

    try:
        from src.config import DEFAULTS, get_embedding_model

        print(f"Loading embeddings: {DEFAULTS.embedding_model}")
        start = time.time()
        embeddings = get_embedding_model(DEFAULTS.embedding_model)
        load_time = time.time() - start

        # Test embedding generation
        test_text = "What is revenue growth?"
        embedding = embeddings.embed_query(test_text)

        print(f"✅ Embeddings loaded in {load_time:.2f}s")
        print(f"✅ Test embedding dimension: {len(embedding)}")
        return True

    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        return False


def check_retrieval(chroma_path: str) -> bool:
    """Verify retrieval returns documents."""
    print("\n" + "=" * 50)
    print("3. CHECKING RETRIEVAL")
    print("=" * 50)

    try:
        from langchain_chroma import Chroma
        from src.config import get_embedding_model, DEFAULTS

        embeddings = get_embedding_model(DEFAULTS.embedding_model)
        db = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )

        query = "What was the total revenue?"
        print(f"Query: '{query}'")

        start = time.time()
        docs = db.similarity_search(query, k=3)
        search_time = time.time() - start

        if not docs:
            print("❌ No documents retrieved")
            return False

        print(f"✅ Retrieved {len(docs)} documents in {search_time:.2f}s")
        print(f"   Source: {docs[0].metadata.get('source', 'unknown')}")
        print(f"   Preview: {docs[0].page_content[:100]}...")
        return True

    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_llm_api() -> bool:
    """Verify LLM API is working."""
    print("\n" + "=" * 50)
    print("4. CHECKING LLM API")
    print("=" * 50)

    try:
        from src.providers import get_provider

        model = "gpt-4o-mini"
        print(f"Testing model: {model}")

        provider = get_provider(model)
        start = time.time()
        response = provider.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'OK' if you can receive this message.",
            max_tokens=10,
            temperature=0.0
        )
        api_time = time.time() - start

        if response.content and "OK" in response.content.upper():
            print(f"✅ LLM API working ({api_time:.2f}s)")
            print(f"   Response: {response.content}")
            return True
        else:
            print(f"⚠️ LLM responded but unexpected: {response.content}")
            return True

    except Exception as e:
        print(f"❌ LLM API error: {e}")
        return False


def check_config() -> bool:
    """Verify configuration is correct."""
    print("\n" + "=" * 50)
    print("5. CHECKING CONFIGURATION")
    print("=" * 50)

    try:
        from src.config import DEFAULTS

        print(f"Default chroma_path: {DEFAULTS.chroma_path}")
        print(f"Default embedding_model: {DEFAULTS.embedding_model}")
        print(f"Default llm_model: {DEFAULTS.llm_model}")
        print(f"Default judge_model: {DEFAULTS.judge_model}")

        if DEFAULTS.chroma_path == "chroma_docling":
            print("✅ Config points to chroma_docling (correct)")
        else:
            print(f"⚠️ Config points to {DEFAULTS.chroma_path}")

        return True

    except Exception as e:
        print(f"❌ Config error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify RAG setup")
    parser.add_argument(
        '--chroma-path', type=str, default='chroma_docling',
        help='Path to ChromaDB directory'
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("RAG SETUP VERIFICATION")
    print("=" * 50)

    results = {
        'ChromaDB': check_chromadb(args.chroma_path),
        'Config': check_config(),
        'Embeddings': check_embeddings(),
        'Retrieval': check_retrieval(args.chroma_path),
        'LLM API': check_llm_api(),
    }

    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)

    all_passed = True
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready to run experiments!")
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before running experiments")
    print("=" * 50)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
