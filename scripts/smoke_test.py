#!/usr/bin/env python3
"""Smoke test to verify all pipeline components work before full evaluation."""

import sys

def main():
    print("=== 1. CRITICAL: Testing PyArrow Fix ===")
    try:
        import pandas as pd
        from datasets import Dataset
        df = pd.DataFrame({'test': [1, 2, 3]})
        ds = Dataset.from_pandas(df)
        print("✓ HF Datasets <-> PyArrow integration working!")
    except Exception as e:
        print(f"✗ CRASHED: {e}")
        sys.exit(1)

    print("\n=== 2. Testing Reranker (GPU Check) ===")
    try:
        import torch
        from sentence_transformers import CrossEncoder
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Device detected: {device}")
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        score = model.predict([('query', 'document')])
        print(f"✓ Reranker loaded and working! Score: {score}")
    except Exception as e:
        print(f"✗ RERANKER FAILED: {e}")
        sys.exit(1)

    print("\n=== 3. Testing Database Access ===")
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        db = Chroma(persist_directory='chroma', embedding_function=embeddings)
        count = db._collection.count()
        if count > 0:
            print(f"✓ ChromaDB connected. Chunks: {count}")
        else:
            print("⚠ ChromaDB connected but EMPTY (0 chunks). This will fail.")
            sys.exit(1)
    except Exception as e:
        print(f"✗ DB FAILED: {e}")
        sys.exit(1)

    print("\n=== 4. Testing OpenAI Embedding API ===")
    try:
        test_emb = embeddings.embed_query('test query')
        print(f"✓ Embedding works! Dimension: {len(test_emb)}")
    except Exception as e:
        print(f"✗ EMBEDDING FAILED: {e}")
        sys.exit(1)

    print("\n=== 5. Testing Together API (LLM) ===")
    try:
        import os
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ['TOGETHER_API_KEY'],
            base_url='https://api.together.xyz/v1'
        )
        response = client.chat.completions.create(
            model='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
            messages=[{'role': 'user', 'content': 'Say OK'}],
            max_tokens=10
        )
        print(f"✓ LLM response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"✗ LLM FAILED: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("🚀 ALL SYSTEMS GO - Safe to run full evaluation!")
    print("=" * 50)


if __name__ == "__main__":
    main()
