"""Lightweight corpus/index manager (stub)."""

from pathlib import Path
from typing import Dict
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CHROMA_PATHS: Dict[str, Path] = {
    "finance": Path(__file__).parent.parent / "chroma",
    "pubmedqa": Path(__file__).parent.parent / "chroma_pubmedqa",
    "cuad": Path(__file__).parent.parent / "chroma_cuad",
    "scienceqa": Path(__file__).parent.parent / "chroma_scienceqa",
}


def load_chroma(corpus_id: str = "finance", embedding_model: str = "text-embedding-3-large") -> Chroma:
    """Load a Chroma index for a corpus id."""
    if corpus_id not in DEFAULT_CHROMA_PATHS:
        raise ValueError(f"Unknown corpus_id '{corpus_id}'")
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return Chroma(
        persist_directory=str(DEFAULT_CHROMA_PATHS[corpus_id]),
        embedding_function=embeddings,
    )
