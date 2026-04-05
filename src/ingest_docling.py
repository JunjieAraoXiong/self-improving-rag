#!/usr/bin/env python3
"""
Optimized PDF ingestion using Docling for accurate table extraction.

Docling provides 97.9% table cell accuracy using the TableFormer model,
significantly better than unstructured's approach.

Usage:
    # Local (GPU recommended)
    python src/ingest_docling.py --input-dir data/pdfs --output-dir chroma

    # On cluster
    sbatch scripts/ingest_docling.slurm
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Try to import torch for GPU memory management
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Setup paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import get_embedding_model, DEFAULTS
from src.metadata_utils import parse_filename


def cleanup_memory():
    """Force garbage collection and clear GPU cache to prevent memory corruption."""
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def process_pdf_docling(pdf_path: Path, chunk_size: int = 2500) -> List[Document]:
    """
    Process a single PDF with Docling for accurate table extraction.

    Args:
        pdf_path: Path to PDF file
        chunk_size: Maximum characters per prose chunk (tables are never split)

    Returns:
        List of Document objects with metadata
    """
    from docling.document_converter import DocumentConverter

    chunks = []

    # Parse filename for metadata
    file_meta = parse_filename(pdf_path.name)
    if file_meta:
        file_meta_dict = file_meta.to_dict()
    else:
        # Fallback for non-standard filenames
        file_meta_dict = {"source_file": pdf_path.name}

    try:
        # Convert PDF with Docling (uses TableFormer for tables)
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))

        # Export to markdown (tables preserved as | col | col | format)
        markdown = result.document.export_to_markdown()

    except Exception as e:
        print(f"  ✗ Docling error on {pdf_path.name}: {e}")
        return []

    # Table-aware chunking
    # Tables (lines starting with |) are kept as single chunks
    # Prose is chunked at chunk_size limit
    current_chunk = ""
    current_type = "prose"
    in_table = False

    for line in markdown.split('\n'):
        line_stripped = line.strip()

        # Detect table start
        if line_stripped.startswith('|') and not in_table:
            # Save any accumulated prose first
            if current_chunk.strip():
                chunks.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={**file_meta_dict, 'element_type': 'prose', 'source': str(pdf_path)}
                ))
            # Start new table
            current_chunk = line + '\n'
            current_type = "table"
            in_table = True

        elif in_table:
            # Continue table if line starts with | or is separator (---)
            if line_stripped.startswith('|') or (line_stripped.startswith('-') and '|' in current_chunk):
                current_chunk += line + '\n'
            else:
                # Table ended - save as single chunk (never split)
                if current_chunk.strip():
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata={**file_meta_dict, 'element_type': 'table', 'source': str(pdf_path)}
                    ))
                # Start prose chunk
                current_chunk = line + '\n'
                current_type = "prose"
                in_table = False

        else:
            # Prose - accumulate and chunk at size limit
            current_chunk += line + '\n'

            if len(current_chunk) > chunk_size:
                # Find a good break point (end of sentence or paragraph)
                break_point = current_chunk.rfind('\n\n', 0, chunk_size)
                if break_point == -1:
                    break_point = current_chunk.rfind('. ', 0, chunk_size)
                if break_point == -1:
                    break_point = chunk_size

                chunk_text = current_chunk[:break_point].strip()
                if chunk_text:
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata={**file_meta_dict, 'element_type': 'prose', 'source': str(pdf_path)}
                    ))
                current_chunk = current_chunk[break_point:].strip() + '\n'

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(Document(
            page_content=current_chunk.strip(),
            metadata={**file_meta_dict, 'element_type': current_type, 'source': str(pdf_path)}
        ))

    return chunks


def get_processed_files(chroma_path: str) -> set:
    """Get set of already processed filenames from ChromaDB."""
    if not os.path.exists(chroma_path):
        return set()

    try:
        embeddings = get_embedding_model(DEFAULTS.embedding_model)
        db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        result = db.get(include=['metadatas'])

        processed = set()
        for meta in result['metadatas']:
            if meta and 'source_file' in meta:
                processed.add(meta['source_file'])

        return processed
    except Exception as e:
        print(f"Warning: Could not check existing database: {e}")
        return set()


def run_ingestion(
    input_dir: str,
    output_dir: str,
    chunk_size: int = 2500,
    batch_size: int = 10,
    sample: Optional[int] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
):
    """
    Run Docling-based ingestion with batch processing.

    Args:
        input_dir: Directory containing PDF files
        output_dir: ChromaDB output directory
        chunk_size: Max characters per prose chunk
        batch_size: Files to process before saving to ChromaDB
        sample: Optional limit on number of files to process
        start_idx: Start index for parallel processing (default: 0)
        end_idx: End index for parallel processing (default: None = all)
    """
    # Get list of PDF files
    pdf_files = sorted(Path(input_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    if sample:
        pdf_files = pdf_files[:sample]

    # Apply start/end indices for parallel processing
    if end_idx is not None:
        pdf_files = pdf_files[start_idx:end_idx]
    elif start_idx > 0:
        pdf_files = pdf_files[start_idx:]

    if start_idx > 0 or end_idx is not None:
        print(f"Parallel mode: processing files [{start_idx}:{end_idx}]")

    total_files = len(pdf_files)

    print("=" * 70)
    print("DOCLING INGESTION (GPU-Accelerated Table Extraction)")
    print("=" * 70)
    print(f"Input Dir:    {input_dir}")
    print(f"Output Dir:   {output_dir}")
    print(f"Total Files:  {total_files}")
    print(f"Chunk Size:   {chunk_size}")
    print(f"Batch Size:   {batch_size}")
    print("=" * 70)

    # Check for already processed files
    processed_files = get_processed_files(output_dir)
    files_to_process = [p for p in pdf_files if p.name not in processed_files]
    skipped = total_files - len(files_to_process)

    if skipped > 0:
        print(f"\nSkipping {skipped} already-ingested files")

    if not files_to_process:
        print("All files already processed!")
        return

    # Initialize ChromaDB
    embeddings = get_embedding_model(DEFAULTS.embedding_model)
    db = Chroma(persist_directory=output_dir, embedding_function=embeddings)

    # Process in batches
    start_time = time.time()
    total_chunks = 0

    for i in range(0, len(files_to_process), batch_size):
        batch = files_to_process[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(files_to_process) + batch_size - 1) // batch_size

        print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} files...")

        batch_chunks = []

        for pdf_path in batch:
            try:
                chunks = process_pdf_docling(pdf_path, chunk_size)
                batch_chunks.extend(chunks)
                print(f"  ✓ {pdf_path.name} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"  ✗ FAILED {pdf_path.name}: {str(e)[:80]}")

        # Save batch to ChromaDB
        if batch_chunks:
            print(f"  -> Saving {len(batch_chunks)} chunks to ChromaDB...", end=" ", flush=True)
            try:
                # ChromaDB batch size limit
                CHROMA_MAX_BATCH = 5000
                for k in range(0, len(batch_chunks), CHROMA_MAX_BATCH):
                    sub_batch = batch_chunks[k:k + CHROMA_MAX_BATCH]
                    db.add_documents(sub_batch)
                    print(".", end="", flush=True)
                print(" Done.")
                total_chunks += len(batch_chunks)
            except Exception as e:
                print(f" ERROR: {e}")

        # Progress update
        elapsed = time.time() - start_time
        files_done = min(i + batch_size, len(files_to_process))
        rate = files_done / elapsed if elapsed > 0 else 0
        eta = (len(files_to_process) - files_done) / rate if rate > 0 else 0
        print(f"  Progress: {files_done}/{len(files_to_process)} files, ETA: {eta/60:.1f} min")

        # Clean up memory after each batch to prevent heap corruption
        cleanup_memory()
        print("  Memory cleaned up.")

    # Summary
    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"Total files processed: {len(files_to_process)}")
    print(f"Total chunks created:  {total_chunks}")
    print(f"Time elapsed:          {(time.time() - start_time)/60:.1f} minutes")
    print(f"ChromaDB location:     {output_dir}")

    # Verify
    final_count = db._collection.count()
    print(f"Final ChromaDB count:  {final_count}")


def main():
    parser = argparse.ArgumentParser(description="Docling-based PDF Ingestion")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with PDFs")
    parser.add_argument("--output-dir", type=str, default="chroma", help="ChromaDB output")
    parser.add_argument("--chunk-size", type=int, default=2500, help="Max chars per chunk")
    parser.add_argument("--batch-size", type=int, default=10, help="Files per batch")
    parser.add_argument("--sample", type=int, help="Only process N files (for testing)")
    parser.add_argument("--start", type=int, default=0, help="Start index for parallel processing")
    parser.add_argument("--end", type=int, default=None, help="End index for parallel processing")

    args = parser.parse_args()

    run_ingestion(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        sample=args.sample,
        start_idx=args.start,
        end_idx=args.end,
    )


if __name__ == "__main__":
    main()
