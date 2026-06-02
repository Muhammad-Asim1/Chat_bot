#!/usr/bin/env python3
"""
Reproducibility Script for NUST Policy RAG System

This script ensures reproducible results by:
1. Setting all random seeds
2. Documenting configuration
3. Reindexing documents (optional)
4. Running evaluation
5. Saving results with full metadata

Usage:
    python reproduce.py                    # Run evaluation only
    python reproduce.py --reindex          # Clear index and reprocess documents
    python reproduce.py --full             # Full pipeline: reindex + evaluate
    python reproduce.py --config-only      # Just show configuration
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, Any

import numpy as np

# Set seeds BEFORE importing anything else
SEED = 42

def set_all_seeds(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to set torch seed if available (for future use)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    print(f"🎲 Random seeds set to {seed}")

# Set seeds immediately
set_all_seeds(SEED)

# Now import project modules
from config import Config
from vectorstore import (
    process_all_pdfs_in_folder, 
    get_vectorstore_stats,
    get_processed_files
)


def get_system_info() -> Dict[str, Any]:
    """Collect system and environment information."""
    import platform
    
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
    }


def get_config_snapshot() -> Dict[str, Any]:
    """Capture all configuration settings."""
    return {
        # Reproducibility
        "random_seed": Config.RANDOM_SEED,
        "index_version": Config.INDEX_VERSION,
        
        # Models
        "embedding_model": Config.JINA_EMBEDDING_MODEL,
        "reranker_model": Config.JINA_RERANKER_MODEL,
        "generation_model": Config.GOOGLE_MODEL,
        "generation_temperature": Config.GENERATION_TEMPERATURE,
        "generation_max_tokens": Config.GENERATION_MAX_TOKENS,
        
        # Qdrant
        "qdrant_host": Config.QDRANT_HOST,
        "qdrant_port": Config.QDRANT_PORT,
        "qdrant_collection": Config.QDRANT_COLLECTION_NAME,
        
        # Chunking
        "chunking_strategy": Config.CHUNKING_STRATEGY,
        "chunk_size": Config.CHUNK_SIZE,
        "chunk_overlap": Config.CHUNK_OVERLAP,
        
        # Retrieval
        "prefetch_limit": Config.PREFETCH_LIMIT,
        "rerank_limit": Config.RERANK_LIMIT,
        "final_limit": Config.FINAL_LIMIT,
        
        # Abstention
        "min_relevance_score": Config.MIN_RELEVANCE_SCORE,
        
        # Citations
        "strict_citation_mode": Config.STRICT_CITATION_MODE,
        "enable_citations": Config.ENABLE_CITATIONS,
        "enable_page_links": Config.ENABLE_PAGE_LINKS,
        
        # Staleness
        "stale_threshold_years": Config.STALE_THRESHOLD_YEARS,
        "potentially_stale_years": Config.POTENTIALLY_STALE_YEARS,
        
        # Documents
        "pdf_source_urls": Config.PDF_SOURCE_URLS,
        "pdf_publication_dates": Config.PDF_PUBLICATION_DATES,
    }


def get_index_snapshot() -> Dict[str, Any]:
    """Capture index state."""
    stats = get_vectorstore_stats()
    processed = get_processed_files()
    
    return {
        "exists": stats.get("exists", False),
        "total_files": stats.get("total_files", 0),
        "total_chunks": stats.get("total_chunks", 0),
        "files": stats.get("files", []),
        "processed_metadata": processed,
    }


def display_config():
    """Display current configuration."""
    print("\n" + "=" * 60)
    print("📋 CONFIGURATION SNAPSHOT")
    print("=" * 60)
    
    config = get_config_snapshot()
    
    print("\n🔧 Reproducibility:")
    print(f"   Random Seed: {config['random_seed']}")
    print(f"   Index Version: {config['index_version']}")
    
    print("\n🤖 Models:")
    print(f"   Embedding: {config['embedding_model']}")
    print(f"   Reranker: {config['reranker_model']}")
    print(f"   Generation: {config['generation_model']}")
    print(f"   Temperature: {config['generation_temperature']}")
    
    print("\n📦 Chunking:")
    print(f"   Strategy: {config['chunking_strategy']}")
    print(f"   Chunk Size: {config['chunk_size']}")
    print(f"   Overlap: {config['chunk_overlap']}")
    
    print("\n🔍 Retrieval:")
    print(f"   Prefetch Limit: {config['prefetch_limit']}")
    print(f"   Rerank Candidates: {config['rerank_limit']}")
    print(f"   Final Results: {config['final_limit']}")
    print(f"   Min Relevance: {config['min_relevance_score']}")
    
    print("\n📚 Citations:")
    print(f"   Strict Mode: {config['strict_citation_mode']}")
    print(f"   Enable Citations: {config['enable_citations']}")
    
    print("\n📅 Staleness:")
    print(f"   Stale Threshold: {config['stale_threshold_years']} years")
    print(f"   Potentially Stale: {config['potentially_stale_years']} years")
    
    print("\n📄 Documents:")
    for filename, url in config['pdf_source_urls'].items():
        pub_date = config['pdf_publication_dates'].get(filename, 'Unknown')
        print(f"   • {filename}: {pub_date}")


def reindex_documents():
    """Clear and reindex all documents."""
    print("\n" + "=" * 60)
    print("🔄 REINDEXING DOCUMENTS")
    print("=" * 60)
    
    # Import clear_index functionality
    from clear_index import clear_all
    
    print("\n🗑️  Clearing existing index...")
    clear_all()
    
    print("\n📥 Processing documents...")
    results = process_all_pdfs_in_folder(Config.UPLOAD_FOLDER)
    
    print(f"\n✅ Reindexing complete:")
    print(f"   Total files: {results.get('total_files', 0)}")
    print(f"   Newly processed: {results.get('newly_processed', 0)}")
    print(f"   Total chunks: {results.get('total_chunks', 0)}")
    
    return results


def run_evaluation(limit: int = None, save_path: str = None):
    """Run the full evaluation pipeline."""
    print("\n" + "=" * 60)
    print("🧪 RUNNING EVALUATION")
    print("=" * 60)
    
    # Generate timestamped filename if not provided
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"reproduce_results_{timestamp}.json"
    
    # Import evaluation module
    from eval import run_evaluation as eval_run
    
    # Run evaluation
    print(f"\n🚀 Starting evaluation...")
    eval_run(limit=limit, save_path=save_path.replace('.json', '.csv'))
    
    return save_path


def save_reproducibility_report(output_path: str = None):
    """Save a complete reproducibility report."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reproducibility_report_{timestamp}.json"
    
    report = {
        "system_info": get_system_info(),
        "config": get_config_snapshot(),
        "index_state": get_index_snapshot(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Reproducibility report saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Reproducibility script for NUST Policy RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reproduce.py                    # Show config + run evaluation
  python reproduce.py --reindex          # Reindex documents only
  python reproduce.py --full             # Full pipeline: reindex + evaluate
  python reproduce.py --config-only      # Just show configuration
  python reproduce.py --limit 5          # Run evaluation on 5 questions only
        """
    )
    
    parser.add_argument('--reindex', action='store_true',
                        help='Clear index and reprocess all documents')
    parser.add_argument('--full', action='store_true',
                        help='Full pipeline: reindex + evaluate')
    parser.add_argument('--config-only', action='store_true',
                        help='Only display configuration, no evaluation')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of evaluation questions')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔬 NUST Policy RAG - Reproducibility Script")
    print("=" * 60)
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print(f"   Seed: {SEED}")
    print(f"   Index Version: {Config.INDEX_VERSION}")
    
    # Always display config
    display_config()
    
    if args.config_only:
        # Save report and exit
        save_reproducibility_report(args.output)
        return
    
    if args.reindex or args.full:
        reindex_documents()
    
    if not args.reindex or args.full:
        # Run evaluation unless only reindexing
        run_evaluation(limit=args.limit, save_path=args.output)
    
    # Always save reproducibility report
    save_reproducibility_report()
    
    print("\n" + "=" * 60)
    print("✅ REPRODUCIBILITY SCRIPT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

