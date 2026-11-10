"""
Standalone Model C embedding script for GLARE.

This script generates the ArabicSBERT embeddings using the SentenceTransformer
checkpoint configured for Model C. It follows the same cache naming and review
text hashing convention as glare_issues/embed.py so the pipeline can identify
the generated embeddings.

Run from the repository root:
    python glare_issues/embed_model_c.py \
        --input /content/drive/MyDrive/GLARE_Project/results/02_problem_reviews.csv \
        --output-dir /content/drive/MyDrive/GLARE_Project/cache \
        --text-col content

Requirements:
    pip install pandas numpy sentence-transformers
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODEL_KEY = "ArabicSBERT"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each embedding row, safely handling zero vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GLARE Model C (ArabicSBERT) embeddings."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the filtered reviews CSV (for example, 02_problem_reviews.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="cache",
        help="Directory in which to save the embedding cache and metadata.",
    )
    parser.add_argument(
        "--text-col",
        default="content",
        help="CSV column containing review text.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SentenceTransformer checkpoint to use for Model C.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and replace the cache file if it already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.max_seq_length < 1:
        raise ValueError("--max-seq-length must be at least 1.")

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if args.text_col not in df.columns:
        raise ValueError(
            f"Column '{args.text_col}' not found in {input_path}. "
            f"Available columns: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError(f"The input CSV contains no reviews: {input_path}")

    # Keep the exact CSV row order; downstream labels/reviews must align with it.
    texts = df[args.text_col].fillna("").astype(str).tolist()

    # Match glare_issues/embed.py exactly: hash texts joined by real newline chars.
    digest = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:10]

    output_path = output_dir / f"emb_{MODEL_KEY}_{digest}.npy"
    metadata_path = output_dir / f"emb_{MODEL_KEY}_{digest}.json"

    if output_path.exists() and not args.force:
        print(f"Embedding cache already exists: {output_path}")
        print("Not overwriting it. Use --force only if replacement is intentional.")
        return

    from sentence_transformers import SentenceTransformer

    print(f"Loading checkpoint: {args.model}")
    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_length

    print(f"Generating embeddings for {len(texts):,} reviews...")
    vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError(
            "Unexpected embedding shape. "
            f"Expected {len(texts)} rows, received {vectors.shape}."
        )

    vectors = l2_normalize(vectors)
    np.save(output_path, vectors)

    metadata = {
        "model_key": MODEL_KEY,
        "checkpoint": args.model,
        "input_csv": str(input_path),
        "text_column": args.text_col,
        "review_count": len(texts),
        "embedding_shape": list(vectors.shape),
        "dtype": str(vectors.dtype),
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "review_text_hash": digest,
        "normalization": "L2",
        "row_order": "Same as input CSV; preserve this order when using the vectors.",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Embeddings: {output_path}")
    print(f"Metadata:   {metadata_path}")
    print(f"Shape:      {vectors.shape}")
    print(f"Data type:  {vectors.dtype}")


if __name__ == "__main__":
    main()
