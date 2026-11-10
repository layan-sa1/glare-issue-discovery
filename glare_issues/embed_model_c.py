"""
Standalone Model C embedding script for GLARE.

Default checkpoint:
    sentence-transformers/paraphrase-multilingual-mpnet-base-v2

Run from the repository root:
    python embed_model_c.py --input /content/drive/MyDrive/GLARE_Project/results/02_problem_reviews.csv --output-dir /content/drive/MyDrive/GLARE_Project/cache --text-col content

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


def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def main():
    parser = argparse.ArgumentParser(description="Generate GLARE Model C embeddings.")
    parser.add_argument("--input", required=True, help="Filtered reviews CSV.")
    parser.add_argument("--output-dir", default="cache", help="Output cache folder.")
    parser.add_argument("--text-col", default="content", help="Review text column.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer checkpoint.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--force", action="store_true", help="Replace an existing cache file.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if args.text_col not in df.columns:
        raise ValueError(
            f"Column '{args.text_col}' not found. Available columns: {list(df.columns)}"
        )

    # Preserve CSV row order so vectors stay aligned with the reviews.
    texts = df[args.text_col].fillna("").astype(str).tolist()
    if not texts:
        raise ValueError("The input CSV contains no reviews.")

    digest = hashlib.md5("\\n".join(texts).encode("utf-8")).hexdigest()[:10]
    output_path = output_dir / f"emb_ArabicSBERT_{digest}.npy"
    metadata_path = output_dir / f"emb_ArabicSBERT_{digest}.json"

    if output_path.exists() and not args.force:
        print(f"Cache already exists; not overwriting: {output_path}")
        print("Use --force only if you intentionally want to replace it.")
        return

    from sentence_transformers import SentenceTransformer

    print(f"Loading checkpoint: {args.model}")
    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_length

    print(f"Embedding {len(texts):,} reviews...")
    vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
    np.save(output_path, vectors)

    metadata = {
        "model_key": "ArabicSBERT",
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
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\\nDone.")
    print(f"Embeddings: {output_path}")
    print(f"Metadata:   {metadata_path}")
    print(f"Shape:      {vectors.shape}")
    print(f"Data type:  {vectors.dtype}")


if __name__ == "__main__":
    main()
