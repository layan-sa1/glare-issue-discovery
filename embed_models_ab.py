# embed_models_ab.py (Models A and B: AraBERT and MARBERT)
# Member 2. Standalone script; does not change glare_issues/embed.py.
#
# What this does:
#   1. Reads the filtered reviews (results/02_problem_reviews.csv)
#   2. Turns every review into a vector with a pre-trained model (NO training)
#   3. Uses mean pooling, then L2 normalization
#   4. Saves one .npy file per model, in the same row order as the CSV

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# ---------------- Settings ----------------
DEMO = True            # True = only 200 reviews (quick test). Set False for the real run.
DEMO_SIZE = 200
INPUT_FILE = "results/02_problem_reviews.csv"
OUTPUT_DIR = "results"
BATCH_SIZE = 64
MAX_LENGTH = 128       # app reviews are short, so 128 tokens is enough

MODELS = {
    "arabert": "aubmindlab/bert-base-arabertv02",   # Model A
    "marbert": "UBC-NLP/MARBERT",                   # Model B
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def mean_pooling(last_hidden_state, attention_mask):
    # Average the token vectors, but ignore the padding tokens
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(texts, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()  # we only use the model, we do not train it

    all_vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded)
        vectors = mean_pooling(output.last_hidden_state, encoded["attention_mask"])
        # L2 normalization: every vector gets length 1
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        all_vectors.append(vectors.cpu().numpy())
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  {start}/{len(texts)} reviews done")

    return np.vstack(all_vectors)


def main():
    df = pd.read_csv(INPUT_FILE)
    if DEMO:
        df = df.head(DEMO_SIZE)
    texts = df["text"].astype(str).tolist()
    print("Number of reviews:", len(texts))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for short_name, model_name in MODELS.items():
        print("Embedding with", model_name)
        vectors = embed_texts(texts, model_name)
        suffix = "_demo" if DEMO else ""
        path = f"{OUTPUT_DIR}/embeddings_{short_name}{suffix}.npy"
        np.save(path, vectors)
        print("Saved", path, "shape:", vectors.shape)


if __name__ == "__main__":
    main()
