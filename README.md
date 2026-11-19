# Discovering user issues from Arabic app reviews (GLARE)

Pipeline: preprocessing -> Stage 1 LLM problem filter -> Stage 2a embeddings (AraBERT, MARBERT, multilingual sentence-transformer)
-> UMAP -> Stage 2b HDBSCAN (main) and K-Means (baseline) -> Stage 3 LLM issue summaries -> evaluation.

## Install
    pip install -r requirements.txt

## Data
Download the engineered reviews file of GLARE (Zenodo DOI 10.5281/zenodo.6457824) and save a CSV (or a chunk/sample) with the
columns `review_id, app_id, content, rating` (names can be changed in `glare_issues/config.py`).
The 30,000-review sample used in the experiments and the annotation files are available in the project Google Drive folder: https://drive.google.com/drive/folders/1wKBiPZxCv-642UH5_Ml4CAba_7ghFgT_

## Repository Structure

```
├── glare_issues/                 # pipeline code (preprocess, embed, cluster, llm, evaluate)
├── notebooks/
│   └── GLARE_Methodology_v2.ipynb    # runs the whole pipeline stage by stage in Colab
├── LLM_Code/                     # Stage 1 and Stage 3 LLM notebook (Member 5)
├── tests/                        # unit tests (run with the demo data)
├── run_pipeline.py               # command-line entry point
└── requirements.txt
```

## Smoke test (no data, no API key, no model download)
    python run_pipeline.py --demo
    python -m pytest tests
The demo uses synthetic reviews, a keyword-rule mock LLM and a TF-IDF stand-in for the embeddings. Its numbers are NOT research results.

## Real run
    export GEMINI_API_KEY=...        # or ANTHROPIC_API_KEY / OPENAI_API_KEY with --llm anthropic / openai
    python run_pipeline.py --input data/glare_sample.csv --llm gemini --model gemini-3.1-flash-lite --target 30000 --trials 30
    
To run `notebooks/GLARE_Methodology_v2.ipynb` in Google Colab, add your Gemini key in Colab Secrets (key icon) with the name `GEMINI_API_KEY`, and turn on Notebook access.

Before running, check in `glare_issues/config.py` that the three embedding checkpoints are available: AraBERT (aubmindlab/bert-base-arabertv02), MARBERT (UBC-NLP/MARBERT), and the multilingual sentence-transformer (sentence-transformers/paraphrase-multilingual-mpnet-base-v2).

## Results

| Part | Result |
|---|---|
| Problem filter (Gemini) | Precision 0.755, Recall 0.959, F1 0.845, kappa 0.805 (400 reviews) |
| Best clustering (K-Means, Model C) | ARI 0.088, NMI 0.322 |
| Issue summaries (6 raters, 42 clusters) | Fluency 4.76, Coherence 4.73, Consistency 4.35, Relevance 3.82 |

## Human parts of the evaluation
1. `results/annotations/filter_sheet.csv`: fill `human_problem` (1/0) and save as `filter_sheet_done.csv`.
2. `results/annotations/grouping_sheet.csv`: write the same `issue_group` name for reviews that describe the same issue; save as `grouping_sheet_done.csv`.
3. `results/annotations/summary_rating_sheet.csv`: one row per cluster per rater, ratings 1-5; save as `summary_rating_sheet_done.csv`.
4. `python run_pipeline.py --input ... --stage all` again (the LLM answers are cached, so it is cheap) to produce `05_evaluation_report.json` and the figures.

## Outputs (results/)
The result files are not stored in this repository. They are included in the project Google Drive folder (see Data).
`01_preprocessed.csv`, `02_problem_reviews.csv`, `03_config_comparison.csv` (6 configurations), `03_trials_*.csv` (all tuning trials),
`04_ranked_issues.csv` (labels, Arabic summaries, top terms), `05_evaluation_report.json`, `05_external_scores.csv`, `fig_*.png`.

## Design choices worth knowing
* Tuning uses only unsupervised criteria (relative validity, silhouette). The human-grouped sample is kept for the final test only.
* K-Means is run on the same UMAP space as the tuned HDBSCAN run, so only the clustering algorithm differs.
* ARI/NMI are reported two ways: noise as one extra group (`_all`) and only on clustered reviews (`_clustered`) with coverage.
* LLM calls use temperature 0 and are cached on disk (`cache/`), so reruns are reproducible.
