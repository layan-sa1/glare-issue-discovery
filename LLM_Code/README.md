# Member 5 – LLM Problem Filtering

This notebook implements Stage 1: Problem Filtering using Google Gemini.

It classifies 30,000 preprocessed Arabic app reviews into two categories:

- **1 – Problem:** The review reports a problem.
- **0 – Not Problem:** The review does not report a problem.

### Model
Google Gemini (`gemini-3.1-flash-lite`)

### Input
`01_preprocessed.csv` – 30,000 Arabic app reviews.

### Output
`02_problem_reviews.csv` – Reviews classified as Problem (1).

### How It Works
1. Loads the preprocessed dataset.
2. Uses Gemini LLM to classify reviews.
3. Processes reviews in batches of 20.
4. Saves classification progress to support resuming interrupted runs.
5. Exports reviews classified as Problem.

### Requirements
- Google Colab
- Gemini API key configured in Colab Secrets as `GEMINI_API_KEY`
- Preprocessed dataset (`01_preprocessed.csv`)
