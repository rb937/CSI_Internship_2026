# How to Run

This project has two ways to work through it: the **notebook** (exploratory,
one-shot) or the **scripts** (reusable, repeatable). Both produce the same
artifacts in `data/`, so you only need to run one path before using the dashboard.

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib streamlit
```
or

```bash
pip install -r requirements.txt # in the base directory (It has all the requirements libraries and additional ones based on other weeks task
```

Place the raw dataset at:
```
dataset/LeadScoring.csv
```
(Download from [Kaggle](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset).)

## Option A: Notebook (recommended first run)

Run:
```
notebook/eda_and_modeling.ipynb
```
This walks through cleaning, EDA, both models, evaluation, and lead scoring —
and saves the trained models + processed data to `data/` at the end. Best for
understanding *why* the pipeline is built this way, not just running it.

## Option B: Scripts (for re-running training later)

Run in this order, from inside `src/`:

1. **`train.py`** — cleans the data, trains both models, saves everything to `data/`
   ```bash
   cd src
   python train.py
   ```
   Produces: `logistic_model.pkl`, `xgboost_model.pkl`, `scaler.pkl`,
   `feature_columns.pkl`, `numeric_columns.pkl`, `processed_leads.csv`

2. **`score.py`** — scores a new CSV of leads using the trained model
   ```bash
   python score.py --input path_to/new_leads.csv --output path_to/scored_leads.csv
   ```
   Requires `train.py` to have been run at least once first (it loads the saved
   model artifacts from `data/`).

`preprocessing.py` isn't run directly — it's shared logic imported by both
`train.py` and `score.py` so cleaning stays identical between training and scoring.

## Option C: Dashboard

Once `data/` has the trained model artifacts (from either Option A or B):
```bash
streamlit run app.py
```
Upload a CSV of new leads (same columns as the raw training data, minus
`Converted`) and get back a ranked, scored, downloadable list.

## Order summary

```
1. Get datasets/LeadScoring.csv in place
2. Run the notebook OR train.py  →  populates data/
3. (optional) score.py for one-off batch scoring from the command line
4. app.py for the interactive dashboard
```
