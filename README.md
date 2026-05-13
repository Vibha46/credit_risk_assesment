# Credit Risk Assessment

An interactive ML-powered web app for credit risk evaluation using the German Credit Dataset. Built with Streamlit and scikit-learn.

## Features

- **Dashboard** — dataset overview with risk distribution, credit amount breakdowns, and a model comparison table
- **Model Performance** — confusion matrix, ROC curves, and metric breakdowns for all four classifiers
- **Feature Analysis** — feature importance, permutation importance, correlation heatmap, and scatter plots
- **Predict Risk** — real-time risk prediction with a gauge chart and similar applicant lookup
- **Data Explorer** — filterable, downloadable view of the full dataset

## Models

| Model | Notes |
|---|---|
| Random Forest | 200 estimators |
| Gradient Boosting | 150 estimators |
| Logistic Regression | Scaled input, max 500 iterations |
| SVM | Scaled input, probability calibration enabled |

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone https://github.com/your-username/credit-risk-app.git
cd credit-risk-app
pip install -r requirements.txt
```

### Dataset

Place the dataset file in the project root and name it exactly:

```
german_credit_data.csv
```

The file must have these columns: `Age`, `Sex`, `Job`, `Housing`, `Saving accounts`, `Checking account`, `Credit amount`, `Duration`, `Purpose`.

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Upload `german_credit_data.csv` via the **Secrets / Files** section or commit it to the repo
5. Click **Deploy**

## Project Structure

```
credit-risk-app/
├── app.py                  # Main Streamlit application
├── preprocessing.py        # Standalone data preprocessing script
├── german_credit_data.csv  # Dataset (add manually, not tracked by git)
├── requirements.txt
├── .gitignore
└── README.md
```

## Data Preprocessing

`preprocessing.py` is a standalone script that encodes the raw CSV into a one-hot-encoded integer DataFrame. Run it independently:

```bash
python preprocessing.py
```

The live app handles its own preprocessing internally via `load_and_prepare_data` and `encode_features` in `app.py`.

## License

MIT
