# Compost GI Streamlit Predictor

This repository contains a Streamlit web app for predicting compost germination index, GI (%), using the paper's XGBoost/P1 feature protocol.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=https://github.com/17609858895/Compost-GI-Streamlit-Predictor&branch=main&mainModule=app.py)

## App Features

- Single-sample GI prediction from feedstock, process, and chemistry inputs.
- Batch CSV/XLSX prediction with downloadable CSV and Excel outputs.
- Template and example input files.
- Validation metrics and applicability-domain warnings.
- Model bundle exported as a fitted scikit-learn Pipeline with XGBoost.

## Model Scope

The model estimates `GI (%)` for compost maturity screening. Random-split metrics describe within-distribution interpolation, while group-holdout metrics are the conservative robustness estimate. Predictions outside the recorded training range should be interpreted cautiously.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud Deployment

Open the badge above or use these settings:

- Repository: `17609858895/Compost-GI-Streamlit-Predictor`
- Branch: `main`
- Main file path: `app.py`
- Python version: `3.12`

If Streamlit Cloud defaults to a newer Python version, open the app dashboard, go to Settings -> Advanced settings, set Python to 3.12, then reboot/redeploy.

## Re-export Model

From this repository folder:

```bash
python scripts/train_export_model.py --data "../Compost dateset.xlsx"
```

The script writes:

- `model/model_bundle.joblib`
- `model/model_metadata.json`
- `model/validation_metrics.csv`
- `data/example_input.csv`
- `data/single_prediction_template.csv`
