# Scoop Forecast — Assignment 3

A Streamlit app that loads a model trained and saved in `train_model.ipynb` to predict daily ice cream sales (scoops/day).

## Assignment checklist

1. **Add a predictor:** `is_weekend` (Saturday/Sunday = 1; weekdays = 0) supplements `temperature_f`.
2. **Retrain and save:** The executed notebook compares temperature-only and updated linear regressions on the same chronological 80/20 split, then refits both on all data and saves `models/ice_cream_model.joblib` with feature order and metadata.
3. **Update the app:** `app.py` loads the saved artifact with `joblib.load` and calls `.predict`; it never calls `.fit`. Both predictors have user controls.
4. **Deploy:** Streamlit Community Cloud; branch `main`, entrypoint `app.py`, Python 3.12. Public URL is listed in the repository description/Streamlit dashboard when deployed.

## Data provenance

No class starter notebook or dataset was supplied. This is a self-contained classroom example using **730 simulated daily observations**, not actual sales. The deterministic generator lives in the notebook (seed 42). It assumes a positive temperature relationship and an additional weekend effect, then adds noise. Model performance measures how well the fitted model recovers this simulated pattern, not performance for a real shop. The added feature's apparent benefit is deliberately built into the simulation. Replace the data with observed temperature, weekend status and sales before making real business decisions.

## Reproduce locally

Use Python 3.12:

```bash
python -m pip install -r requirements-notebook.txt
```

Open `train_model.ipynb` in Jupyter or VS Code, select that Python environment, and run all cells. Alternatively:

```bash
python execute_notebook.py
python -m streamlit run app.py
```

The notebook regenerates the CSV and model, records MAE/RMSE/R² without fitting on held-out targets, refits for serving, and confirms that a reloaded model returns the same prediction. Dependencies are pinned to the versions used to train and test the artifact. Only load this trusted bundled joblib artifact; no user-supplied serialized models are accepted.

`execute_notebook.py` executes the notebook's real Python cells in order and captures their stdout and final expressions, including DataFrame tables. It supports running this notebook in environments where socket-based Jupyter kernels are unavailable. You can also run the notebook normally in your Jupyter environment.

## Files

- `train_model.ipynb`: executed training, evaluation and export notebook.
- `data/ice_cream_sales.csv`: generated data with dates, both predictors, and sales.
- `models/ice_cream_model.joblib`: trained updated model and baseline with feature names.
- `models/metadata.json`: provenance, split, metrics, coefficients, and package versions.
- `app.py`: prediction controls, comparison chart, metrics, and artifact downloads.
- `test_app.py`: checks saved-model predictions and interactive app behavior.

No API key or secrets are required.
