Food Waste Predictor

A small Flask web app that predicts household weekly food waste (kg/week) using a Random Forest model trained on a dataset (Kaggle or synthetic fallback). The app also renders Plotly charts (two core charts + optional 10-year trend charts) and provides a simple login flow for demo purposes.

---

What’s included

- `finalapp.py` — main Flask application and ML pipeline (training, prediction, charts).
- `templates/` — Jinja templates (expected `index.html` and `login.html`).
- `data/` — (optional) directory for Kaggle dataset download and local CSV storage.

Screenshots attached to this README (see file names below):

- ![Screenshot (360)](https://github.com/user-attachments/assets/b6220eae-8041-4c92-962d-3be36a1c6de0)

— Login screen (desktop).
- <img width="1920" height="1080" alt="Screenshot (361)" src="https://github.com/user-attachments/assets/bb159b64-0d79-4e11-ac20-8a754c67ded3" />
 — Main form (initial state).
- <img width="1920" height="1080" alt="Screenshot (362)" src="https://github.com/user-attachments/assets/3be9b425-0786-4704-8ec1-5fe5436823e9" />
 — Prediction result with KPIs and charts.
- <img width="1920" height="1080" alt="Screenshot (363)" src="https://github.com/user-attachments/assets/7f31b309-dc00-46b3-8419-fbd6ba106284" />
 — Full page after prediction (shows charts and KPIs).

> These screenshots were provided by the developer to illustrate the UI and demo flow.

---

Features

- Auto-download Kaggle dataset (optional) using `KAGGLE_DATASET_SLUG` env var.
- Schema guard: attempts to adapt external datasets to expected schema and falls back to a synthetic generator when necessary.
- Train-on-load RandomForest model with preprocessing pipeline (one-hot encoding + numeric passthrough).
- Plotly charts exported as JSON and injected into templates:
  - `chart1` — Average food waste by meal type (bar).
  - `chart2` — Average food waste by cultural cuisine (bar).
  - `chart3` / `chart4` — Optional 10-year line charts built from the dataset (`Year` column is attached if not present).
- Simple login/logout (demo credentials in code: `Akhil` / `123456`).

---

Requirements

- Python 3.10+ (or 3.8+ should work)
- pip packages (install using `requirements.txt` or pip):

```bash
pip install flask pandas numpy scikit-learn plotly
optional: kaggle
pip install kaggle
```

If you need a `requirements.txt`, you can create one with:

```text
Flask
pandas
numpy
scikit-learn
plotly
kaggle    # optional
```

---

How to run (development)

1. (Optional) set up Kaggle credentials if you want the app to download the dataset automatically:
   - Place your `kaggle.json` under `~/.kaggle/kaggle.json` or set `%KAGGLE_CONFIG_DIR%`.
   - Set `KAGGLE_DATASET_SLUG` environment variable if you want a dataset other than the default (`joebeachcapital/food-waste`).

2. Start the app:

```bash
python finalapp.py
```

The app will open `http://127.0.0.1:5000/login` in your browser automatically (development mode). Login using demo credentials or implement your own authentication.

---

Configuration options

- `CSV_FILENAME` — expected local filename the app prefers when found.
- `KAGGLE_DATASET_SLUG` — environment variable to override the default Kaggle dataset slug.
- `app.secret_key` — currently a hard-coded key for the demo. Replace with a secure randomly-generated secret for production.

---

Data handling / schema compatibility

- The app expects the dataset to contain the feature columns listed in `FEATURE_COLUMNS` and the target `Weekly_Food_Waste_kg`.
- If the dataset is missing the target or most features, the code falls back to a synthetic dataset generator (see `generate_synthetic_dataset`).
- If some features are missing but the target exists, the app will attempt to fill missing features with safe defaults via `ensure_expected_schema`.

---

Plotly charts

Charts are generated on the server and serialized as JSON (via `plotly.utils.PlotlyJSONEncoder`) and passed into Jinja templates as `chart1`, `chart2`, `chart3`, `chart4`.

If you want to remove the 10-year charts (as in an earlier edit), remove the `chart3` / `chart4` blocks from the template and stop sending them from the Flask routes.

---

Security / production notes

- This project is demo-level. Do **not** use the demo credentials or `app.secret_key` in production.
- Set `debug=False` when deploying.
- Consider persisting trained models to disk (joblib/pickle) instead of retraining on every start.
- Add CSRF protection and secure authentication for real deployments.

---

License & credits

MIT-style usage permitted. App developed by Akhil (see footer in UI). Screenshots supplied by developer and used for documentation.

---

Contact / Next steps

If you want any of the following, I can update the README or code:

- Add a `requirements.txt` and a `Procfile` for Heroku/Gunicorn deployment.
- Persist the trained model to `models/model.joblib` and load it on startup.
- Convert Plotly charts to client-side Plotly code (send raw data only) to reduce server JSON size.
- Replace demo login with a proper Flask-Login integration.

---

*README generated on request. Screenshots are included in the repository under `/mnt/data/` as provided.*



