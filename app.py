# finalapp.py
from flask import Flask, render_template, request, redirect, url_for, session
import os
import pandas as pd
import numpy as np
import json
import webbrowser
import threading
import datetime as dt
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  # (kept if you want to extend later)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import plotly
import plotly.graph_objs as go

# Optional: Kaggle (import safely so app works even without it)
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception:
    KaggleApi = None

app = Flask(__name__, template_folder="templates")
app.secret_key = "supersecretkey"

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
CSV_FILENAME = "food_waste_dataset.csv"
KAGGLE_DATASET_SLUG = os.getenv("KAGGLE_DATASET_SLUG", "joebeachcapital/food-waste")

FEATURE_COLUMNS = [
    "Household_Size",
    "Daily_Meal_Count",
    "Meal_Type",
    "Shopping_Habit",
    "Storage_Availability",
    "Awareness_of_Waste_Management",
    "Leftovers_Frequency",
    "Income_Range",
    "Cooking_Preference",
    "Cultural_Cuisine_Preference",
    "Perishability_Awareness",
    "Seasonal_Variation"
]
TARGET = "Weekly_Food_Waste_kg"

# ----------------------------------------------------------------
# Kaggle Helper
# ----------------------------------------------------------------
def download_kaggle_dataset(slug: str, dest_dir: str = "data") -> Path:
    """
    Download & unzip a Kaggle dataset to dest_dir if not present.
    Returns Path to a CSV file inside dest_dir (prefers CSV_FILENAME if found).
    """
    data_dir = Path(dest_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if KaggleApi is None:
        raise RuntimeError("Kaggle package not available. Please install it: pip install kaggle")

    api = KaggleApi()
    api.authenticate()  # uses ~/.kaggle/kaggle.json or %KAGGLE_CONFIG_DIR%

    # Only download if no CSVs yet (keeps runs fast)
    if not any(data_dir.glob("*.csv")):
        print(f"📥 Downloading dataset {slug} ...")
        api.dataset_download_files(slug, path=str(data_dir), unzip=True)

    # Prefer expected filename if present
    preferred = data_dir / CSV_FILENAME
    if preferred.exists():
        return preferred

    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {data_dir} after Kaggle download.")
    return csvs[0]

# ----------------------------------------------------------------
# Synthetic Data Generator
# ----------------------------------------------------------------
def generate_synthetic_dataset(n=1200, filename=CSV_FILENAME, random_state=42):
    rng = np.random.default_rng(random_state)

    meal_types = ["Vegetarian", "Non-Vegetarian", "Mixed"]
    shopping_habits = ["Daily", "Weekly", "Bulk Buying"]
    storage = ["Small", "Medium", "Large"]
    awareness = ["Low", "Medium", "High"]
    leftovers = ["Never", "Sometimes", "Often"]
    income = ["Low", "Medium", "High"]
    cooking = ["Fresh Daily", "Batch Cooking", "Reheated"]
    cuisines = ["Indian", "Chinese", "Italian", "North Indian", "South Indian", "Fusion", "Street Food"]
    perishability = ["Not aware", "Somewhat aware", "Aware"]
    seasonal = ["All year", "Summer", "Winter", "Monsoon"]

    rows = []
    for _ in range(n):
        hs = int(rng.integers(1, 7))
        dmc = int(rng.integers(1, 5))
        mt = rng.choice(meal_types)
        sh = rng.choice(shopping_habits, p=[0.35, 0.45, 0.2])
        st = rng.choice(storage, p=[0.3, 0.5, 0.2])
        aw = rng.choice(awareness, p=[0.3, 0.5, 0.2])
        lf = rng.choice(leftovers, p=[0.2, 0.5, 0.3])
        inc = rng.choice(income, p=[0.4, 0.4, 0.2])
        cp = rng.choice(cooking, p=[0.45, 0.35, 0.2])
        cc = rng.choice(cuisines)
        pe = rng.choice(perishability, p=[0.5, 0.35, 0.15])
        sv = rng.choice(seasonal)

        base = 0.2 * hs + 0.5 * dmc
        if mt == "Non-Vegetarian": base += 0.5
        if sh == "Bulk Buying": base += 0.8
        if st == "Small": base += 0.3
        if aw == "High": base -= 0.7
        if lf == "Often": base += 1.0
        if inc == "High": base += 0.3
        if cp == "Fresh Daily": base -= 0.2
        if pe == "Not aware": base += 0.5
        if cc == "Street Food": base += 0.7

        noise = rng.normal(0, 0.4)
        weekly_waste = max(0.1, round(base + noise, 2))

        rows.append({
            "Household_Size": hs,
            "Daily_Meal_Count": dmc,
            "Meal_Type": mt,
            "Shopping_Habit": sh,
            "Storage_Availability": st,
            "Awareness_of_Waste_Management": aw,
            "Leftovers_Frequency": lf,
            "Income_Range": inc,
            "Cooking_Preference": cp,
            "Cultural_Cuisine_Preference": cc,
            "Perishability_Awareness": pe,
            "Seasonal_Variation": sv,
            "Weekly_Food_Waste_kg": weekly_waste
        })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    return df

# ----------------------------------------------------------------
# Schema Guard / Adapter
# ----------------------------------------------------------------
def ensure_expected_schema(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    If df has all needed columns, returns df.
    If some features are missing but TARGET exists, fills missing features with safe defaults.
    If TARGET missing or most features missing, returns None (caller should fallback to synthetic).
    """
    cols = set(df.columns.astype(str))

    # If target missing, we cannot train -> force fallback
    if TARGET not in cols:
        print(f"⚠️ TARGET '{TARGET}' missing in loaded data. Falling back to synthetic.")
        return None

    # Count how many of our expected features exist
    present = [c for c in FEATURE_COLUMNS if c in cols]
    missing = [c for c in FEATURE_COLUMNS if c not in cols]

    # If less than half the features are present, schema is incompatible -> fallback
    if len(present) < len(FEATURE_COLUMNS) // 2:
        print(f"⚠️ Too many missing features ({len(missing)}/{len(FEATURE_COLUMNS)}). Falling back to synthetic.")
        return None

    # Fill missing with safe defaults
    filled = df.copy()
    defaults_num = {"Household_Size": 2, "Daily_Meal_Count": 3}
    defaults_cat = {
        "Meal_Type": "Mixed",
        "Shopping_Habit": "Weekly",
        "Storage_Availability": "Medium",
        "Awareness_of_Waste_Management": "Medium",
        "Leftovers_Frequency": "Sometimes",
        "Income_Range": "Medium",
        "Cooking_Preference": "Fresh Daily",
        "Cultural_Cuisine_Preference": "Indian",
        "Perishability_Awareness": "Somewhat aware",
        "Seasonal_Variation": "All year",
    }

    for m in missing:
        if m in defaults_num:
            filled[m] = defaults_num[m]
        else:
            filled[m] = defaults_cat.get(m, "Unknown")

    # Keep only expected columns + target to avoid stray columns breaking the pipeline
    keep_cols = FEATURE_COLUMNS + [TARGET]
    return filled[keep_cols]

# ----------------------------------------------------------------
# Dataset Loading (with Kaggle + schema guard + fallback)
# ----------------------------------------------------------------
df = None
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

try:
    if KaggleApi is not None and KAGGLE_DATASET_SLUG:
        csv_path = download_kaggle_dataset(KAGGLE_DATASET_SLUG, dest_dir=str(data_dir))
        print(f"✅ Loaded Kaggle CSV: {csv_path}")
        raw_df = pd.read_csv(csv_path)

        adapted = ensure_expected_schema(raw_df)
        if adapted is None:
            print("ℹ️ Kaggle data not compatible with expected schema.")
        else:
            df = adapted
except Exception as e:
    print(f"[⚠️ Kaggle fetch skipped] {e}")

if df is None:
    if os.path.exists(CSV_FILENAME):
        print(f"📂 Using local CSV: {CSV_FILENAME}")
        local_df = pd.read_csv(CSV_FILENAME)
        adapted = ensure_expected_schema(local_df)
        if adapted is None:
            print("ℹ️ Local CSV not compatible; generating synthetic instead.")
            df = generate_synthetic_dataset(n=1200)
        else:
            df = adapted
    else:
        print(f"🧪 {CSV_FILENAME} not found. Generating synthetic dataset...")
        df = generate_synthetic_dataset(n=1200)

# ----------------------------------------------------------------
# ML Pipeline
# ----------------------------------------------------------------
categorical_cols = [
    "Meal_Type", "Shopping_Habit", "Storage_Availability",
    "Awareness_of_Waste_Management", "Leftovers_Frequency", "Income_Range",
    "Cooking_Preference", "Cultural_Cuisine_Preference",
    "Perishability_Awareness", "Seasonal_Variation"
]
numeric_cols = ["Household_Size", "Daily_Meal_Count"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

pipeline = Pipeline(steps=[
    ("pre", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

X = df[FEATURE_COLUMNS]
y = df[TARGET]
pipeline.fit(X, y)

# ----------------------------------------------------------------
# Visualization Helpers
# ----------------------------------------------------------------
def make_plotly_bar(x, y, title, x_title="", y_title=""):
    bar = go.Bar(x=x, y=y)
    layout = go.Layout(
        title=title,
        plot_bgcolor="#0f1724",
        paper_bgcolor="#0f1724",
        font=dict(color="#e6eef8"),
        xaxis=dict(title=x_title, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title=y_title, gridcolor="rgba(255,255,255,0.08)"),
        margin=dict(l=50, r=20, t=60, b=50),
    )
    fig = go.Figure(data=[bar], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_plotly_line(x, y, title, x_title="", y_title=""):
    line = go.Scatter(x=x, y=y, mode="lines+markers")
    layout = go.Layout(
        title=title,
        plot_bgcolor="#0f1724",
        paper_bgcolor="#0f1724",
        font=dict(color="#e6eef8"),
        xaxis=dict(title=x_title, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title=y_title, gridcolor="rgba(255,255,255,0.08)"),
        margin=dict(l=50, r=20, t=60, b=50),
    )
    fig = go.Figure(data=[line], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# ----------------------------------------------------------------
# Time dimension helper for 10-year charts
# ----------------------------------------------------------------
def attach_years(df_in: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Ensure a 'Year' column exists by deterministically distributing rows
    across the last 10 calendar years (inclusive of current year).
    """
    df_out = df_in.copy()
    if "Year" in df_out.columns:
        return df_out

    current_year = dt.date.today().year
    years = list(range(current_year - 9, current_year + 1))
    rng = np.random.default_rng(random_state)

    n = len(df_out)
    repeats, remainder = divmod(n, len(years))
    assigned = years * repeats + years[:remainder]
    rng.shuffle(assigned)  # deterministic with seed
    df_out["Year"] = assigned
    return df_out

# ----------------------------------------------------------------
# Build charts 3 & 4 from a DataFrame with Year
# ----------------------------------------------------------------
AWARE_MAP = {"Low": 1, "Medium": 2, "High": 3}

def build_10y_charts(source_df: pd.DataFrame):
    temp = attach_years(source_df)

    # Chart 3: yearly mean food waste
    waste_by_year = (
        temp.groupby("Year")[TARGET].mean().reset_index().sort_values("Year")
    )
    chart3 = make_plotly_line(
        waste_by_year["Year"],
        waste_by_year[TARGET],
        "Food Wastage — Last 10 Years",
        "Year",
        "kg/week (mean)"
    )

    # Chart 4: yearly mean awareness index
    aware_df = temp.copy()
    aware_df["AwarenessIndex"] = aware_df["Awareness_of_Waste_Management"].map(AWARE_MAP)
    awareness_by_year = (
        aware_df.groupby("Year")["AwarenessIndex"].mean().reset_index().sort_values("Year")
    )
    chart4 = make_plotly_line(
        awareness_by_year["Year"],
        awareness_by_year["AwarenessIndex"],
        "Awareness — Last 10 Years",
        "Year",
        "Awareness index (1–3)"
    )

    return chart3, chart4

# ----------------------------------------------------------------
# Flask Routes
# ----------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == "Akhil" and password == "123456":
            session["user"] = username
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid credentials!")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET"])
def signup():
    # The signup page uses client-side Firebase Authentication to create users.
    return render_template("signup.html")


@app.route('/session_login', methods=['POST'])
def session_login():
    """
    Lightweight session creation endpoint.
    The client should POST JSON {"email": "user@example.com"} after successful Firebase sign-in.
    NOTE: This endpoint does not verify Firebase ID tokens. For production, add server-side
    token verification using the Firebase Admin SDK before creating a session.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return ("Bad request", 400)

    email = data.get('email') if isinstance(data, dict) else None
    if not email:
        return ({'ok': False, 'error': 'email required'}, 400)

    session['user'] = email
    return ({'ok': True}, 200)

@app.route("/", methods=["GET"])
def index():
    if "user" not in session:
        return redirect(url_for("login"))

    # Existing two charts
    avg_by_meal = df.groupby("Meal_Type")[TARGET].mean().reset_index()
    chart1 = make_plotly_bar(
        avg_by_meal["Meal_Type"], avg_by_meal[TARGET],
        "Average Food Waste by Meal Type", "Meal Type", "kg/week"
    )
    avg_by_cuisine = df.groupby("Cultural_Cuisine_Preference")[TARGET].mean().reset_index()
    chart2 = make_plotly_bar(
        avg_by_cuisine["Cultural_Cuisine_Preference"], avg_by_cuisine[TARGET],
        "Average Food Waste by Cultural Cuisine Preference", "Cuisine", "kg/week"
    )

    # NEW: 10-year charts
    chart3, chart4 = build_10y_charts(df)

    return render_template(
        "index.html",
        chart1=chart1,
        chart2=chart2,
        chart3=chart3,
        chart4=chart4,
        prediction=None,
        form_values={}
    )

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    form = request.form
    input_record = {}
    for col in FEATURE_COLUMNS:
        val = form.get(col)
        if col in ["Household_Size", "Daily_Meal_Count"]:
            try:
                val = int(val)
            except:
                val = 1
        input_record[col] = val

    X_input = pd.DataFrame([input_record], columns=FEATURE_COLUMNS)
    pred = round(float(pipeline.predict(X_input)[0]), 2)

    # Existing two charts
    avg_by_meal = df.groupby("Meal_Type")[TARGET].mean().reset_index()
    chart1 = make_plotly_bar(
        avg_by_meal["Meal_Type"], avg_by_meal[TARGET],
        "Average Food Waste by Meal Type", "Meal Type", "kg/week"
    )
    avg_by_cuisine = df.groupby("Cultural_Cuisine_Preference")[TARGET].mean().reset_index()
    chart2 = make_plotly_bar(
        avg_by_cuisine["Cultural_Cuisine_Preference"], avg_by_cuisine[TARGET],
        "Average Food Waste by Cultural Cuisine Preference", "Cuisine", "kg/week"
    )

    # NEW: 10-year charts
    chart3, chart4 = build_10y_charts(df)

    return render_template(
        "index.html",
        chart1=chart1,
        chart2=chart2,
        chart3=chart3,
        chart4=chart4,
        prediction=pred,
        form_values=input_record
    )

# ----------------------------------------------------------------
# App Startup
# ----------------------------------------------------------------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/login")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    app.run(debug=True)
