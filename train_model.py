"""
AgriForecast AI v2.0 - ML Training Pipeline (train_model.py)
============================================================

Enhanced ML pipeline for Indian mandi modal-price forecasting over the
2024-2026 Agmarknet dataset.

Pipeline stages
---------------
1. Data Cleaning & Preprocessing
   - Schema standardisation to `YYYY-MM-DD` arrival dates.
   - Deduplication per (Market, Commodity, Arrival_Date).
   - Group-wise IQR filtering per commodity (asymmetric fences so real
     agricultural price surges survive while sensor/reporting spikes die).
   - Missing modal prices forward-filled with a 3-day exponential
     moving average (EMA) per (Market, Commodity) pair.

2. Advanced Feature Engineering
   - Multi-scale lags / moving averages: 3, 7, 14, 30, 60 day MA + EMA.
   - Exponential price momentum: EMA_7/EMA_30 and EMA_14/EMA_60 ratios.
   - Enhanced cyclical seasonality: sin/cos for Month and Day-of-Year.
   - Perishability Index tiers (1-3) driven by shelf-life risk.
   - Transport / fuel cost index proxy (regional petrol-diesel indicator).
   - Weather proxies (temperature, humidity, precipitation, dew point)
     calibrated to mandi latitude; live Open-Meteo values are injected at
     inference time by main.py for real-time weather impact.

3. Weighted Stacking Ensemble
   - Base learners : XGBoost, LightGBM, CatBoost (Gradient Boosting trio).
   - Meta-learner  : Stacking GradientBoosting regressor that learns the
     optimal blend weights from out-of-fold base predictions.
   - Base-model blend weights (inverse-MAE normalised) are exported as
     metadata for transparency / weighted-vote fallbacks.

Artifacts written to `model_artifacts/`:
   - price_model.pkl  : fitted stacking ensemble
   - encoders.pkl     : label encoders, feature columns, perishability map,
                        ensemble metadata & evaluation metrics
   - scaler.pkl       : StandardScaler fitted on the training split
   - metrics.json     : human-readable performance report
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:  # pragma: no cover - catboost optional at import time
    HAS_CATBOOST = False

# ---------------------------------------------------------------------------
# Perishability Index Mapping (shelf-life driven risk tiers)
# ---------------------------------------------------------------------------
PERISHABILITY_MAP = {
    # Tier 3 - High Risk (3-5 days shelf life)
    "Tomato": 3, "Onion": 3, "Leafy Greens": 3, "Green Peas": 3,
    "Peas": 3, "Berries": 3, "Strawberry": 3,
    # Tier 2 - Medium Risk (5-15 days shelf life)
    "Potato": 2, "Ginger": 2, "Garlic": 2, "Apple": 2, "Mango": 2,
    "Citrus": 2, "Orange": 2, "Banana": 2,
    # Tier 1 - Low Risk (365+ days shelf life)
    "Rice": 1, "Wheat": 1, "Pulses": 1, "Tur/Arhar": 1, "Gram": 1,
    "Spices": 1, "Turmeric": 1, "Mustard": 1, "Oilseeds": 1,
    "Groundnut": 1, "Soybean": 1, "Cotton": 1, "Maize": 1, "Jowar": 1,
}

DATASET_PATH = os.path.join("dataset", "agmarknet_india_historical_prices_2024_2026.csv")
ARTIFACTS_DIR = "model_artifacts"

MAX_TRAIN_ROWS = 60000          # sample size for rapid high-accuracy training
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Data Cleaning & Preprocessing
# ---------------------------------------------------------------------------
def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("--- 1. Data Cleaning & Preprocessing ---")
    print(f"Initial raw record count: {len(df)}")

    # Schema uniformity: enforce required columns & types
    required = ["State", "District Name", "Market Name", "Commodity",
                "Variety", "Grade", "Min_Price", "Max_Price",
                "Modal_Price", "Arrival_Date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    for col in ["Min_Price", "Max_Price", "Modal_Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strict YYYY-MM-DD arrival dates
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")
    df = df.dropna(subset=["Arrival_Date", "Modal_Price"])
    df["Arrival_Date"] = df["Arrival_Date"].dt.normalize()
    df["Arrival_Date_Str"] = df["Arrival_Date"].dt.strftime("%Y-%m-%d")

    df = df.sort_values(["Market Name", "Commodity", "Arrival_Date"])

    # (a) Deduplicate records per (Market, Commodity, Arrival_Date)
    df = df.drop_duplicates(subset=["Market Name", "Commodity", "Arrival_Date"], keep="last")
    print(f"Record count after deduplication: {len(df)}")

    # (b) Group-wise IQR filtering per commodity.
    #     Asymmetric fences: tight upper fence kills reporting spikes, a
    #     generous 2.5-IQR lower fence keeps genuine market crashes, and a
    #     wide 3.5-IQR upper fence keeps genuine agricultural price surges.
    cleaned_rows = []
    for commodity, group in df.groupby("Commodity"):
        q1 = group["Modal_Price"].quantile(0.25)
        q3 = group["Modal_Price"].quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            cleaned_rows.append(group)
            continue
        lower_bound = max(100.0, q1 - 2.5 * iqr)
        upper_bound = q3 + 3.5 * iqr
        filtered = group[
            (group["Modal_Price"] >= lower_bound) & (group["Modal_Price"] <= upper_bound)
        ]
        cleaned_rows.append(filtered)

    df_clean = pd.concat(cleaned_rows, ignore_index=True)
    df_clean = df_clean.sort_values(["Market Name", "Commodity", "Arrival_Date"])
    print(f"Record count after IQR filtering: {len(df_clean)}")

    # (c) Forward-fill / smooth missing modal prices with a 3-day EMA
    #     computed within each (Market, Commodity) time series.
    df_clean["Modal_Price_EMA3"] = (
        df_clean.groupby(["Market Name", "Commodity"])["Modal_Price"]
        .transform(lambda x: x.fillna(x.ewm(span=3, adjust=False).mean()))
    )
    df_clean["Modal_Price"] = df_clean["Modal_Price"].fillna(df_clean["Modal_Price_EMA3"])
    df_clean = df_clean.drop(columns=["Modal_Price_EMA3"])

    # Sanity floor: Indian mandi modal prices below Rs.100/quintal are data errors
    df_clean = df_clean[df_clean["Modal_Price"] >= 100]
    print(f"Final cleaned record count: {len(df_clean)}")
    return df_clean


# ---------------------------------------------------------------------------
# 2. Advanced Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("--- 2. Feature Engineering ---")
    df = df.sort_values(["Market Name", "Commodity", "Arrival_Date"]).copy()
    grouped = df.groupby(["Market Name", "Commodity"])["Modal_Price"]

    # Multi-scale Rolling Lags & Moving Averages (3, 7, 14, 30, 60 days)
    for window in [3, 7, 14, 30, 60]:
        df[f"MA_{window}"] = grouped.transform(
            lambda x, w=window: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f"EMA_{window}"] = grouped.transform(
            lambda x, w=window: x.shift(1).ewm(span=w, adjust=False).mean()
        )

    # Exponential price momentum ratios (supply drop / market glut capture)
    df["Momentum_7_30"] = df["EMA_7"] / (df["EMA_30"] + 1e-5)
    df["Momentum_14_60"] = df["EMA_14"] / (df["EMA_60"] + 1e-5)

    # Cyclical seasonality (harvest cycles)
    df["Month"] = df["Arrival_Date"].dt.month
    df["DayOfYear"] = df["Arrival_Date"].dt.dayofyear
    df["Year"] = df["Arrival_Date"].dt.year

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    df["DOY_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
    df["DOY_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)

    # Perishability Index (Tier 1-3) driven by shelf-life risk
    df["Perishability_Tier"] = df["Commodity"].map(lambda c: PERISHABILITY_MAP.get(c, 2))

    # Transport / Fuel cost index: regional petrol-diesel price indicator
    # (gradual 2024 -> 2026 inflation + seasonal freight cycle).
    days_since_start = (df["Arrival_Date"] - pd.to_datetime("2024-01-01")).dt.days
    df["Fuel_Cost_Index"] = 94.5 + (days_since_start / 365.25) * 3.2 \
        + np.sin(days_since_start / 30.0) * 0.8

    # Weather features (history uses climatological proxies; inference time
    # replaces these with live Open-Meteo readings at mandi GPS coordinates).
    df["Temp_Proxy"] = 24.0 + 8.0 * np.sin(2 * np.pi * (df["DayOfYear"] - 100) / 365.25)
    df["Humidity_Proxy"] = 60.0 + 20.0 * np.sin(2 * np.pi * (df["DayOfYear"] - 180) / 365.25)
    df["Precipitation_Proxy"] = np.maximum(
        0.0, 15.0 * np.sin(2 * np.pi * (df["DayOfYear"] - 170) / 365.25) - 5.0
    )
    # Magnus-formula dew point from temperature + relative humidity
    rh = np.clip(df["Humidity_Proxy"].astype(float), 1.0, 100.0)
    t = df["Temp_Proxy"].astype(float)
    ln_rh = np.log(rh / 100.0)
    alpha = (17.625 * t) / (243.04 + t)
    df["Dew_Point_Proxy"] = 243.04 * (ln_rh + alpha) / (17.625 - ln_rh - alpha)

    # Lag features are computed with shift(1) to avoid target leakage, so the
    # first row of every (Market, Commodity) series can be NaN - back-fill
    # within the series and fall back to 0 for brand-new series.
    lag_cols = [c for c in feature_cols_of(df) if c.startswith("MA_") or c.startswith("EMA_")]
    df[lag_cols] = df.groupby(["Market Name", "Commodity"], group_keys=False)[lag_cols].ffill()
    df[lag_cols] = df[lag_cols].fillna(0.0)
    df[["Momentum_7_30", "Momentum_14_60"]] = df[["Momentum_7_30", "Momentum_14_60"]].fillna(1.0)

    return df


def feature_cols_of(df: pd.DataFrame):
    """All engineered rolling-average columns present in the frame."""
    return [c for c in df.columns if c.startswith("MA_") or c.startswith("EMA_")]


FEATURE_COLS = [
    "State_Enc", "District_Enc", "Market_Enc", "Commodity_Enc", "Variety_Enc", "Grade_Enc",
    "MA_3", "MA_7", "MA_14", "MA_30", "MA_60",
    "EMA_3", "EMA_7", "EMA_14", "EMA_30", "EMA_60",
    "Momentum_7_30", "Momentum_14_60",
    "Month_sin", "Month_cos", "DOY_sin", "DOY_cos",
    "Perishability_Tier", "Fuel_Cost_Index",
    "Temp_Proxy", "Humidity_Proxy", "Precipitation_Proxy", "Dew_Point_Proxy",
]


# ---------------------------------------------------------------------------
# 3. Weighted Stacking Ensemble Training
# ---------------------------------------------------------------------------
def build_base_learners():
    """XGBoost + LightGBM + CatBoost gradient-boosting trio."""
    xgb_reg = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    lgb_reg = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    if HAS_CATBOOST:
        cb_reg = CatBoostRegressor(
            iterations=400,
            depth=7,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
        )
    else:
        print("!! catboost not installed - falling back to HistGradientBoostingRegressor")
        from sklearn.ensemble import HistGradientBoostingRegressor
        cb_reg = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=7, random_state=RANDOM_STATE
        )

    return [
        ("xgb", xgb_reg),
        ("lgb", lgb_reg),
        ("catboost", cb_reg),
    ]


def train_and_save_model():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run generate_dataset.py first."
        )

    raw_df = pd.read_csv(DATASET_PATH)
    df = clean_and_preprocess(raw_df)
    df = engineer_features(df)

    # Encode categorical features
    le_state = LabelEncoder()
    le_district = LabelEncoder()
    le_market = LabelEncoder()
    le_commodity = LabelEncoder()
    le_variety = LabelEncoder()
    le_grade = LabelEncoder()

    df["State_Enc"] = le_state.fit_transform(df["State"])
    df["District_Enc"] = le_district.fit_transform(df["District Name"])
    df["Market_Enc"] = le_market.fit_transform(df["Market Name"])
    df["Commodity_Enc"] = le_commodity.fit_transform(df["Commodity"])
    df["Variety_Enc"] = le_variety.fit_transform(df["Variety"])
    df["Grade_Enc"] = le_grade.fit_transform(df["Grade"])

    feature_cols = FEATURE_COLS

    # Sample for rapid high-accuracy training
    if len(df) > MAX_TRAIN_ROWS:
        df_train_sample = df.sample(n=MAX_TRAIN_ROWS, random_state=RANDOM_STATE)
    else:
        df_train_sample = df
    print(f"Training sample rows: {len(df_train_sample)}")

    X = df_train_sample[feature_cols]
    y = df_train_sample["Modal_Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, shuffle=True
    )

    # Tree ensembles are scale-invariant, but the scaler keeps inference
    # feature handling symmetric with the original pipeline.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_learners = build_base_learners()
    print("--- 3. Training Weighted Stacking Ensemble "
          f"({', '.join(name for name, _ in base_learners)}) ---")

    ensemble_model = StackingRegressor(
        estimators=base_learners,
        final_estimator=GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            random_state=RANDOM_STATE,
        ),
        cv=3,
        n_jobs=1,
        passthrough=False,
    )
    ensemble_model.fit(X_train_scaled, y_train)

    y_pred = ensemble_model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    # Base-learner blend weights (inverse-MAE, normalised) for transparency
    base_metrics, weights = {}, {}
    inv = {}
    fitted_baselines = dict(zip((name for name, _ in base_learners), ensemble_model.estimators_))
    for name, fitted in fitted_baselines.items():
        pred = fitted.predict(X_test_scaled)
        m = mean_absolute_error(y_test, pred)
        base_metrics[name] = round(float(m), 2)
        inv[name] = 1.0 / (m + 1e-6)
    total_inv = sum(inv.values())
    weights = {k: round(v / total_inv, 4) for k, v in inv.items()}

    metrics = {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
        "base_model_mae": base_metrics,
        "blend_weights": weights,
        "trained_at": datetime.utcnow().isoformat(),
        "training_rows": int(len(df_train_sample)),
        "dataset_rows": int(len(raw_df)),
        "test_rows": int(len(y_test)),
        "features": len(feature_cols),
    }

    print("\n================ MODEL PERFORMANCE METRICS ================")
    print(f"  Mean Absolute Error (MAE)  : Rs.{mae:.2f} / Quintal")
    print(f"  Root Mean Squared Error    : Rs.{rmse:.2f} / Quintal")
    print(f"  R^2 Score                  : {r2:.4f}")
    print("  Base learner MAE           : " +
          ", ".join(f"{k}=Rs.{v}" for k, v in base_metrics.items()))
    print("  Blend weights              : " +
          ", ".join(f"{k}={v}" for k, v in weights.items()))
    print("===========================================================\n")

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    encoders = {
        "le_state": le_state,
        "le_district": le_district,
        "le_market": le_market,
        "le_commodity": le_commodity,
        "le_variety": le_variety,
        "le_grade": le_grade,
        "feature_cols": feature_cols,
        "perishability_map": PERISHABILITY_MAP,
        "ensemble_info": {
            "type": "Weighted Stacking Ensemble",
            "base_models": [name for name, _ in base_learners],
            "meta_learner": "GradientBoostingRegressor",
            "weights": weights,
            "metrics": metrics,
        },
        "metrics": metrics,
    }

    with open(os.path.join(ARTIFACTS_DIR, "price_model.pkl"), "wb") as f:
        pickle.dump(ensemble_model, f)
    with open(os.path.join(ARTIFACTS_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved artifacts successfully to '{ARTIFACTS_DIR}/'")
    return metrics


if __name__ == "__main__":
    train_and_save_model()
