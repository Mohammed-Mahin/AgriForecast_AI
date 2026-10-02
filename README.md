# AgriForecast AI v2.0 🌾📈

**AgriForecast AI v2.0** is an enterprise-grade AI Full-Stack Platform designed for Indian agricultural mandi price forecasting, perishability storage advisory, and live weather impact analysis covering **2024–2026**.

---

## 🌟 Key Capabilities & Features

1. **Dual-Language Support (English + Hindi):**
   - Seamless bilingual UI text, notification alerts, market insights, and storage advisories in English (🇬🇧) and Hindi (🇮🇳).
2. **High-Accuracy ML Weighted Stacking Ensemble:**
   - Combines **XGBoost Regressor**, **LightGBM**, and **CatBoost** via a **weighted Stacking Ensemble** (GradientBoosting meta-learner learns the blend weights; inverse-MAE weights are exported to `model_artifacts/metrics.json`).
   - Multi-scale rolling lags (3, 7, 14, 30, 60 days), EMA momentum ratios (`EMA_7 / EMA_30`, `EMA_14 / EMA_60`), cyclical seasonality (`sin`/`cos` for Month & Day of Year), perishability index, fuel/transport cost index, and live weather features (temperature, humidity, precipitation, dew point).
   - Inference seeds the lag features from **real trailing prices** in the 2024-2026 series, and the 7-day confidence band is scaled by each series' historical volatility (95% interval).
3. **Role-Based Authentication (JWT + bcrypt):**
   - Secure login & registration for `farmer` (किसान) and `mandi_officer` (मंडी अधिकारी) backed by SQLite/PostgreSQL (`database.py` & `auth.py`).
4. **Interactive Mandi Market Map & Intelligence:**
   - Visual map powered by Leaflet JS detailing 12 major Indian Mandi hubs (Azadpur, Vashi, Lasalgaon, Kolar, Gundlupet, Jaipur, Guntur, Kolkata, Mahbubnagar, Ahmedabad, Amritsar, Indore) with GPS coordinates, speciality commodities, and trade stats.
5. **Perishability & Storage Advisory Engine:**
   - **Tier 3 (High Risk):** Tomato, Onion, Leafy Greens, Peas (3-5 days shelf life) - Cool storage (2-5°C), reefer transport.
   - **Tier 2 (Medium Risk):** Potato, Ginger, Garlic, Apple, Mango (5-15 days shelf life) - Shaded ventilated storage.
   - **Tier 1 (Low Risk):** Rice, Wheat, Pulses, Turmeric, Cotton (365+ days shelf life) - Gunny bags below 12% moisture.
6. **Live Weather & Fuel Cost Integration:**
   - Current conditions **plus a 7-day forecast** (`temperature`, `humidity`, `precipitation_mm`, `dew_point`, wind) fetched dynamically from Open-Meteo at each mandi's GPS coordinates, with bilingual (EN/HI) impact interpretation.
   - Regional petrol/diesel **fuel/transport cost index** used as a freight-cost feature proxy (live API via `FUEL_API_URL` env var, with a regional reference table fallback).

---

## 📁 Repository Structure

```text
agriforecast-ai/
├── dataset/
│   └── agmarknet_india_historical_prices_2024_2026.csv
├── model_artifacts/
│   ├── price_model.pkl
│   ├── encoders.pkl
│   ├── scaler.pkl
│   └── metrics.json
├── static/
│   └── index.html
├── database.py
├── auth.py
├── train_model.py
├── main.py
├── generate_dataset.py
├── requirements.txt
└── README.md
```

---

## 🚀 Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset & ML Model Generation
```bash
# Generate 2024-2026 Agmarknet Mandi Dataset
python generate_dataset.py

# Train Stacking Ensemble Model and Export Artifacts
python train_model.py
```

### 3. Launch FastAPI Application Server
```bash
python main.py
```
Or with Uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser to view the AgriForecast AI v2.0 UI.

---

## 🔌 API Endpoints Summary

- `POST /api/auth/register`: User registration (`farmer` or `mandi_officer`)
- `POST /api/auth/login`: JWT login authentication
- `GET /api/auth/me`: Authenticated profile details
- `GET /api/health`: System health metrics, memory usage, dataset stats, model feature count, ensemble components & evaluation metrics
- `GET /api/mandis`: List of Indian Mandi hubs with GPS coordinates, state, and commodity specialities
- `GET /api/forecasts/trending`: High-volatility commodities computed live from the 2024-2026 series - modal price, momentum, volatility, bilingual risk tags/levels and 7-day outlooks (static fallback if the dataset is unavailable)
- `POST /api/predict`: High-accuracy price calculation (₹/Quintal & ₹/Kg), 7-day 95% confidence interval forecast, storage advisory (EN/HI), live weather + 7-day forecast, **price momentum indicator (up/down/stable with green/red/amber color)**, bilingual **market insight**, fuel/transport cost index, and real trailing lag features.
