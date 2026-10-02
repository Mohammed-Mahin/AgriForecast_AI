import os
import math
import pickle
import psutil
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from database import init_db, get_db, SessionLocal, User, SavedForecast, MandiReport
from auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user
)

# Initialize database tables
init_db()

app = FastAPI(
    title="AgriForecast AI v2.0 API",
    description="High-performance Indian Mandi Price Forecasting & Storage Advisory Engine (2024-2026)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Artifacts
MODEL_DIR = "model_artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "price_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

model = None
encoders = None
scaler = None

def load_artifacts():
    global model, encoders, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH) and os.path.exists(SCALER_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(ENCODERS_PATH, "rb") as f:
                encoders = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            print("Successfully loaded ML model and encoders!")
        except Exception as e:
            print(f"Error loading model artifacts: {e}")

load_artifacts()

# Mandis Static Reference Data
MANDIS_DATA = [
    {
        "market": "Azadpur Mandi", "state": "Delhi", "district": "Delhi",
        "lat": 28.716, "lng": 77.170, "speciality": "Fruits & Vegetables Hub",
        "crops": ["Tomato", "Onion", "Potato", "Apple", "Mango", "Berries"]
    },
    {
        "market": "Vashi APMC Mandi", "state": "Maharashtra", "district": "Mumbai",
        "lat": 19.076, "lng": 73.008, "speciality": "Spices, Fruits & Pulses",
        "crops": ["Mango", "Onion", "Turmeric", "Rice", "Wheat", "Leafy Greens"]
    },
    {
        "market": "Lasalgaon Mandi", "state": "Maharashtra", "district": "Nashik",
        "lat": 20.147, "lng": 74.227, "speciality": "Asia's Largest Onion Market",
        "crops": ["Onion", "Tomato", "Garlic", "Ginger", "Wheat"]
    },
    {
        "market": "Kolar Mandi", "state": "Karnataka", "district": "Kolar",
        "lat": 13.136, "lng": 78.129, "speciality": "Major Tomato & Vegetable Mandi",
        "crops": ["Tomato", "Green Peas", "Potato", "Leafy Greens", "Rice"]
    },
    {
        "market": "Gundlupet Mandi", "state": "Karnataka", "district": "Chamarajanagar",
        "lat": 11.808, "lng": 76.690, "speciality": "Horticulture & Spices",
        "crops": ["Turmeric", "Ginger", "Garlic", "Pulses", "Citrus"]
    },
    {
        "market": "Jaipur Mandi", "state": "Rajasthan", "district": "Jaipur",
        "lat": 26.912, "lng": 75.787, "speciality": "Grains, Mustard & Oilseeds",
        "crops": ["Mustard", "Wheat", "Pulses", "Onion", "Cotton"]
    },
    {
        "market": "Guntur Mandi", "state": "Andhra Pradesh", "district": "Guntur",
        "lat": 16.306, "lng": 80.436, "speciality": "Chilli & Commercial Cash Crops",
        "crops": ["Cotton", "Turmeric", "Rice", "Mango", "Citrus"]
    },
    {
        "market": "Kolkata Mandi", "state": "West Bengal", "district": "Kolkata",
        "lat": 22.572, "lng": 88.363, "speciality": "Eastern Region Grain & Produce Hub",
        "crops": ["Rice", "Potato", "Leafy Greens", "Green Peas", "Ginger"]
    },
    {
        "market": "Mahbubnagar Mandi", "state": "Telangana", "district": "Mahbubnagar",
        "lat": 16.748, "lng": 77.984, "speciality": "Cotton & Pulses Market",
        "crops": ["Cotton", "Pulses", "Rice", "Mango", "Mustard"]
    },
    {
        "market": "Ahmedabad Mandi", "state": "Gujarat", "district": "Ahmedabad",
        "lat": 23.022, "lng": 72.571, "speciality": "Cotton, Groundnut & Vegetables",
        "crops": ["Cotton", "Mustard", "Potato", "Onion", "Wheat"]
    },
    {
        "market": "Amritsar Mandi", "state": "Punjab", "district": "Amritsar",
        "lat": 31.634, "lng": 74.872, "speciality": "Wheat & Basmati Rice Granary",
        "crops": ["Wheat", "Rice", "Green Peas", "Potato", "Apple"]
    },
    {
        "market": "Indore Mandi", "state": "Madhya Pradesh", "district": "Indore",
        "lat": 22.719, "lng": 75.857, "speciality": "Soybean, Wheat & Garlic Hub",
        "crops": ["Wheat", "Garlic", "Pulses", "Mustard", "Potato"]
    },
]

PERISHABILITY_ADVISORY = {
    3: {
        "tier": "Tier 3 (High Perishability / उच्च जोखिम)",
        "shelf_life_days": "3 - 5 Days",
        "en": {
            "risk_label": "High Shelf-Life Risk",
            "storage_tip": "Maintain cool storage (2-5°C) with 85-90% humidity. Sell quickly or process immediately to prevent post-harvest loss.",
            "preservation": "Use ventilated crates. Avoid stacking more than 3 layers deep. Transport in refrigerated reefer vans if distance > 100km."
        },
        "hi": {
            "risk_label": "उच्च भंडारण जोखिम",
            "storage_tip": "ठंडे तापमान (2-5°C) और 85-90% आर्द्रता में रखें। कटाई के बाद नुकसान से बचने के लिए इसे 3-5 दिनों में बेचें या प्रसंस्कृत करें।",
            "preservation": "हवादार क्रेटों का उपयोग करें। 3 परतों से अधिक न थोपें। लंबे परिवहन के लिए रीफर वैन का उपयोग करें।"
        }
    },
    2: {
        "tier": "Tier 2 (Medium Perishability / मध्यम जोखिम)",
        "shelf_life_days": "5 - 15 Days",
        "en": {
            "risk_label": "Medium Shelf-Life Risk",
            "storage_tip": "Store in dry, well-ventilated shade at 12-18°C. Monitor regularly for humidity buildup and moisture.",
            "preservation": "Cure roots/tubers for 3 days after harvest. Keep off wet floors using wooden pallets."
        },
        "hi": {
            "risk_label": "मध्यम भंडारण जोखिम",
            "storage_tip": "सूखी, हवादार छायादार जगह (12-18°C) में रखें। नियमित रूप से नमी और सड़ांध की जांच करें।",
            "preservation": "फसल कटाई के बाद 3 दिन तक छाया में सुखाएं। लकड़ी के पैलेट का उपयोग करें।"
        }
    },
    1: {
        "tier": "Tier 1 (Low Perishability / निम्न जोखिम)",
        "shelf_life_days": "365+ Days",
        "en": {
            "risk_label": "Low Shelf-Life Risk",
            "storage_tip": "Store in moisture-proof gunny bags in airtight silos or godowns below 12% moisture content.",
            "preservation": "Apply neem oil treatment or herbal pest repellents. Fumigate every 6 months."
        },
        "hi": {
            "risk_label": "कम भंडारण जोखिम",
            "storage_tip": "12% से कम नमी वाले एयरटाइट गोदामों या जूट के बोरों में सुरक्षित रूप से स्टोर करें।",
            "preservation": "नीम के तेल या हर्बल कीट निवारक का प्रयोग करें। हर 6 महीने में कीटनाशक उपचार करें।"
        }
    }
}

# Pydantic Request/Response Models
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: Optional[str] = "farmer" # "farmer" or "mandi_officer"
    full_name: Optional[str] = ""
    state: Optional[str] = ""
    preferred_mandi: Optional[str] = ""

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]

class PredictRequest(BaseModel):
    market: str
    commodity: str
    date: Optional[str] = None # YYYY-MM-DD
    variety: Optional[str] = None
    grade: Optional[str] = None

class SaveForecastRequest(BaseModel):
    market_name: str
    commodity: str
    predicted_modal_price: float
    predicted_kg_price: float
    perishability_tier: Optional[int] = 2

class MandiReportRequest(BaseModel):
    market_name: str
    commodity: str
    reported_price: float
    arrival_tonnes: float
    notes: Optional[str] = ""

# ---------------------------------------------------------------------------
# Live Weather Integration (Open-Meteo, real GPS coordinates per mandi)
# ---------------------------------------------------------------------------
def _dew_point(temp: float, humidity: float) -> float:
    """Magnus-formula dew point - mirrors the training feature construction."""
    rh = max(1.0, min(100.0, float(humidity or 60.0)))
    t = float(temp)
    ln_rh = math.log(rh / 100.0)
    alpha = (17.625 * t) / (243.04 + t)
    return round(243.04 * (ln_rh + alpha) / (17.625 - ln_rh - alpha), 2)


def _weather_impact(temp, humidity, precip, weathercode, forecast_precip):
    """Bilingual (EN/HI) weather -> mandi price impact interpretation."""
    rainy = (51 <= weathercode <= 67) or (80 <= weathercode <= 82) or weathercode >= 95
    if forecast_precip >= 10 or (rainy and precip >= 5):
        return ("Heavy rainfall may delay arrivals and firm up prices",
                "भारी बारिश से आवक में देरी और मूल्यों में तेजी संभव")
    if rainy or forecast_precip >= 2:
        return ("Light rain may slow transport; prices stable-to-firm",
                "हल्की बारिश से परिवहन धीमा; मूल्य स्थिर से मजबूत")
    if temp >= 38:
        return ("Heat stress may cut arrivals and lift perishable prices",
                "गर्मी से आवक घट सकती है; नाशवान फसलों के दाम बढ़ सकते हैं")
    if humidity >= 85:
        return ("High humidity raises spoilage risk for perishable produce",
                "अधिक आर्द्रता से नाशवान उपज में खराबी का खतरा")
    return ("Favorable harvesting & transport conditions",
            "कटाई एवं परिवहन के लिए अनुकूल मौसम")


def get_live_weather(lat: float, lng: float) -> Dict[str, Any]:
    """Current conditions + 7-day forecast from Open-Meteo at mandi GPS point."""
    try:
        res = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lng,
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,dew_point_2m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_max",
                "timezone": "auto", "forecast_days": 7,
            },
            timeout=4,
        )
        if res.status_code == 200:
            data = res.json()
            cur = data.get("current", {}) or {}
            daily = data.get("daily", {}) or {}

            temp = float(cur.get("temperature_2m", 28.5))
            humidity = float(cur.get("relative_humidity_2m", 62.0) or 62.0)
            precip = float(cur.get("precipitation", 0.0) or 0.0)
            dew = cur.get("dew_point_2m")
            dew = float(dew) if dew is not None else _dew_point(temp, humidity)
            wcode = int(cur.get("weather_code", 0) or 0)

            def _daily_val(key, i, default=None):
                vals = daily.get(key) or []
                if i < len(vals) and vals[i] is not None:
                    return vals[i]
                return default

            forecast_7d, total_precip = [], 0.0
            for i, day_str in enumerate(daily.get("time") or []):
                p = float(_daily_val("precipitation_sum", i, 0.0) or 0.0)
                total_precip += p
                forecast_7d.append({
                    "date": day_str,
                    "temp_max": _daily_val("temperature_2m_max", i),
                    "temp_min": _daily_val("temperature_2m_min", i),
                    "precipitation_mm": round(p, 2),
                    "humidity_max": _daily_val("relative_humidity_2m_max", i),
                })

            cond_en, cond_hi = _weather_impact(temp, humidity, precip, wcode, total_precip)
            return {
                "temperature": round(temp, 1),
                "humidity": round(humidity, 1),
                "precipitation_mm": round(precip, 2),
                "dew_point": round(dew, 1),
                "windspeed": round(float(cur.get("wind_speed_10m", 10.0) or 10.0), 1),
                "weathercode": wcode,
                "status": "Live Open-Meteo GPS API",
                "condition": cond_en,
                "condition_hi": cond_hi,
                "forecast_precipitation_mm": round(total_precip, 2),
                "forecast_7d": forecast_7d,
            }
    except Exception:
        pass

    # Localized climate proxy fallback (climatological, mirrors training proxies)
    doy = datetime.utcnow().timetuple().tm_yday
    temp = round(24.0 + 8.0 * math.sin(2 * math.pi * (doy - 100) / 365.25), 1)
    humidity = round(60.0 + 20.0 * math.sin(2 * math.pi * (doy - 180) / 365.25), 1)
    precip = round(max(0.0, 15.0 * math.sin(2 * math.pi * (doy - 170) / 365.25) - 5.0), 2)
    cond_en, cond_hi = _weather_impact(temp, humidity, precip, 0, precip * 3)
    return {
        "temperature": temp, "humidity": humidity, "precipitation_mm": precip,
        "dew_point": _dew_point(temp, humidity), "windspeed": 10.5, "weathercode": 0,
        "status": "Localized Climate Proxy", "condition": cond_en, "condition_hi": cond_hi,
        "forecast_precipitation_mm": round(precip * 3, 2), "forecast_7d": [],
    }


# ---------------------------------------------------------------------------
# Fuel / Transport Cost Index (regional petrol-diesel price indicator)
# ---------------------------------------------------------------------------
REGIONAL_FUEL_PRICES = {
    "Delhi": (103.5, 89.7), "Maharashtra": (111.5, 96.3), "Karnataka": (102.9, 89.4),
    "Rajasthan": (113.7, 96.9), "Andhra Pradesh": (110.4, 95.6), "West Bengal": (106.0, 92.6),
    "Telangana": (109.7, 97.9), "Gujarat": (106.4, 92.7), "Punjab": (110.2, 98.4),
    "Madhya Pradesh": (118.6, 101.3),
}
NATIONAL_FUEL = (108.5, 94.5)  # (petrol, diesel) Rs./L national reference


def get_fuel_cost_index(state: str = None, as_of: datetime = None) -> Dict[str, Any]:
    """
    Transport-cost feature proxy.
    Priority: (1) optional live fuel API via FUEL_API_URL env var (JSON with
    petrol/diesel, optionally keyed by state), else (2) regional petrol/diesel
    reference prices + temporal drift consistent with the training-time
    Fuel_Cost_Index series.
    """
    as_of = as_of or datetime.utcnow()
    days = (as_of.date() - datetime(2024, 1, 1).date()).days
    base = 94.5 + (days / 365.25) * 3.2 + math.sin(days / 30.0) * 0.8
    petrol, diesel = REGIONAL_FUEL_PRICES.get(state, NATIONAL_FUEL)
    source = "Regional petrol/diesel reference indicator"
    live = False

    api_url = os.getenv("FUEL_API_URL")
    if api_url:
        try:
            res = requests.get(api_url, timeout=3)
            if res.status_code == 200:
                payload = res.json()
                node = payload.get(state) if isinstance(payload.get(state), dict) else payload
                p = float(node.get("petrol", petrol))
                d = float(node.get("diesel", diesel))
                nat = 0.4 * NATIONAL_FUEL[0] + 0.6 * NATIONAL_FUEL[1]
                base = (0.4 * p + 0.6 * d) / nat * 94.5
                petrol, diesel, source, live = p, d, "Live Fuel Price API", True
        except Exception:
            pass

    if not live:
        regional_delta = ((petrol + diesel) - sum(NATIONAL_FUEL)) / sum(NATIONAL_FUEL) * 100.0
        base += regional_delta * 0.1

    return {
        "index": round(base, 3),
        "petrol_rs_l": petrol,
        "diesel_rs_l": diesel,
        "state": state or "National Average",
        "source": source,
    }


# ---------------------------------------------------------------------------
# Historical price series cache (2024-2026 Agmarknet dataset)
# ---------------------------------------------------------------------------
PRICE_HISTORY: Dict[tuple, Any] = {}
_HISTORY_LOADED = False
_DATASET_RECORDS: Optional[int] = None


def load_price_history() -> Dict[tuple, Any]:
    global _HISTORY_LOADED, _DATASET_RECORDS
    if _HISTORY_LOADED:
        return PRICE_HISTORY
    _HISTORY_LOADED = True
    dataset_file = os.path.join("dataset", "agmarknet_india_historical_prices_2024_2026.csv")
    if not os.path.exists(dataset_file):
        return PRICE_HISTORY
    try:
        hist_df = pd.read_csv(dataset_file, usecols=["Market Name", "Commodity", "Arrival_Date", "Modal_Price"])
        hist_df["Arrival_Date"] = pd.to_datetime(hist_df["Arrival_Date"], errors="coerce")
        hist_df = hist_df.dropna(subset=["Arrival_Date", "Modal_Price"]).sort_values("Arrival_Date")
        _DATASET_RECORDS = int(len(hist_df))
        for (market, commodity), g in hist_df.groupby(["Market Name", "Commodity"]):
            g = g.sort_values("Arrival_Date")
            prices = g["Modal_Price"].to_numpy(dtype=float)
            # Same group-wise IQR policy as training: drop reporting spikes so
            # momentum / volatility stats reflect real market moves.
            q1, q3 = np.percentile(prices, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = max(100.0, q1 - 3.0 * iqr), q3 + 3.0 * iqr
                keep = (prices >= lo) & (prices <= hi)
                g, prices = g[keep], prices[keep]
            # Drop day-over-day reporting spikes (>30% jumps) that survive IQR
            if len(prices) > 2:
                keep_spike = np.ones(len(prices), dtype=bool)
                prev = prices[0]
                for i in range(1, len(prices)):
                    ratio = prices[i] / prev if prev > 0 else 1.0
                    if 0.7 <= ratio <= 1.3:
                        prev = prices[i]
                    else:
                        keep_spike[i] = False
                if keep_spike.any() and not keep_spike.all():
                    g, prices = g[keep_spike], prices[keep_spike]
            if len(prices) == 0:
                continue
            PRICE_HISTORY[(str(market).lower(), str(commodity).lower())] = (
                g["Arrival_Date"].to_numpy(),
                prices,
            )
        print(f"Loaded {len(PRICE_HISTORY)} historical price series "
              f"({_DATASET_RECORDS} records, 2024-2026).")
    except Exception as e:
        print(f"Price history load failed: {e}")
    return PRICE_HISTORY


def get_price_series(market: str, commodity: str):
    hist = load_price_history()
    key = (str(market).lower(), str(commodity).lower())
    if key in hist:
        return hist[key]
    for (mk, cm), series in hist.items():
        if cm == key[1] and (mk.startswith(key[0]) or key[0].startswith(mk)):
            return series
    return None


def _series_as_of(series, target_date):
    """Prices strictly before the target date (falls back to full series)."""
    dates, prices = series
    mask = dates < np.datetime64(target_date.strftime("%Y-%m-%d"))
    p = prices[mask]
    return p if len(p) >= 7 else prices


def _price_stats(p: np.ndarray) -> Dict[str, float]:
    """7v14 momentum, daily log-return volatility and drift of a price series."""
    s = pd.Series(p)
    n = len(s)
    if n >= 14:
        last7, prev7 = float(s.tail(7).mean()), float(s.iloc[-14:-7].mean())
    elif n >= 8:
        last7, prev7 = float(s.tail(7).mean()), float(s.iloc[: n - 7].mean())
    else:
        last7 = prev7 = float(s.mean())
    momentum = (last7 - prev7) / prev7 * 100.0 if prev7 else 0.0
    rets = np.diff(np.log(np.clip(p, 1.0, None))) if n > 2 else np.array([0.0])
    recent = rets[-30:]
    return {
        "momentum_pct": round(float(momentum), 2),
        "daily_vol": float(max(np.std(recent), 0.004)),
        "daily_drift": float(np.mean(recent)) if len(recent) else 0.0,
        "last_price": float(p[-1]),
        "observations": int(n),
    }


def build_lag_features(series, target_date):
    """Real trailing MA/EMA lags & momentum from the 2024-2026 series."""
    if series is None:
        return None, None, 0
    p = _series_as_of(series, target_date)
    if len(p) < 5:
        return None, (float(p[-1]) if len(p) else None), len(p)
    s = pd.Series(p)
    feats = {}
    for w in (3, 7, 14, 30, 60):
        feats[f"MA_{w}"] = float(s.tail(w).mean())
        feats[f"EMA_{w}"] = float(s.ewm(span=w, adjust=False).mean().iloc[-1])
    feats["Momentum_7_30"] = feats["EMA_7"] / (feats["EMA_30"] + 1e-5)
    feats["Momentum_14_60"] = feats["EMA_14"] / (feats["EMA_60"] + 1e-5)
    return feats, float(p[-1]), len(p)


def build_7d_outlook(base_price: float, stats: Dict[str, float], start_date: datetime):
    """Drift + historical-volatility scaled 95% confidence interval (7 days)."""
    growth = math.exp(min(max(stats["daily_drift"], -0.015), 0.015)) - 1.0
    vol = stats["daily_vol"]
    rows, price = [], float(base_price)
    for d in range(7):
        if d:
            price *= (1.0 + growth)
        band = min(1.96 * vol * math.sqrt(d + 1), 0.25)
        expected = round(price, 2)
        curr_dt = start_date + timedelta(days=d)
        rows.append({
            "day": curr_dt.strftime("%a (%b %d)"),
            "date": curr_dt.strftime("%Y-%m-%d"),
            "predicted_modal_price": expected,
            "predicted_kg_price": round(expected / 100.0, 2),
            "min_confidence": round(expected * (1 - band), 2),
            "max_confidence": round(expected * (1 + band), 2),
        })
    mom7 = ((1.0 + growth) ** 7 - 1.0) * 100.0
    return rows, round(mom7, 2)


def _ensemble_components():
    known = {"xgb": "XGBoost Regressor", "lgb": "LightGBM Regressor",
             "catboost": "CatBoost Regressor"}
    if encoders and isinstance(encoders.get("ensemble_info"), dict):
        info = encoders["ensemble_info"]
        base = [known.get(m, f"{m} Regressor") for m in info.get("base_models", [])]
        meta = info.get("meta_learner", "GradientBoosting")
        if base:
            return base + [f"Stacking {meta} Meta-Learner"]
    return ["XGBoost Regressor", "LightGBM Regressor", "CatBoost Regressor",
            "Stacking GradientBoosting Meta-Learner"]

# ================= REST ENDPOINTS =================

@app.post("/api/auth/register", response_model=TokenResponse)
def register(user_in: UserRegister, db: Session = Depends(get_db)):
    db_user = db.query(User).filter((User.username == user_in.username) | (User.email == user_in.email)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or Email already registered")

    new_user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        role=user_in.role,
        full_name=user_in.full_name,
        state=user_in.state,
        preferred_mandi=user_in.preferred_mandi
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token = create_access_token(data={"sub": new_user.username, "role": new_user.role})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": new_user.id,
            "username": new_user.username,
            "email": new_user.email,
            "role": new_user.role,
            "full_name": new_user.full_name,
            "preferred_mandi": new_user.preferred_mandi
        }
    }

@app.post("/api/auth/login", response_model=TokenResponse)
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == credentials.username).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "full_name": user.full_name,
            "preferred_mandi": user.preferred_mandi
        }
    }

@app.get("/api/auth/me")
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role,
        "full_name": current_user.full_name,
        "state": current_user.state,
        "preferred_mandi": current_user.preferred_mandi,
        "created_at": current_user.created_at.isoformat()
    }

@app.get("/api/health")
def health_check():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    load_price_history()
    dataset_records = _DATASET_RECORDS or 0

    return {
        "status": "healthy",
        "service": "AgriForecast AI v2.0",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None,
        "dataset_coverage": "2024 - 2026",
        "dataset_total_records": dataset_records,
        "active_features_count": len(encoders['feature_cols']) if encoders else 28,
        "memory_usage_mb": round(mem_info.rss / (1024 * 1024), 2),
        "ensemble_components": _ensemble_components(),
        "model_metrics": (encoders or {}).get("metrics"),
        "weather_api": "Open-Meteo live GPS (current + 7-day forecast)",
        "fuel_index_source": get_fuel_cost_index().get("source")
    }

@app.get("/api/mandis")
def get_mandis():
    return {
        "total_mandis": len(MANDIS_DATA),
        "mandis": MANDIS_DATA
    }

TRENDING_WATCHLIST = [
    ("Tomato", "Azadpur Mandi"),
    ("Onion", "Lasalgaon Mandi"),
    ("Potato", "Vashi APMC Mandi"),
    ("Wheat", "Amritsar Mandi"),
    ("Apple", "Azadpur Mandi"),
    ("Green Peas", "Kolar Mandi"),
    ("Cotton", "Ahmedabad Mandi"),
    ("Turmeric", "Guntur Mandi"),
]

# Static fallback (used only when the historical dataset is unavailable)
STATIC_TRENDING_FALLBACK = [
    {
        "commodity": "Tomato", "market": "Azadpur Mandi",
        "modal_price": 2950.00, "kg_price": 29.50,
        "momentum": "+4.2%", "momentum_pct": 4.2, "trend": "up",
        "perishability": 3, "risk_level": "High", "risk_level_hi": "उच्च",
        "risk_tag": "High Volatility", "risk_tag_hi": "उच्च अस्थिरता",
        "volatility_pct": 19.4,
        "outlook_7d": [2950, 2980, 3020, 3090, 3120, 3100, 3050]
    },
    {
        "commodity": "Onion", "market": "Lasalgaon Mandi",
        "modal_price": 3200.00, "kg_price": 32.00,
        "momentum": "+2.8%", "momentum_pct": 2.8, "trend": "up",
        "perishability": 3, "risk_level": "High", "risk_level_hi": "उच्च",
        "risk_tag": "Supply Glut Alert", "risk_tag_hi": "आपूर्ति अधिकता चेतावनी",
        "volatility_pct": 16.1,
        "outlook_7d": [3200, 3210, 3240, 3280, 3310, 3350, 3380]
    },
    {
        "commodity": "Potato", "market": "Vashi APMC Mandi",
        "modal_price": 1950.00, "kg_price": 19.50,
        "momentum": "-1.1%", "momentum_pct": -1.1, "trend": "down",
        "perishability": 2, "risk_level": "Medium", "risk_level_hi": "मध्यम",
        "risk_tag": "Stable Supply", "risk_tag_hi": "स्थिर आपूर्ति",
        "volatility_pct": 7.2,
        "outlook_7d": [1950, 1940, 1930, 1925, 1920, 1915, 1910]
    },
    {
        "commodity": "Wheat", "market": "Amritsar Mandi",
        "modal_price": 2650.00, "kg_price": 26.50,
        "momentum": "+0.5%", "momentum_pct": 0.5, "trend": "stable",
        "perishability": 1, "risk_level": "Low", "risk_level_hi": "निम्न",
        "risk_tag": "Low Perishability / Secure", "risk_tag_hi": "कम जोखिम / सुरक्षित",
        "volatility_pct": 3.4,
        "outlook_7d": [2650, 2652, 2655, 2658, 2660, 2662, 2665]
    },
    {
        "commodity": "Apple", "market": "Azadpur Mandi",
        "modal_price": 8800.00, "kg_price": 88.00,
        "momentum": "+3.6%", "momentum_pct": 3.6, "trend": "up",
        "perishability": 2, "risk_level": "Medium", "risk_level_hi": "मध्यम",
        "risk_tag": "Seasonal Festival Demand", "risk_tag_hi": "त्योहारी मांग",
        "volatility_pct": 9.8,
        "outlook_7d": [8800, 8880, 8950, 9020, 9100, 9150, 9200]
    },
]


@app.get("/api/forecasts/trending")
def get_trending_forecasts():
    """High-volatility crops with live modal prices, risk levels & 7-day outlooks
    computed from the 2024-2026 Agmarknet series (static fallback if unavailable)."""
    items = []
    for commodity, market in TRENDING_WATCHLIST:
        series = get_price_series(market, commodity)
        if series is None:
            continue
        stats = _price_stats(series[1][-30:])

        tier = 2
        if encoders and "perishability_map" in encoders:
            tier = encoders["perishability_map"].get(commodity, 2)

        mom = stats["momentum_pct"]
        trend = "up" if mom > 0.75 else ("down" if mom < -0.75 else "stable")
        vol_pct = stats["daily_vol"] * math.sqrt(7) * 100.0

        if vol_pct > 18:
            risk_en, risk_hi = "High Volatility", "उच्च अस्थिरता"
        elif mom >= 3:
            risk_en, risk_hi = "Price Surge Alert", "मूल्य वृद्धि चेतावनी"
        elif mom <= -3:
            risk_en, risk_hi = "Supply Glut / Price Drop", "आपूर्ति अधिकता / मूल्य गिरावट"
        elif trend == "stable":
            risk_en, risk_hi = "Stable Trade Range", "स्थिर व्यापार सीमा"
        else:
            risk_en, risk_hi = "Firming Prices", "मूल्य मजबूत हो रहे"

        if vol_pct > 15 or tier == 3:
            risk_level, risk_level_hi = "High", "उच्च"
        elif vol_pct > 8 or tier == 2:
            risk_level, risk_level_hi = "Medium", "मध्यम"
        else:
            risk_level, risk_level_hi = "Low", "निम्न"

        outlook_rows, _ = build_7d_outlook(stats["last_price"], stats, datetime.utcnow())
        items.append({
            "commodity": commodity,
            "market": market,
            "modal_price": round(stats["last_price"], 2),
            "kg_price": round(stats["last_price"] / 100.0, 2),
            "momentum": f"{mom:+.1f}%",
            "momentum_pct": mom,
            "trend": trend,
            "perishability": tier,
            "volatility_pct": round(vol_pct, 1),
            "risk_level": risk_level,
            "risk_level_hi": risk_level_hi,
            "risk_tag": risk_en,
            "risk_tag_hi": risk_hi,
            "outlook_7d": [row["predicted_modal_price"] for row in outlook_rows],
            "last_updated": datetime.utcnow().isoformat(),
        })

    if not items:
        return {"trending": STATIC_TRENDING_FALLBACK}
    items.sort(key=lambda x: abs(x["momentum_pct"]) * max(x["volatility_pct"], 0.1), reverse=True)
    return {"trending": items}

@app.post("/api/predict")
def predict_price(req: PredictRequest):
    # Find matching Mandi info
    mandi_info = next((m for m in MANDIS_DATA if m["market"].lower() == req.market.lower()), MANDIS_DATA[0])
    lat, lng = mandi_info["lat"], mandi_info["lng"]

    # Live weather at mandi GPS coordinates (Open-Meteo current + 7-day)
    weather = get_live_weather(lat, lng)

    # Date parsing
    if req.date:
        try:
            target_date = datetime.strptime(req.date, "%Y-%m-%d")
        except Exception:
            target_date = datetime.now()
    else:
        target_date = datetime.now()

    # Perishability mapping
    perishability_tier = 2
    if encoders and 'perishability_map' in encoders:
        perishability_tier = encoders['perishability_map'].get(req.commodity, 2)
    elif req.commodity in ["Tomato", "Onion", "Leafy Greens", "Green Peas", "Berries"]:
        perishability_tier = 3
    elif req.commodity in ["Rice", "Wheat", "Pulses", "Turmeric", "Mustard", "Cotton"]:
        perishability_tier = 1

    # Real trailing lag features from the 2024-2026 Agmarknet series
    series = get_price_series(req.market, req.commodity)
    lag_feats, last_observed, history_days = build_lag_features(series, target_date)

    # Regional transport / fuel cost index (freight-cost feature proxy)
    fuel = get_fuel_cost_index(mandi_info["state"], target_date)

    # Heuristic base price when no history exists for the pair
    base_p = 3000.0
    if req.commodity == "Tomato": base_p = 2800.0
    elif req.commodity == "Onion": base_p = 3200.0
    elif req.commodity == "Potato": base_p = 1950.0
    elif req.commodity == "Rice": base_p = 3900.0
    elif req.commodity == "Wheat": base_p = 2650.0
    elif req.commodity == "Apple": base_p = 8600.0
    elif req.commodity == "Garlic": base_p = 11200.0
    elif req.commodity == "Turmeric": base_p = 13800.0
    if last_observed:
        base_p = float(last_observed)

    # Base price prediction: trained weighted stacking regressor
    modal_price_val = 3000.0
    if model and encoders and scaler:
        try:
            # Transform categorical features
            def safe_transform(le, val):
                val_str = str(val)
                if val_str in le.classes_:
                    return le.transform([val_str])[0]
                return 0

            state_enc = safe_transform(encoders['le_state'], mandi_info["state"])
            dist_enc = safe_transform(encoders['le_district'], mandi_info["district"])
            mkt_enc = safe_transform(encoders['le_market'], mandi_info["market"])
            comm_enc = safe_transform(encoders['le_commodity'], req.commodity)
            var_enc = safe_transform(encoders['le_variety'], req.variety or "FAQ")
            grd_enc = safe_transform(encoders['le_grade'], req.grade or "FAQ")

            # Date features
            doy = target_date.timetuple().tm_yday
            month = target_date.month

            if lag_feats is None:
                lag_feats = {}
                for w in (3, 7, 14, 30, 60):
                    lag_feats[f"MA_{w}"] = base_p
                    lag_feats[f"EMA_{w}"] = base_p
                lag_feats["Momentum_7_30"] = 1.02
                lag_feats["Momentum_14_60"] = 1.03

            feature_dict = {
                'State_Enc': state_enc,
                'District_Enc': dist_enc,
                'Market_Enc': mkt_enc,
                'Commodity_Enc': comm_enc,
                'Variety_Enc': var_enc,
                'Grade_Enc': grd_enc,
                **lag_feats,
                'Month_sin': np.sin(2 * np.pi * month / 12.0),
                'Month_cos': np.cos(2 * np.pi * month / 12.0),
                'DOY_sin': np.sin(2 * np.pi * doy / 365.25),
                'DOY_cos': np.cos(2 * np.pi * doy / 365.25),
                'Perishability_Tier': perishability_tier,
                'Fuel_Cost_Index': fuel["index"],
                'Temp_Proxy': weather.get("temperature", 28.0),
                'Humidity_Proxy': weather.get("humidity", 65.0),
                'Precipitation_Proxy': weather.get("precipitation_mm", 0.0),
                'Dew_Point_Proxy': weather.get(
                    "dew_point",
                    _dew_point(weather.get("temperature", 28.0), weather.get("humidity", 65.0))
                ),
            }

            feat_df = pd.DataFrame([feature_dict])[encoders['feature_cols']]
            feat_scaled = scaler.transform(feat_df)
            modal_price_val = float(model.predict(feat_scaled)[0])
        except Exception as e:
            print(f"Prediction inference exception: {e}")
            modal_price_val = base_p
    else:
        # Fallback price calculation
        base_map = {"Tomato": 2800, "Onion": 3200, "Potato": 1950, "Rice": 3900, "Wheat": 2650, "Apple": 8600}
        modal_price_val = float(base_p if last_observed else base_map.get(req.commodity, 3100.0))

    modal_price_val = round(max(400.0, modal_price_val), 2)
    kg_price_val = round(modal_price_val / 100.0, 2)

    # 7-day confidence interval: real series drift + historical volatility
    stats = None
    if series is not None:
        stats = _price_stats(_series_as_of(series, target_date))
    if stats is None:
        stats = {"momentum_pct": 0.0, "daily_vol": 0.015, "daily_drift": 0.001,
                 "last_price": modal_price_val, "observations": 0}

    forecast_7d, mom7_pct = build_7d_outlook(modal_price_val, stats, target_date)

    direction = "up" if mom7_pct > 0.75 else ("down" if mom7_pct < -0.75 else "stable")
    direction_hi = {"up": "चढ़ता", "down": "उतरता", "stable": "स्थिर"}[direction]
    direction_en = {"up": "Rising", "down": "Falling", "stable": "Stable"}[direction]
    color = {"up": "green", "down": "red", "stable": "amber"}[direction]

    advisory_data = PERISHABILITY_ADVISORY.get(perishability_tier, PERISHABILITY_ADVISORY[2])

    market_insight = {
        "en": (f"{req.commodity} at {mandi_info['market']} is trending {direction_en.lower()} "
               f"({mom7_pct:+.1f}% over 7 days). Weather: {weather.get('condition', '')}. "
               f"{advisory_data['en']['risk_label']}."),
        "hi": (f"{mandi_info['market']} में {req.commodity} का रुझान {direction_hi} है "
               f"(7 दिनों में {mom7_pct:+.1f}%)। मौसम: {weather.get('condition_hi', '')}। "
               f"{advisory_data['hi']['risk_label']}।"),
    }

    return {
        "status": "success",
        "market": mandi_info["market"],
        "state": mandi_info["state"],
        "district": mandi_info["district"],
        "commodity": req.commodity,
        "arrival_date": target_date.strftime("%Y-%m-%d"),
        "modal_price": modal_price_val,
        "kg_price": kg_price_val,
        "currency": "₹",
        "unit_quintal": "Quintal (100 Kg)",
        "unit_kg": "Kg",
        "perishability_tier": perishability_tier,
        "advisory": advisory_data,
        "live_weather": weather,
        "weather_impact": {"en": weather.get("condition"), "hi": weather.get("condition_hi")},
        "price_momentum": {
            "direction": direction,
            "color": color,
            "pct_7d": mom7_pct,
            "label": f"{mom7_pct:+.1f}%",
            "label_en": f"{direction_en} ({mom7_pct:+.1f}%)",
            "label_hi": f"{direction_hi} ({mom7_pct:+.1f}%)",
        },
        "market_insight": market_insight,
        "fuel_transport_index": fuel,
        "last_observed_price": round(float(last_observed), 2) if last_observed else None,
        "history_days": history_days,
        "confidence_range_7d": forecast_7d,
        "model_version": "AgriForecast AI v2.0 (Weighted Stacking: XGBoost + LightGBM + CatBoost)"
    }

# Mount static directory for Frontend single page app
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
