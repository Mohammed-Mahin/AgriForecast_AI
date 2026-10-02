import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

os.makedirs('dataset', exist_ok=True)
os.makedirs('model_artifacts', exist_ok=True)

# List of major Indian Agricultural Mandis with coordinates & details
MANDIS = [
    {"state": "Delhi", "district": "Delhi", "market": "Azadpur Mandi", "lat": 28.716, "lng": 77.170},
    {"state": "Maharashtra", "district": "Mumbai", "market": "Vashi APMC Mandi", "lat": 19.076, "lng": 73.008},
    {"state": "Maharashtra", "district": "Nashik", "market": "Lasalgaon Mandi", "lat": 20.147, "lng": 74.227},
    {"state": "Karnataka", "district": "Kolar", "market": "Kolar Mandi", "lat": 13.136, "lng": 78.129},
    {"state": "Karnataka", "district": "Chamarajanagar", "market": "Gundlupet Mandi", "lat": 11.808, "lng": 76.690},
    {"state": "Rajasthan", "district": "Jaipur", "market": "Jaipur Mandi", "lat": 26.912, "lng": 75.787},
    {"state": "Andhra Pradesh", "district": "Guntur", "market": "Guntur Mandi", "lat": 16.306, "lng": 80.436},
    {"state": "West Bengal", "district": "Kolkata", "market": "Kolkata Mandi", "lat": 22.572, "lng": 88.363},
    {"state": "Telangana", "district": "Mahbubnagar", "market": "Mahbubnagar Mandi", "lat": 16.748, "lng": 77.984},
    {"state": "Gujarat", "district": "Ahmedabad", "market": "Ahmedabad Mandi", "lat": 23.022, "lng": 72.571},
    {"state": "Punjab", "district": "Amritsar", "market": "Amritsar Mandi", "lat": 31.634, "lng": 74.872},
    {"state": "Madhya Pradesh", "district": "Indore", "market": "Indore Mandi", "lat": 22.719, "lng": 75.857},
]

COMMODITIES_INFO = {
    # Tier 3 (High Perishability)
    "Tomato": {"tier": 3, "base_price": 2800, "volatility": 0.28, "variety": "Hybrid", "grade": "FAQ", "shelf_days": 4},
    "Onion": {"tier": 3, "base_price": 3100, "volatility": 0.25, "variety": "Red Nashik", "grade": "FAQ", "shelf_days": 5},
    "Leafy Greens": {"tier": 3, "base_price": 1800, "volatility": 0.32, "variety": "Spinach/Coriander", "grade": "FAQ", "shelf_days": 3},
    "Green Peas": {"tier": 3, "base_price": 4500, "volatility": 0.22, "variety": "Local Fresh", "grade": "FAQ", "shelf_days": 4},
    "Berries": {"tier": 3, "base_price": 12000, "volatility": 0.35, "variety": "Strawberry", "grade": "FAQ", "shelf_days": 3},

    # Tier 2 (Medium Perishability)
    "Potato": {"tier": 2, "base_price": 1900, "volatility": 0.15, "variety": "Jyoti", "grade": "FAQ", "shelf_days": 12},
    "Ginger": {"tier": 2, "base_price": 7500, "volatility": 0.18, "variety": "Fresh", "grade": "FAQ", "shelf_days": 14},
    "Garlic": {"tier": 2, "base_price": 11000, "volatility": 0.20, "variety": "Desi", "grade": "FAQ", "shelf_days": 15},
    "Apple": {"tier": 2, "base_price": 8500, "volatility": 0.16, "variety": "Royal Delicious", "grade": "Grade A", "shelf_days": 14},
    "Mango": {"tier": 2, "base_price": 6200, "volatility": 0.24, "variety": "Alphonso", "grade": "Grade A", "shelf_days": 7},
    "Citrus": {"tier": 2, "base_price": 3800, "volatility": 0.14, "variety": "Nagpur Orange", "grade": "FAQ", "shelf_days": 10},

    # Tier 1 (Low Perishability)
    "Rice": {"tier": 1, "base_price": 3900, "volatility": 0.08, "variety": "Sona Masoori", "grade": "FAQ", "shelf_days": 365},
    "Wheat": {"tier": 1, "base_price": 2600, "volatility": 0.07, "variety": "Lokwan", "grade": "FAQ", "shelf_days": 365},
    "Pulses": {"tier": 1, "base_price": 7200, "volatility": 0.10, "variety": "Tur/Arhar", "grade": "FAQ", "shelf_days": 365},
    "Turmeric": {"tier": 1, "base_price": 13500, "volatility": 0.12, "variety": "Finger", "grade": "Grade A", "shelf_days": 365},
    "Cotton": {"tier": 1, "base_price": 7100, "volatility": 0.09, "variety": "Medium Staple", "grade": "FAQ", "shelf_days": 365},
    "Mustard": {"tier": 1, "base_price": 5400, "volatility": 0.11, "variety": "Yellow", "grade": "FAQ", "shelf_days": 365},
}

def generate_historical_dataset():
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 10, 1)
    total_days = (end_date - start_date).days + 1
    
    date_list = [start_date + timedelta(days=i) for i in range(total_days)]

    records = []
    np.random.seed(42)

    for mandi in MANDIS:
        # Market-specific pricing factor
        market_factor = 1.0 + np.random.uniform(-0.08, 0.08)

        for commodity, info in COMMODITIES_INFO.items():
            base_p = info["base_price"] * market_factor
            vol = info["volatility"]
            
            # Long-term macroeconomic inflation trend (approx 4% yearly)
            # Seasonal harmonic wave based on day of year
            for i, dt in enumerate(date_list):
                date_str = dt.strftime("%Y-%m-%d")
                doy = dt.timetuple().tm_yday
                
                # Seasonality effect (harvest vs off-season)
                seasonal_wave = np.sin(2 * np.pi * (doy - 45) / 365.25) * vol * base_p * 0.7
                # Random noise / short term shocks
                shock = np.random.normal(0, vol * 0.3 * base_p)
                # Trend over time (2024 -> 2026)
                trend = (i / 365.25) * 0.04 * base_p

                modal_price = max(500, round(base_p + seasonal_wave + shock + trend, 2))
                
                # Introduce occasional realistic minor outliers to test IQR cleaner
                if np.random.rand() < 0.005:
                    modal_price = round(modal_price * np.random.choice([0.25, 3.2]), 2)
                
                min_price = round(modal_price * np.random.uniform(0.88, 0.96), 2)
                max_price = round(modal_price * np.random.uniform(1.04, 1.15), 2)

                records.append({
                    "State": mandi["state"],
                    "District Name": mandi["district"],
                    "Market Name": mandi["market"],
                    "Commodity": commodity,
                    "Variety": info["variety"],
                    "Grade": info["grade"],
                    "Min_Price": min_price,
                    "Max_Price": max_price,
                    "Modal_Price": modal_price,
                    "Arrival_Date": date_str
                })

    df = pd.DataFrame(records)
    output_path = os.path.join('dataset', 'agmarknet_india_historical_prices_2024_2026.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated dataset with {len(df)} records at {output_path}")

if __name__ == "__main__":
    generate_historical_dataset()
