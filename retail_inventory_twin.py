"""
Retail Inventory Twin
=====================
1. Load CSV → Pandas DataFrame
2. Fetch historical weather from OpenWeatherMap & join
3. Train Prophet model for one category
4. Predict next 30 days & visualise forecast
"""

import sqlite3
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────
CSV_PATH        = "inventory_demand_data.csv"   # path to your CSV file
CSV_PATH_2024   = "inventory_demand_data_2024.csv"  # 2024 data for combining
OWM_API_KEY     = "e1d69092dc4501736ab7604f73ecf233"
CITY            = "San Jose,US"            # city for weather lookup
TARGET_CATEGORY = "Hot Beverages"          # category to forecast
DB_PATH         = "inventory.db"          # SQLite file path


# ─────────────────────────────────────────────
# 1. LOAD & COMBINE CSV → DATAFRAME
# ─────────────────────────────────────────────
def load_sales(path: str) -> pd.DataFrame:
    """Load and combine 2024 and 2025 data for 2-year training dataset."""
    # Load 2024 data
    df_2024 = pd.read_csv(CSV_PATH_2024, parse_dates=["ds"])
    df_2024 = df_2024.rename(columns={"ds": "Date"})
    
    # Load 2025 data
    df_2025 = pd.read_csv(path, parse_dates=["Date"])
    
    # Combine both years
    df_combined = pd.concat([df_2024, df_2025], ignore_index=True)
    df_combined.columns = df_combined.columns.str.strip().str.lower().str.replace(" ", "_")
    df_combined = df_combined.sort_values("date").reset_index(drop=True)
    
    print(f"✅ Loaded {len(df_combined):,} rows | {df_combined['date'].min().date()} → {df_combined['date'].max().date()}")
    print(f"   📅 2024 data: {len(df_2024):,} rows")
    print(f"   📅 2025 data: {len(df_2025):,} rows")
    return df_combined


# ─────────────────────────────────────────────
# 2. FETCH HISTORICAL WEATHER (OWM free tier)
#    Smart Caching: Check cache first, only fetch missing dates from API
# ─────────────────────────────────────────────
def fetch_weather_range(api_key: str, city: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch historical weather data from OpenWeatherMap API with smart caching.
    Checks historical_weather_data.csv first and only fetches missing dates from API.
    """
    # 1. Load existing historical weather data if available
    existing_weather = pd.DataFrame(columns=["date", "owm_temp_f", "owm_humidity_pct", "owm_weather"])
    if os.path.exists("historical_weather_data.csv"):
        try:
            existing_weather = pd.read_csv("historical_weather_data.csv")
            existing_weather["date"] = pd.to_datetime(existing_weather["date"]).dt.date
            print(f"📚 Loaded existing weather data ({len(existing_weather)} records)")
        except Exception as e:
            print(f"⚠️  Could not load existing weather: {e}")
    
    # 2. Identify date range that needs fetching
    date_range = pd.date_range(start, end, freq="D")
    existing_dates = set(existing_weather["date"].unique()) if not existing_weather.empty else set()
    missing_dates = [d for d in date_range if d.date() not in existing_dates]
    
    if not missing_dates:
        print(f"✅ All weather data available in historical cache ({len(existing_weather)} records)")
        existing_weather["date"] = pd.to_datetime(existing_weather["date"])
        return existing_weather
    
    print(f"📡 Fetching weather for {len(missing_dates)} missing dates (using cache for {len(existing_dates)} dates)")
    
    try:
        # Step 1 – resolve lat/lon once
        geo_url = "http://api.openweathermap.org/geo/1.0/direct"
        geo = requests.get(geo_url, params={"q": city, "limit": 1, "appid": api_key}, timeout=5).json()
        if not geo:
            print(f"⚠️  City '{city}' not found. Using cached data.")
            if not existing_weather.empty:
                existing_weather["date"] = pd.to_datetime(existing_weather["date"])
                return existing_weather
            return pd.DataFrame(columns=["date", "owm_temp_f", "owm_humidity_pct", "owm_weather"])
        
        lat, lon = geo[0]["lat"], geo[0]["lon"]
        print(f"📍 {city} → lat={lat:.4f}, lon={lon:.4f}")
        
        # Step 2 – fetch only missing dates
        new_records = []
        timemachine_url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
        
        for missing_date in missing_dates:
            ts = int(missing_date.timestamp())
            try:
                resp = requests.get(timemachine_url, params={
                    "lat": lat, "lon": lon,
                    "dt": ts,
                    "appid": api_key,
                    "units": "imperial"
                }, timeout=5)
                
                if resp.status_code == 200:
                    day_data = resp.json().get("current", {}) if "current" in resp.json() else resp.json().get("data", [{}])[0]
                    new_records.append({
                        "date":              missing_date.date(),
                        "owm_temp_f":        day_data.get("temp"),
                        "owm_humidity_pct":  day_data.get("humidity"),
                        "owm_weather":       day_data.get("weather", [{}])[0].get("main", "") if day_data.get("weather") else "",
                    })
                else:
                    print(f"  ⚠️  {missing_date.date()} — OWM returned {resp.status_code}")
            except Exception as e:
                print(f"  ⚠️  Error fetching {missing_date.date()}: {e}")
                continue
        
        if new_records:
            new_df = pd.DataFrame(new_records)
            print(f"🌤  Fetched {len(new_df)} new records from API")
            
            # Combine with existing data
            if not existing_weather.empty:
                combined = pd.concat([existing_weather, new_df], ignore_index=True).drop_duplicates(subset=["date"])
            else:
                combined = new_df
            
            combined["date"] = pd.to_datetime(combined["date"])
            return combined
        else:
            print(f"⚠️  No new weather data from API. Using cached data.")
            if not existing_weather.empty:
                existing_weather["date"] = pd.to_datetime(existing_weather["date"])
                return existing_weather
            return pd.DataFrame(columns=["date", "owm_temp_f", "owm_humidity_pct", "owm_weather"])
    
    except Exception as e:
        print(f"⚠️  Error fetching weather: {e}")
        print(f"   Using cached weather data ({len(existing_weather)} records)")
        if not existing_weather.empty:
            existing_weather["date"] = pd.to_datetime(existing_weather["date"])
            return existing_weather
        return pd.DataFrame(columns=["date", "owm_temp_f", "owm_humidity_pct", "owm_weather"])



# ─────────────────────────────────────────────
# 3. MERGE SALES + WEATHER → SQLITE
# ─────────────────────────────────────────────
def build_joined_db(sales_df: pd.DataFrame, weather_df: pd.DataFrame, db_path: str) -> pd.DataFrame:
    """
    Merge sales data with weather data, extracting weather from CSV if API data is unavailable.
    Save weather data to CSV file for reference.
    """
    # If API weather is empty, extract from existing CSV data
    if weather_df.empty:
        print("📝 Extracting weather data from sales CSV...")
        weather_from_csv = sales_df[["date", "temperature_f", "humidity_pct", "weather_condition"]].drop_duplicates()
        weather_from_csv = weather_from_csv.rename(columns={
            "temperature_f": "owm_temp_f",
            "humidity_pct": "owm_humidity_pct",
            "weather_condition": "owm_weather"
        })
        weather_df = weather_from_csv.copy()
    
    # Save weather data to CSV
    weather_output = weather_df.copy()
    weather_output["date"] = pd.to_datetime(weather_output["date"])
    weather_output = weather_output.sort_values("date").reset_index(drop=True)
    weather_output.to_csv("historical_weather_data.csv", index=False)
    print(f"💾 Saved weather data → historical_weather_data.csv ({len(weather_output)} records)")
    
    # Merge sales + weather
    merged = sales_df.merge(weather_df, on="date", how="left")

    # Prefer OWM columns; fallback to dataset columns when OWM is missing
    merged["final_temp_f"]       = merged["owm_temp_f"].fillna(merged.get("temperature_f", float("nan")))
    merged["final_humidity_pct"] = merged["owm_humidity_pct"].fillna(merged.get("humidity_pct", float("nan")))

    con = sqlite3.connect(db_path)
    merged.to_sql("inventory", con, if_exists="replace", index=False)
    con.close()
    print(f"💾 Saved {len(merged):,} rows → {db_path}")
    return merged


# ─────────────────────────────────────────────
# 4. PROPHET MODEL FOR ONE CATEGORY per RETAILER
# ─────────────────────────────────────────────
def train_and_forecast(df: pd.DataFrame, category: str, retailer_id: str = None, forecast_days: int = 30):
    """
    Train Prophet model for a specific category and optionally filter by retailer.
    If retailer_id is None, trains on all retailers combined.
    """
    if retailer_id:
        cat_df = (
            df[(df["category"].str.lower() == category.lower()) & 
               (df["retailer_id"] == retailer_id)]
            .groupby("date", as_index=False)["sales_volume"]
            .sum()
            .rename(columns={"date": "ds", "sales_volume": "y"})
            .sort_values("ds")
        )
        label = f"{category} @ {retailer_id}"
    else:
        cat_df = (
            df[df["category"].str.lower() == category.lower()]
            .groupby("date", as_index=False)["sales_volume"]
            .sum()
            .rename(columns={"date": "ds", "sales_volume": "y"})
            .sort_values("ds")
        )
        label = category
    
    print(f"\n📦 Training on '{label}' — {len(cat_df)} daily observations")

    # Add weather regressors if available
    has_weather = "final_temp_f" in df.columns and df["final_temp_f"].notna().any()

    if has_weather:
        if retailer_id:
            weather_agg = (
                df[(df["category"].str.lower() == category.lower()) & 
                   (df["retailer_id"] == retailer_id)]
                .groupby("date", as_index=False)[["final_temp_f", "final_humidity_pct"]]
                .mean()
                .rename(columns={"date": "ds"})
            )
        else:
            weather_agg = (
                df[df["category"].str.lower() == category.lower()]
                .groupby("date", as_index=False)[["final_temp_f", "final_humidity_pct"]]
                .mean()
                .rename(columns={"date": "ds"})
            )
        cat_df = cat_df.merge(weather_agg, on="ds", how="left")

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
    )

    if has_weather:
        m.add_regressor("final_temp_f")
        m.add_regressor("final_humidity_pct")

    m.fit(cat_df)

    # Build future dataframe
    future = m.make_future_dataframe(periods=forecast_days)

    if has_weather:
        # For future dates we extrapolate regressors using last 30-day rolling mean
        last_temp     = cat_df["final_temp_f"].iloc[-30:].mean()
        last_humidity = cat_df["final_humidity_pct"].iloc[-30:].mean()
        
        # Merge with historical weather data
        future = future.merge(cat_df[["ds", "final_temp_f", "final_humidity_pct"]], on="ds", how="left")
        
        # Fill NaN values using assignment (not inplace) to avoid ChainedAssignmentError
        # Use last 30-day mean for future dates, or 0 as fallback if mean is also NaN
        future["final_temp_f"] = future["final_temp_f"].fillna(last_temp if pd.notna(last_temp) else 0)
        future["final_humidity_pct"] = future["final_humidity_pct"].fillna(last_humidity if pd.notna(last_humidity) else 50)

    forecast = m.predict(future)
    return m, forecast, cat_df


# ─────────────────────────────────────────────
# 5. VISUALISE
# ─────────────────────────────────────────────
def visualise(m, forecast, cat_df, category: str):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f"Retail Inventory Twin — '{category}' Demand Forecast", fontsize=15, fontweight="bold")

    # — Panel 1: full forecast —
    ax = axes[0]
    history_mask = forecast["ds"].isin(cat_df["ds"])
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                    alpha=0.25, color="steelblue", label="95% CI")
    ax.plot(forecast["ds"], forecast["yhat"], color="steelblue", lw=1.8, label="Forecast")
    ax.scatter(cat_df["ds"], cat_df["y"], s=8, color="black", alpha=0.5, label="Actual sales")
    # Shade the future window
    split = cat_df["ds"].max()
    ax.axvline(split, color="red", ls="--", lw=1, label="Forecast start")
    ax.set_title("Daily Sales Volume — History + 30-day Forecast")
    ax.set_ylabel("Units sold")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # — Panel 2: trend component —
    ax2 = axes[1]
    ax2.plot(forecast["ds"], forecast["trend"], color="darkorange", lw=1.8)
    ax2.set_title("Trend Component")
    ax2.set_ylabel("Trend")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # — Panel 3: weekly seasonality —
    ax3 = axes[2]
    weekly_cols = [c for c in forecast.columns if c.startswith("weekly")]
    if weekly_cols:
        # Extract one representative week from the middle of the dataset
        mid = len(forecast) // 2
        week_slice = forecast.iloc[mid:mid+7].copy()
        week_slice["dow"] = week_slice["ds"].dt.day_name()
        ax3.bar(week_slice["dow"], week_slice[weekly_cols[0]], color="mediumseagreen")
        ax3.set_title("Weekly Seasonality Effect")
        ax3.set_ylabel("Effect on sales")
        ax3.axhline(0, color="black", lw=0.8)
    else:
        ax3.text(0.5, 0.5, "Weekly seasonality not available", ha="center", transform=ax3.transAxes)

    plt.tight_layout()
    out = f"charts/forecast_{category.replace(' ', '_').lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n📊 Chart saved → {out}")
    plt.show()


# ─────────────────────────────────────────────
# 6. STOCK-OUT ESTIMATION
# ─────────────────────────────────────────────
def estimate_stockout(forecast: pd.DataFrame, cat_df: pd.DataFrame,
                      current_stock: float, category: str, retailer_id: str = None):
    """
    Estimate stock-out based on cumulative demand forecast.
    """
    future_fc = forecast[forecast["ds"] > cat_df["ds"].max()][["ds", "yhat"]].copy()
    future_fc["cumulative_demand"] = future_fc["yhat"].clip(lower=0).cumsum()
    stockout_row = future_fc[future_fc["cumulative_demand"] >= current_stock]

    label = f"{category} @ {retailer_id}" if retailer_id else category
    
    print(f"\n{'─'*60}")
    print(f"📦 Category & Retailer: {label}")
    print(f"📦 Current stock      : {current_stock:,.0f} units")
    if stockout_row.empty:
        days_left = len(future_fc)
        print(f"✅ Stock sufficient for all {days_left} forecast days.")
    else:
        row = stockout_row.iloc[0]
        days = (row["ds"] - cat_df["ds"].max()).days
        print(f"⚠️  Predicted stock-out in {days} day(s) (on {row['ds'].date()})")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Create charts folder if it doesn't exist
    os.makedirs("charts", exist_ok=True)
    
    # 1. Load
    sales_df = load_sales(CSV_PATH)

    # 2. Weather
    start_dt = sales_df["date"].min().to_pydatetime()
    end_dt   = sales_df["date"].max().to_pydatetime()
    weather_df = fetch_weather_range(OWM_API_KEY, CITY, start_dt, end_dt)

    # 3. Merge & persist
    merged_df = build_joined_db(sales_df, weather_df, DB_PATH)

    # 4. Get unique retailers for target category
    category_data = merged_df[merged_df["category"].str.lower() == TARGET_CATEGORY.lower()]
    retailers = sorted(category_data["retailer_id"].unique())
    
    print(f"\n{'='*60}")
    print(f"🏪 Forecasting for {len(retailers)} retailers in '{TARGET_CATEGORY}'")
    print(f"   Retailers: {', '.join(retailers)}")
    print(f"{'='*60}")
    
    # 5. Train and forecast for each retailer
    stockout_summary = []
    
    for retailer_id in retailers:
        # Get current stock for this retailer-category combination
        retailer_cat_data = category_data[category_data["retailer_id"] == retailer_id]
        current_stock = retailer_cat_data["stock_quantity"].iloc[-1] if len(retailer_cat_data) > 0 else 100
        
        # Train model
        model, forecast, cat_df = train_and_forecast(
            merged_df, 
            TARGET_CATEGORY, 
            retailer_id=retailer_id, 
            forecast_days=30
        )
        
        # Visualise
        visualise(model, forecast, cat_df, f"{TARGET_CATEGORY} @ {retailer_id}")
        
        # Stock-out estimate
        estimate_stockout(forecast, cat_df, current_stock=current_stock, 
                         category=TARGET_CATEGORY, retailer_id=retailer_id)
        
        # Store summary
        future_fc = forecast[forecast["ds"] > cat_df["ds"].max()][["ds", "yhat"]].copy()
        future_fc["cumulative_demand"] = future_fc["yhat"].clip(lower=0).cumsum()
        stockout_row = future_fc[future_fc["cumulative_demand"] >= current_stock]
        
        if not stockout_row.empty:
            stockout_date = stockout_row.iloc[0]["ds"].date()
            days_until = (stockout_date - cat_df["ds"].max().date()).days
        else:
            stockout_date = None
            days_until = None
        
        stockout_summary.append({
            "retailer": retailer_id,
            "category": TARGET_CATEGORY,
            "current_stock": current_stock,
            "stockout_date": stockout_date,
            "days_until_stockout": days_until
        })
    
    # 6. Print summary table
    print(f"\n{'='*60}")
    print(f"📊 STOCK-OUT SUMMARY")
    print(f"{'='*60}\n")
    
    summary_df = pd.DataFrame(stockout_summary)
    print(summary_df.to_string(index=False))
    print(f"\n{'='*60}\n")
