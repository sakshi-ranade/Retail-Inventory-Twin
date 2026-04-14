"""
Fine-Tuned Retail Inventory Twin — Prophet Model v2 (Clean Slate)
==================================================================
Simplified, Stable Approach:
1. Holiday & Promotion Regressors ✅
2. External Regressors (Weather: Temp, Humidity) ✅
3. Stable Baseline Trend (changepoint_prior_scale=0.01) ✅
4. Linear Growth (simple, predictable) ✅
5. Log-transformation (handles heteroscedasticity) ✅
6. Standardized Weather Features (balanced influence) ✅
"""

# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import sqlite3
import requests
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CSV_PATH        = "inventory_demand_data.csv"
CSV_PATH_2024   = "inventory_demand_data_2024.csv"
OWM_API_KEY     = " PASTE API KEY HERE"
CITY            = "San Jose,US"
TARGET_CATEGORY = "Frozen"
DB_PATH         = "inventory.db"


def load_sales(path: str) -> pd.DataFrame:
    df_2024 = pd.read_csv(CSV_PATH_2024, parse_dates=["ds"])
    df_2024 = df_2024.rename(columns={"ds": "Date"})
    df_2025 = pd.read_csv(path, parse_dates=["Date"])
    df_combined = pd.concat([df_2024, df_2025], ignore_index=True)
    df_combined.columns = df_combined.columns.str.strip().str.lower().str.replace(" ", "_")
    df_combined = df_combined.sort_values("date").reset_index(drop=True)
    print(f"✅ Loaded {len(df_combined):,} rows | {df_combined['date'].min().date()} → {df_combined['date'].max().date()}")
    print(f"   📅 2024 data: {len(df_2024):,} rows")
    print(f"   📅 2025 data: {len(df_2025):,} rows")
    return df_combined


def add_holiday_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_holiday"] = 0
    df["is_promotion"] = 0
    holidays = {
        (1, 1): "New Year",
        (2, 14): "Valentine's Day",
        (3, 17): "St. Patrick's Day",
        (5, 27): "Memorial Day",
        (7, 4): "Independence Day",
        (11, 28): "Black Friday",
        (11, 29): "Cyber Monday",
        (12, 25): "Christmas",
        (12, 26): "Boxing Day/Post-Christmas Sales",
        (12, 31): "New Year's Eve",
    }
    for (month, day), name in holidays.items():
        mask = (df["date"].dt.month == month) & (df["date"].dt.day == day)
        df.loc[mask, "is_holiday"] = 1
        if name in ["New Year", "Black Friday", "Cyber Monday", "Christmas", "Boxing Day/Post-Christmas Sales", "New Year's Eve"]:
            df.loc[mask, "is_promotion"] = 1
    mask = (df["date"].dt.month == 12) & (df["date"].dt.day >= 26)
    df.loc[mask, "is_promotion"] = 1
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_cold"] = (df["final_temp_f"] < 50).astype(int)
    df["is_hot"] = (df["final_temp_f"] > 75).astype(int)
    df["is_humid"] = (df["final_humidity_pct"] > 70).astype(int)
    df["cold_weekend_interaction"] = df["is_cold"] * df["is_weekend"]
    df["hot_weekend_interaction"] = df["is_hot"] * df["is_weekend"]
    df["humid_cold_interaction"] = df["is_humid"] * df["is_cold"]
    return df


def fetch_weather_range(api_key: str, city: str, start: datetime, end: datetime) -> pd.DataFrame:
    existing_weather = pd.DataFrame(columns=["date", "owm_temp_f", "owm_humidity_pct", "owm_weather"])
    if os.path.exists("historical_weather_data.csv"):
        try:
            existing_weather = pd.read_csv("historical_weather_data.csv")
            existing_weather["date"] = pd.to_datetime(existing_weather["date"]).dt.date
            print(f"📚 Loaded existing weather data ({len(existing_weather)} records)")
        except Exception as e:
            print(f"⚠️  Could not load existing weather: {e}")
    date_range = pd.date_range(start, end, freq="D")
    existing_dates = set(existing_weather["date"].unique()) if not existing_weather.empty else set()
    missing_dates = [d for d in date_range if d.date() not in existing_dates]
    if not missing_dates:
        print(f"✅ All weather data available in historical cache ({len(existing_weather)} records)")
        existing_weather["date"] = pd.to_datetime(existing_weather["date"])
        return existing_weather
    print(f"📡 Fetching weather for {len(missing_dates)} missing dates (using cache for {len(existing_dates)} dates)")
    try:
        geo_url = "http://api.openweathermap.org/geo/1.0/direct"
        geo = requests.get(geo_url, params={"q": city, "limit": 1, "appid": api_key}, timeout=5).json()
        if not geo:
            print(f"⚠️  City not found. Using existing cached data.")
            if not existing_weather.empty:
                existing_weather["date"] = pd.to_datetime(existing_weather["date"])
                return existing_weather
            return pd.DataFrame(columns=["date", "owm_temp_f", "owm_humidity_pct", "owm_weather"])
        lat, lon = geo[0]["lat"], geo[0]["lon"]
        print(f"📍 {city} → lat={lat:.4f}, lon={lon:.4f}")
        new_records = []
        timemachine_url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
        for missing_date in missing_dates:
            ts = int(missing_date.timestamp())
            try:
                resp = requests.get(timemachine_url, params={
                    "lat": lat, "lon": lon, "dt": ts, "appid": api_key, "units": "imperial"
                }, timeout=5)
                if resp.status_code == 200:
                    day_data = resp.json().get("current", {}) if "current" in resp.json() else resp.json().get("data", [{}])[0]
                    new_records.append({
                        "date": missing_date.date(),
                        "owm_temp_f": day_data.get("temp"),
                        "owm_humidity_pct": day_data.get("humidity"),
                        "owm_weather": day_data.get("weather", [{}])[0].get("main", "") if day_data.get("weather") else "",
                    })
            except Exception as e:
                print(f"⚠️  Error fetching {missing_date.date()}: {e}")
                continue
        if new_records:
            new_df = pd.DataFrame(new_records)
            print(f"🌤  Fetched {len(new_df)} new records from API")
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
        if not existing_weather.empty:
            existing_weather["date"] = pd.to_datetime(existing_weather["date"])
            return existing_weather
        return pd.DataFrame(columns=["date", "owm_temp_f", "owm_humidity_pct", "owm_weather"])


def build_joined_db(sales_df: pd.DataFrame, weather_df: pd.DataFrame, db_path: str) -> pd.DataFrame:
    if weather_df.empty:
        print("📝 Extracting weather from sales CSV...")
        weather_from_csv = sales_df[["date", "temperature_f", "humidity_pct", "weather_condition"]].drop_duplicates()
        weather_from_csv = weather_from_csv.rename(columns={
            "temperature_f": "owm_temp_f",
            "humidity_pct": "owm_humidity_pct",
            "weather_condition": "owm_weather"
        })
        weather_df = weather_from_csv.copy()
    weather_output = weather_df.copy()
    weather_output["date"] = pd.to_datetime(weather_output["date"])
    weather_output = weather_output.sort_values("date").reset_index(drop=True)
    weather_output.to_csv("historical_weather_data.csv", index=False)
    print(f"💾 Saved weather data → historical_weather_data.csv ({len(weather_output)} records)")
    merged = sales_df.merge(weather_df, on="date", how="left")
    merged["final_temp_f"] = merged["owm_temp_f"].fillna(merged.get("temperature_f", float("nan")))
    merged["final_humidity_pct"] = merged["owm_humidity_pct"].fillna(merged.get("humidity_pct", float("nan")))
    merged = add_holiday_flags(merged)
    merged = add_interaction_features(merged)
    con = sqlite3.connect(db_path)
    merged.to_sql("inventory", con, if_exists="replace", index=False)
    con.close()
    print(f"💾 Saved {len(merged):,} rows → {db_path}")
    return merged


def standardize_weather_features(train_df: pd.DataFrame, future_df: pd.DataFrame) -> tuple:
    scaler = StandardScaler()
    weather_cols = ["final_temp_f", "final_humidity_pct"]
    train_df[weather_cols] = scaler.fit_transform(train_df[weather_cols])
    future_df[weather_cols] = scaler.transform(future_df[weather_cols])
    return train_df, future_df, scaler


def train_and_forecast_v2(df: pd.DataFrame, category: str, retailer_id: str = None, forecast_days: int = 30):
    if retailer_id:
        cat_df = (
            df[(df["category"].str.lower() == category.lower()) &
               (df["retailer_id"] == retailer_id)]
            .groupby("date", as_index=False)
            .agg({
                "sales_volume": "sum",
                "final_temp_f": "mean",
                "final_humidity_pct": "mean",
                "is_holiday": "max",
                "is_promotion": "max"
            })
            .rename(columns={"date": "ds", "sales_volume": "y"})
            .sort_values("ds")
        )
        label = f"{category} @ {retailer_id}"
    else:
        cat_df = (
            df[df["category"].str.lower() == category.lower()]
            .groupby("date", as_index=False)
            .agg({
                "sales_volume": "sum",
                "final_temp_f": "mean",
                "final_humidity_pct": "mean",
                "is_holiday": "max",
                "is_promotion": "max"
            })
            .rename(columns={"date": "ds", "sales_volume": "y"})
            .sort_values("ds")
        )
        label = category

    training_cutoff = pd.Timestamp("2025-12-01")
    cat_df_train = cat_df[cat_df["ds"] < training_cutoff].copy()

    print(f"\n📦 Training on '{label}' — {len(cat_df_train)} daily observations (Jan 2024 - Nov 2025)")
    print(f"   Date range: {cat_df_train['ds'].min().date()} → {cat_df_train['ds'].max().date()}")

    max_sales = cat_df_train["y"].max()
    capacity_cap = max_sales * 1.5
    print(f"   Max historical sales: {max_sales:.0f} units")
    print(f"   Setting physical cap: {capacity_cap:.0f} units (1.5x max)")
    m = Prophet(
        growth="logistic",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.01,
        interval_width=0.95
    )
    m.add_regressor("final_temp_f")
    m.add_regressor("final_humidity_pct")
    m.add_regressor("is_holiday")
    m.add_regressor("is_promotion")
    print(f"   Fitting model with logistic growth and capacity cap...")
    train_df = cat_df_train[["ds", "y", "final_temp_f", "final_humidity_pct",
                       "is_holiday", "is_promotion"]].copy()
    train_df["cap"] = capacity_cap
    train_df["floor"] = 0
    m.fit(train_df)

    last_train_date = cat_df_train["ds"].max()
    dec_31 = pd.Timestamp("2025-12-31")
    future_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1), end=dec_31, freq="D")
    future = pd.DataFrame({"ds": future_dates})

    future["cap"] = capacity_cap
    future["floor"] = 0
    last_temp = cat_df_train["final_temp_f"].iloc[-30:].mean()
    last_humidity = cat_df_train["final_humidity_pct"].iloc[-30:].mean()
    last_promotion = cat_df_train["is_promotion"].iloc[-7:].mean()
    future = future.merge(cat_df_train[["ds", "final_temp_f", "final_humidity_pct", "is_holiday",
                                  "is_promotion"]],
                         on="ds", how="left")
    future["final_temp_f"] = future["final_temp_f"].fillna(last_temp if pd.notna(last_temp) else 0)
    future["final_humidity_pct"] = future["final_humidity_pct"].fillna(last_humidity if pd.notna(last_humidity) else 50)
    future["is_holiday"] = future["is_holiday"].fillna(0)
    future["is_promotion"] = future["is_promotion"].fillna(last_promotion if pd.notna(last_promotion) else 0)
    forecast = m.predict(future)
    cat_df_train["y_original"] = cat_df_train["y"]
    cat_df["y_original"] = cat_df["y"]
    return m, forecast, cat_df_train


def visualise(m, forecast, cat_df, category: str, df_full: pd.DataFrame = None, retailer_id: str = None):
    dec_forecast = forecast[(forecast["ds"].dt.month == 12) & (forecast["ds"].dt.year == 2025)].copy()
    dec_actual = None
    if df_full is not None:
        if retailer_id:
            dec_actual = (
                df_full[(df_full["category"].str.lower() == category.lower()) &
                       (df_full["retailer_id"] == retailer_id) &
                       (df_full["date"].dt.month == 12) &
                       (df_full["date"].dt.year == 2025)]
                .groupby("date", as_index=False)
                .agg({"sales_volume": "sum"})
                .rename(columns={"date": "ds", "sales_volume": "y"})
                .sort_values("ds")
            )
        else:
            dec_actual = (
                df_full[(df_full["category"].str.lower() == category.lower()) &
                       (df_full["date"].dt.month == 12) &
                       (df_full["date"].dt.year == 2025)]
                .groupby("date", as_index=False)
                .agg({"sales_volume": "sum"})
                .rename(columns={"date": "ds", "sales_volume": "y"})
                .sort_values("ds")
            )

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.fill_between(dec_forecast["ds"], dec_forecast["yhat_lower"], dec_forecast["yhat_upper"],
                    alpha=0.25, color="steelblue", label="95% CI (Forecast)")
    ax.plot(dec_forecast["ds"], dec_forecast["yhat"], color="steelblue", lw=2.5, label="Predicted Sales", zorder=2)
    ax.scatter(dec_forecast["ds"], dec_forecast["yhat"], s=80, color="steelblue", alpha=0.8, marker="o", zorder=3)
    if dec_actual is not None and not dec_actual.empty:
        ax.scatter(dec_actual["ds"], dec_actual["y"], s=100, color="darkgreen", alpha=0.7, label="Actual Sales", marker="o", edgecolors="darkgreen", linewidths=2, zorder=3)
    ax.set_title(f"Retail Inventory Twin — {category} @ {retailer_id} December 2025\nForecast vs Actual Sales",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Sales Volume (units)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    out = f"charts/forecast_v2_{category.replace(' ', '_').lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n📊 Chart saved → {out}")
    plt.close()

    if dec_actual is not None and not dec_actual.empty:
        print(f"\n{'─'*60}")
        print(f"📊 DECEMBER 2025 VALIDATION METRICS")
        print(f"{'─'*60}")
        print(f"Forecast days: {len(dec_forecast)} | Actual days: {len(dec_actual)}")
        if len(dec_forecast) > 0 and len(dec_actual) > 0:
            min_len = min(len(dec_forecast), len(dec_actual))
            mae = np.mean(np.abs(dec_forecast["yhat"].values[:min_len] - dec_actual["y"].values[:min_len]))
            rmse = np.sqrt(np.mean((dec_forecast["yhat"].values[:min_len] - dec_actual["y"].values[:min_len]) ** 2))
            mape = np.mean(np.abs((dec_forecast["yhat"].values[:min_len] - dec_actual["y"].values[:min_len]) / dec_actual["y"].values[:min_len])) * 100
            print(f"MAE:  {mae:.2f} units")
            print(f"RMSE: {rmse:.2f} units")
            print(f"MAPE: {mape:.2f}%")
            print(f"Avg Forecast: {dec_forecast['yhat'].mean():.2f} units")
            print(f"Avg Actual:   {dec_actual['y'].mean():.2f} units")
        print(f"{'─'*60}")


def plot_daily_errors(forecast, dec_actual, category: str, retailer_id: str):
    if dec_actual is None or dec_actual.empty:
        print(f"⚠️  No actual data available for error plot")
        return
    dec_forecast = forecast[(forecast["ds"].dt.month == 12) & (forecast["ds"].dt.year == 2025)].copy()
    min_len = min(len(dec_forecast), len(dec_actual))
    dec_forecast_aligned = dec_forecast.iloc[:min_len].copy().reset_index(drop=True)
    dec_actual_aligned = dec_actual.iloc[:min_len].copy().reset_index(drop=True)
    errors = dec_forecast_aligned["yhat"].values - dec_actual_aligned["y"].values
    abs_errors = np.abs(errors)
    dates = dec_actual_aligned["ds"].dt.strftime("%b %d").values
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    colors = ["#d62728" if e > 0 else "#1f77b4" for e in errors]
    ax.bar(range(len(errors)), errors, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.set_title(f"Retail Inventory Twin — {category} @ {retailer_id} December 2025\nDaily Prediction Error (Forecast - Actual)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Error (units)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    mean_error = np.mean(errors)
    mae = np.mean(abs_errors)
    textstr = f"Mean Error: {mean_error:.2f} units\nMAE: {mae:.2f} units"
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    out = f"charts/error_v2_{category.replace(' ', '_').lower()}_{retailer_id}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"📊 Error chart saved → {out}")
    plt.close()


def estimate_stockout(forecast: pd.DataFrame, cat_df: pd.DataFrame,
                      current_stock: float, category: str, retailer_id: str = None):
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


if __name__ == "__main__":
    os.makedirs("charts", exist_ok=True)

    print("\n" + "="*60)
    print("🚀 RETAIL INVENTORY TWIN — CLEAN SLATE MODEL v2")
    print("="*60)
    print("\nStrategy:")
    print("  🧹 CLEAN SLATE: Simple, stable, interpretable")
    print("  ✅ Keep: Holiday/Promotion regressors")
    print("  ✅ Keep: Weather regressors (Temp, Humidity)")
    print("  ✅ Reset: changepoint_prior_scale=0.01 (ultra-stable baseline)")
    print("  ✅ Use: Linear growth (simple & predictable)")
    print("="*60 + "\n")

    sales_df = load_sales(CSV_PATH)

    start_dt = sales_df["date"].min().to_pydatetime()
    end_dt   = sales_df["date"].max().to_pydatetime()
    weather_df = fetch_weather_range(OWM_API_KEY, CITY, start_dt, end_dt)

    merged_df = build_joined_db(sales_df, weather_df, DB_PATH)

    category_data = merged_df[merged_df["category"].str.lower() == TARGET_CATEGORY.lower()]
    retailers = sorted(category_data["retailer_id"].unique())

    print(f"\n{'='*60}")
    print(f"🏪 Forecasting for {len(retailers)} retailers in '{TARGET_CATEGORY}'")
    print(f"   Retailers: {', '.join(retailers)}")
    print(f"{'='*60}")

    stockout_summary = []

    for retailer_id in retailers:
        retailer_cat_data = category_data[category_data["retailer_id"] == retailer_id]
        current_stock = retailer_cat_data["stock_quantity"].iloc[-1] if len(retailer_cat_data) > 0 else 100

        model, forecast, cat_df = train_and_forecast_v2(
            merged_df,
            TARGET_CATEGORY,
            retailer_id=retailer_id,
            forecast_days=31
        )

        visualise(model, forecast, cat_df, TARGET_CATEGORY, df_full=merged_df, retailer_id=retailer_id)

        dec_actual = (
            merged_df[(merged_df["category"].str.lower() == TARGET_CATEGORY.lower()) &
                     (merged_df["retailer_id"] == retailer_id) &
                     (merged_df["date"].dt.month == 12) &
                     (merged_df["date"].dt.year == 2025)]
            .groupby("date", as_index=False)
            .agg({"sales_volume": "sum"})
            .rename(columns={"date": "ds", "sales_volume": "y"})
            .sort_values("ds")
        )

        plot_daily_errors(forecast, dec_actual, TARGET_CATEGORY, retailer_id)

        estimate_stockout(forecast, cat_df, current_stock=current_stock,
                         category=TARGET_CATEGORY, retailer_id=retailer_id)

        future_fc = forecast[forecast["ds"] > cat_df["ds"].max()][["ds", "yhat"]].copy()
        future_fc["cumulative_demand"] = future_fc["yhat"].clip(lower=0).cumsum()
        stockout_row = future_fc[future_fc["cumulative_demand"] >= current_stock]

        if not stockout_row.empty:
            stockout_date = stockout_row.iloc[0]["ds"].date()
            days_until = (stockout_date - cat_df["ds"].max().date()).days
        else:
            stockout_date = None
            days_until = None

        reorder_qty = int(retailer_cat_data["reorder_quantity"].iloc[-1]) if len(retailer_cat_data) > 0 else 0

        stockout_summary.append({
            "retailer": retailer_id,
            "category": TARGET_CATEGORY,
            "current_stock": current_stock,
            "stockout_date": stockout_date,
            "days_until_stockout": days_until,
            "reorder_quantity": reorder_qty
        })

    print(f"\n{'='*60}")
    print(f"📊 STOCK-OUT SUMMARY — CLEAN SLATE MODEL v2")
    print(f"{'='*60}\n")

    summary_df = pd.DataFrame(stockout_summary)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("stockout_results.csv", index=False)
    print("📄 Stockout results saved to stockout_results.csv")

    print(f"\n{'='*60}")
    print("✅ Clean Slate model completed successfully!")
    print("   Stable, interpretable, focus on core drivers:")
    print("   • Holiday/Promotion effects")
    print("   • Weather influence")
    print("   • Predictable trend baseline")
    print(f"{'='*60}\n")
