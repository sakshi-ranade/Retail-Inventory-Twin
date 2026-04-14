"""
Model Comparison: v1 (Original) vs v2 (Clean Slate v2.2)
========================================================
Compare forecasting accuracy between original and Clean Slate v2.2 model
with logistic growth, physical capacity cap, and binary holiday/promotion features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from prophet import Prophet

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DB_PATH = "inventory.db"
CATEGORY = "Hot Beverages"

print("="*70)
print("📊 MODEL COMPARISON: v1 vs v2")
print("="*70)

# Load data
con = sqlite3.connect(DB_PATH)
merged_df = pd.read_sql("SELECT * FROM inventory", con)
con.close()
merged_df["date"] = pd.to_datetime(merged_df["date"])

# Filter category
cat_daily = (
    merged_df[merged_df["category"].str.lower() == CATEGORY.lower()]
    .groupby("date", as_index=False)
    .agg({
        "sales_volume": "sum",
        "final_temp_f": "mean",
        "final_humidity_pct": "mean",
        "is_holiday": "max",
        "is_promotion": "max",
        "cold_weekend_interaction": "max",
        "humid_cold_interaction": "max"
    })
    .sort_values("date")
    .reset_index(drop=True)
)

# Split: 11 months training, 1 month test
last_month_start = cat_daily["date"].max() - pd.DateOffset(months=1)
train_data = cat_daily[cat_daily["date"] < last_month_start].copy()
test_data = cat_daily[cat_daily["date"] >= last_month_start].copy()

print(f"\nData split:")
print(f"  Train: {train_data['date'].min().date()} to {train_data['date'].max().date()} ({len(train_data)} days)")
print(f"  Test:  {test_data['date'].min().date()} to {test_data['date'].max().date()} ({len(test_data)} days)")

# ─────────────────────────────────────────────
# MODEL v1: ORIGINAL (No fine-tuning)
# ─────────────────────────────────────────────
print(f"\n{'─'*70}")
print("🔄 TRAINING MODEL v1 (ORIGINAL - No fine-tuning)")
print(f"{'─'*70}")

prophet_train_v1 = train_data[["date", "sales_volume", "final_temp_f", "final_humidity_pct"]].copy()
prophet_train_v1.columns = ["ds", "y", "final_temp_f", "final_humidity_pct"]

m1 = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.1,  # Original value
)
m1.add_regressor("final_temp_f")
m1.add_regressor("final_humidity_pct")
m1.fit(prophet_train_v1)

future_v1 = pd.DataFrame({
    "ds": test_data["date"].values,
    "final_temp_f": test_data["final_temp_f"].values,
    "final_humidity_pct": test_data["final_humidity_pct"].values
})
forecast_v1 = m1.predict(future_v1)

# ─────────────────────────────────────────────
# MODEL v2: FINE-TUNED
# ─────────────────────────────────────────────
print(f"\n{'─'*70}")
print("🔄 TRAINING MODEL v2 (FINE-TUNED)")
print(f"{'─'*70}")

# Log-transform for v2 (Clean Slate Model)
train_data_v2 = train_data.copy()

# v2 uses simplified regressors (no interaction features, no log transformation)
prophet_train_v2 = train_data_v2[["date", "sales_volume", "final_temp_f", "final_humidity_pct",
                                   "is_holiday", "is_promotion"]].copy()
prophet_train_v2.columns = ["ds", "y", "final_temp_f", "final_humidity_pct",
                            "is_holiday", "is_promotion"]

# ✅ ADD PHYSICAL CAPACITY CAP (logistic growth constraint)
max_sales_v2 = prophet_train_v2["y"].max()
capacity_cap_v2 = max_sales_v2 * 1.5
prophet_train_v2["cap"] = capacity_cap_v2
prophet_train_v2["floor"] = 0
print(f"\nv2 Capacity Cap: {capacity_cap_v2:.0f} units (1.5x max sales of {max_sales_v2:.0f})")

# Clean Slate Model v2.2: Logistic growth, ultra-stable baseline (changepoint_prior_scale=0.01)
m2 = Prophet(
    growth="logistic",  # ← LOGISTIC GROWTH with cap (prevents unrealistic over-selling)
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.01,  # ← ULTRA-STABLE (prevents trend/weather/promotion competition)
    interval_width=0.95
)
m2.add_regressor("final_temp_f")
m2.add_regressor("final_humidity_pct")
m2.add_regressor("is_holiday")
m2.add_regressor("is_promotion")
m2.fit(prophet_train_v2)

# Prepare test data with capacity cap
test_weather = test_data[["final_temp_f", "final_humidity_pct"]].values

future_v2 = pd.DataFrame({
    "ds": test_data["date"].values,
    "final_temp_f": test_data["final_temp_f"].values,
    "final_humidity_pct": test_data["final_humidity_pct"].values,
    "is_holiday": test_data["is_holiday"].values,
    "is_promotion": test_data["is_promotion"].values,
    "cap": capacity_cap_v2,
    "floor": 0
})
forecast_v2 = m2.predict(future_v2)

# ─────────────────────────────────────────────
# COMPARE ACCURACY
# ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("📈 ACCURACY COMPARISON")
print(f"{'='*70}\n")

y_actual = test_data["sales_volume"].values

# v1 metrics
mae_v1 = np.mean(np.abs(y_actual - forecast_v1["yhat"].values))
rmse_v1 = np.sqrt(np.mean((y_actual - forecast_v1["yhat"].values) ** 2))
mape_v1 = np.mean(np.abs((y_actual - forecast_v1["yhat"].values) / y_actual)) * 100

# v2 metrics
mae_v2 = np.mean(np.abs(y_actual - forecast_v2["yhat"].values))
rmse_v2 = np.sqrt(np.mean((y_actual - forecast_v2["yhat"].values) ** 2))
mape_v2 = np.mean(np.abs((y_actual - forecast_v2["yhat"].values) / y_actual)) * 100

# Improvements
mae_improvement = ((mae_v1 - mae_v2) / mae_v1) * 100
mape_improvement = ((mape_v1 - mape_v2) / mape_v1) * 100
rmse_improvement = ((rmse_v1 - rmse_v2) / rmse_v1) * 100

print(f"{'Model':<20} {'MAE':<15} {'RMSE':<15} {'MAPE':<12}")
print(f"{'-'*70}")
print(f"{'v1 (Original)':<20} {mae_v1:<15.1f} {rmse_v1:<15.1f} {mape_v1:<12.2f}%")
print(f"{'v2 (Fine-tuned)':<20} {mae_v2:<15.1f} {rmse_v2:<15.1f} {mape_v2:<12.2f}%")
print(f"{'-'*70}")
print(f"{'IMPROVEMENT':<20} {mae_improvement:+.1f}%{'':<10} {rmse_improvement:+.1f}%{'':<10} {mape_improvement:+.1f}%")

if mae_improvement > 0:
    print(f"\n✅ MODEL v2 IS BETTER: {abs(mae_improvement):.1f}% improvement in MAE")
else:
    print(f"\n⚠️  No clear improvement (or marginal)")

# ─────────────────────────────────────────────
# VISUALIZE COMPARISON
# ─────────────────────────────────────────────
print(f"\n📊 Creating comparison visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"Model Comparison: Original vs Fine-Tuned ({CATEGORY})", fontsize=16, fontweight="bold")

# Plot 1: Actual vs v1 vs v2
ax = axes[0, 0]
ax.plot(test_data["date"], y_actual, "ko-", lw=2.5, markersize=8, label="Actual", zorder=3)
ax.plot(forecast_v1["ds"], forecast_v1["yhat"], "r--", lw=2, label="v1 (Original)", alpha=0.7)
ax.plot(forecast_v2["ds"], forecast_v2["yhat"], "g-", lw=2, label="v2 (Fine-tuned)", alpha=0.7)
ax.fill_between(forecast_v2["ds"], forecast_v2["yhat_lower"], forecast_v2["yhat_upper"], 
               alpha=0.2, color="green", label="v2 95% CI")
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Sales Volume (units)", fontsize=11)
ax.set_title("Forecast Comparison: Actual vs Models")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

# Plot 2: Error comparison (v1 vs v2)
ax = axes[0, 1]
error_v1 = y_actual - forecast_v1["yhat"].values
error_v2 = y_actual - forecast_v2["yhat"].values
x_pos = np.arange(len(test_data))
width = 0.35
ax.bar(x_pos - width/2, error_v1, width, label="v1 Error", alpha=0.7, color="red")
ax.bar(x_pos + width/2, error_v2, width, label="v2 Error", alpha=0.7, color="green")
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Day in Test Period", fontsize=11)
ax.set_ylabel("Error (Actual - Predicted)", fontsize=11)
ax.set_title("Daily Prediction Error: v1 vs v2")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Plot 3: Error distribution
ax = axes[1, 0]
ax.hist(error_v1, bins=8, alpha=0.6, label="v1", color="red", edgecolor="black")
ax.hist(error_v2, bins=8, alpha=0.6, label="v2", color="green", edgecolor="black")
ax.axvline(np.mean(error_v1), color="red", ls="--", lw=2, label=f"v1 mean = {np.mean(error_v1):.1f}")
ax.axvline(np.mean(error_v2), color="green", ls="--", lw=2, label=f"v2 mean = {np.mean(error_v2):.1f}")
ax.set_xlabel("Error (units)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title("Error Distribution")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Plot 4: Metrics comparison (bar chart)
ax = axes[1, 1]
models = ["v1\n(Original)", "v2\n(Fine-tuned)"]
mae_values = [mae_v1, mae_v2]
mape_values = [mape_v1, mape_v2]

x = np.arange(len(models))
width = 0.35

ax2 = ax.twinx()
bars1 = ax.bar(x - width/2, mae_values, width, label="MAE", color="steelblue", alpha=0.8)
bars2 = ax2.bar(x + width/2, mape_values, width, label="MAPE %", color="coral", alpha=0.8)

ax.set_ylabel("MAE (units)", fontsize=11, color="steelblue")
ax2.set_ylabel("MAPE (%)", fontsize=11, color="coral")
ax.set_title("Accuracy Metrics Comparison")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.tick_params(axis="y", labelcolor="steelblue")
ax2.tick_params(axis="y", labelcolor="coral")
ax.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax.text(bar1.get_x() + bar1.get_width()/2., height1,
           f'{height1:.0f}', ha='center', va='bottom', fontsize=10, color="steelblue")
    ax2.text(bar2.get_x() + bar2.get_width()/2., height2,
            f'{height2:.1f}%', ha='center', va='bottom', fontsize=10, color="coral")

plt.tight_layout()
plt.savefig("model_comparison_v1_vs_v2.png", dpi=150, bbox_inches="tight")
print(f"✅ Chart saved → model_comparison_v1_vs_v2.png")
plt.show()

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("📋 IMPROVEMENTS SUMMARY")
print(f"{'='*70}\n")

print("✅ MODEL v2 ENHANCEMENTS (CLEAN SLATE v2.2):")
print(f"  1. Logistic Growth: Respects physical capacity ceiling")
print(f"  2. Capacity Cap: 1.5x max sales (prevents over-selling)")
print(f"  3. Ultra-Stable Baseline: changepoint_prior_scale=0.01 (prevents trend/regressor competition)")
print(f"  4. Direct Sales Volume: NO log transformation (cleaner predictions)")
print(f"  5. Binary Holiday/Promotion: Handles spikes as 1-day coefficients")
print(f"  6. Simplified Features: Only weather + holidays/promotions (no interactions)")

print(f"\n📊 QUANTIFIED IMPROVEMENTS:")
print(f"  • MAE reduction:     {mae_improvement:+.1f}%")
print(f"  • RMSE reduction:    {rmse_improvement:+.1f}%")
print(f"  • MAPE reduction:    {mape_improvement:+.1f}%")

if mape_improvement > 5:
    print(f"\n✅ SIGNIFICANT IMPROVEMENT! v2 is noticeably better.")
elif mape_improvement > 0:
    print(f"\n✅ MODEST IMPROVEMENT: v2 provides better accuracy.")
else:
    print(f"\n⚠️  Similar performance. Further fine-tuning may be needed.")

print(f"\n{'='*70}\n")
