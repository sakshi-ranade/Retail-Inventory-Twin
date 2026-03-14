# Retail Inventory Twin - Prophet Forecasting Model
import pandas as pd
import sqlite3
from prophet import Prophet
import matplotlib.pyplot as plt

# ── 1. Load merged data from SQLite ─────────────────────────────
conn = sqlite3.connect('inventory_twin.db')
df = pd.read_sql("SELECT * FROM merged_data", conn)
conn.close()

# ── 2. Convert date column ───────────────────────────────────────
df['ds'] = pd.to_datetime(df['ds'])

# ── 3. Group by date and category (total units sold per day) ─────
category_sales = df.groupby(['ds', 'Category']).agg(
    units_sold=('units_sold', 'sum'),
    temperature=('temperature', 'mean'),
    humidity=('humidity', 'mean'),
    rainfall=('rainfall', 'mean')
).reset_index()

# ── 4. Train Prophet model for each category ─────────────────────
categories = ['Office Supplies', 'Technology', 'Furniture']
all_forecasts = []

for category in categories:
    print(f"\nTraining Prophet model for: {category}")
    
    # Filter data for this category
    cat_df = category_sales[category_sales['Category'] == category].copy()
    
    # Prophet requires columns named 'ds' and 'y'
    cat_df = cat_df.rename(columns={'units_sold': 'y'})
    
    # Add weather as extra regressors
    cat_df['temperature'] = cat_df['temperature']
    cat_df['humidity'] = cat_df['humidity']
    cat_df['rainfall'] = cat_df['rainfall']
    
    # Initialize Prophet model with weather regressors
    model = Prophet(daily_seasonality=False, weekly_seasonality=True)
    model.add_regressor('temperature')
    model.add_regressor('humidity')
    model.add_regressor('rainfall')
    
    # Train the model
    model.fit(cat_df[['ds', 'y', 'temperature', 'humidity', 'rainfall']])
    
    # Create future dataframe for next 30 days
    future = model.make_future_dataframe(periods=30)
    
    # Add weather values for future dates (use average of last 7 days)
    avg_temp = cat_df['temperature'].tail(7).mean()
    avg_humidity = cat_df['humidity'].tail(7).mean()
    avg_rainfall = cat_df['rainfall'].tail(7).mean()
    
    future['temperature'] = avg_temp
    future['humidity'] = avg_humidity
    future['rainfall'] = avg_rainfall
    
    # Generate forecast
    forecast = model.predict(future)
    forecast['Category'] = category
    all_forecasts.append(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'Category']])
    
    print(f"Forecast for {category} complete!")
    print(forecast[['ds', 'yhat']].tail(10))

# ── 5. Combine all forecasts ─────────────────────────────────────
final_forecast = pd.concat(all_forecasts, ignore_index=True)
final_forecast.rename(columns={'yhat': 'predicted_sales'}, inplace=True)
final_forecast['predicted_sales'] = final_forecast['predicted_sales'].clip(lower=0).round(2)

# ── 6. Save forecasts to SQLite ──────────────────────────────────
conn = sqlite3.connect('inventory_twin.db')
final_forecast.to_sql('forecasts', conn, if_exists='replace', index=False)
conn.close()
print("\nAll forecasts saved to SQLite database!")

# ── 7. Plot forecasts for each category ──────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
colors = ['blue', 'green', 'red']

for i, category in enumerate(categories):
    cat_forecast = final_forecast[final_forecast['Category'] == category]
    axes[i].plot(cat_forecast['ds'], cat_forecast['predicted_sales'], 
                 color=colors[i], linewidth=1.5)
    axes[i].set_title(f'{category} - 30 Day Sales Forecast')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('Predicted Units Sold')
    axes[i].grid(True, alpha=0.3)
    axes[i].axvline(x=pd.Timestamp('2017-12-30'), color='black', 
                    linestyle='--', label='Forecast Start')
    axes[i].legend()

plt.tight_layout()
plt.savefig('forecast_chart.png')
print("Forecast chart saved as forecast_chart.png")
plt.show()