import pandas as pd
import sqlite3
from prophet import Prophet
import matplotlib.pyplot as plt

# 1. Load merged data from SQLite
conn = sqlite3.connect('inventory_twin.db')
df = pd.read_sql("SELECT * FROM merged_data", conn)
conn.close()

# 2. Convert date column
df['ds'] = pd.to_datetime(df['ds'])

# 3. Filter for Office Supplies only (1 category as proof of concept)
category = 'Office Supplies'
cat_df = df[df['Category'] == category].groupby('ds').agg(
    y=('units_sold', 'sum'),
    temperature=('temperature', 'mean'),
    humidity=('humidity', 'mean'),
    rainfall=('rainfall', 'mean')
).reset_index()

print(f"Training Prophet for: {category}")
print(f"Data points: {len(cat_df)}")

# 4. Train Prophet model with weather regressors
model = Prophet(daily_seasonality=False, weekly_seasonality=True)
model.add_regressor('temperature')
model.add_regressor('humidity')
model.add_regressor('rainfall')

model.fit(cat_df[['ds', 'y', 'temperature', 'humidity', 'rainfall']])

# 5. Predict next 30 days
future = model.make_future_dataframe(periods=30)
avg_temp = cat_df['temperature'].tail(7).mean()
avg_humidity = cat_df['humidity'].tail(7).mean()
avg_rainfall = cat_df['rainfall'].tail(7).mean()

future['temperature'] = avg_temp
future['humidity'] = avg_humidity
future['rainfall'] = avg_rainfall

forecast = model.predict(future)
forecast['Category'] = category
forecast['predicted_sales'] = forecast['yhat'].clip(lower=0).round(2)

# 6. Save to SQLite
conn = sqlite3.connect('inventory_twin.db')
forecast[['ds', 'predicted_sales', 'yhat_lower', 'yhat_upper', 'Category']].to_sql(
    'forecasts', conn, if_exists='replace', index=False)
conn.close()
print("Forecast saved to database!")

# 7. Plot forecast
plt.figure(figsize=(12, 5))
plt.plot(cat_df['ds'], cat_df['y'], color='blue', label='Actual Sales')
plt.plot(forecast['ds'], forecast['predicted_sales'], color='red', 
         linestyle='--', label='Forecast')
plt.axvline(x=pd.Timestamp('2017-12-30'), color='black', 
            linestyle='--', label='Forecast Start')
plt.title(f'{category} - 30 Day Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('forecast_chart.png')
print("Forecast chart saved!")
plt.show()

