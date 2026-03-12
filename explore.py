# Retail Inventory Twin - Data Exploration & Merging
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

#  1. Load data from SQLite 
conn = sqlite3.connect('inventory_twin.db')
sales_df = pd.read_sql("SELECT * FROM daily_sales", conn)
weather_df = pd.read_sql("SELECT * FROM weather_data", conn)
conn.close()

# 2. Convert dates 
sales_df['ds'] = pd.to_datetime(sales_df['ds'])
weather_df['ds'] = pd.to_datetime(weather_df['ds'])

# 3. Merge sales and weather on date 
merged_df = pd.merge(sales_df, weather_df, on='ds', how='left')
print("Merged dataset shape:", merged_df.shape)
print("\nSample merged data:")
print(merged_df.head(10))

# 4. Save merged dataset to SQLite
conn = sqlite3.connect('inventory_twin.db')
merged_df.to_sql('merged_data', conn, if_exists='replace', index=False)
conn.close()
print("\nMerged data saved to SQLite!")

# 5. Plot sales trends by category 
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

categories = ['Office Supplies', 'Technology', 'Furniture']
colors = ['blue', 'green', 'red']

for i, (cat, color) in enumerate(zip(categories, colors)):
    cat_data = merged_df[merged_df['Category'] == cat].groupby('ds')['units_sold'].sum()
    axes[i].plot(cat_data.index, cat_data.values, color=color, linewidth=1.5)
    axes[i].set_title(f'{cat} - Daily Sales Trend')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('Units Sold')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sales_trends.png')
print("\nSales trend chart saved as sales_trends.png")
plt.show()

# 6. Plot temperature vs total daily sales 
daily_total = merged_df.groupby('ds')['units_sold'].sum().reset_index()
daily_total = pd.merge(daily_total, weather_df, on='ds', how='left')

plt.figure(figsize=(10, 5))
plt.scatter(daily_total['temperature'], daily_total['units_sold'], 
            alpha=0.6, color='purple')
plt.xlabel('Temperature (°F)')
plt.ylabel('Total Units Sold')
plt.title('Temperature vs Total Daily Sales')
plt.grid(True, alpha=0.3)
plt.savefig('weather_correlation.png')
print("Weather correlation chart saved as weather_correlation.png")
plt.show()

