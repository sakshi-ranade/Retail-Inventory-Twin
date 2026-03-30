# Retail Inventory Twin - Stock-Out Prediction Engine
import pandas as pd
import sqlite3

# ── 1. Load data from SQLite ─────────────────────────────────────
conn = sqlite3.connect('inventory_twin.db')
forecasts = pd.read_sql("SELECT * FROM forecasts", conn)
inventory = pd.read_sql("SELECT * FROM inventory", conn)
conn.close()

# ── 2. Convert date column ───────────────────────────────────────
forecasts['ds'] = pd.to_datetime(forecasts['ds'])

# ── 3. Filter only future predictions ───────────────────────────
future = forecasts[forecasts['ds'] > '2017-12-30'].copy()

# ── 4. Calculate days until stock-out per category ───────────────
print("\n" + "="*50)
print("   RETAIL INVENTORY TWIN - STOCK-OUT REPORT")
print("="*50)

results = []

for _, row in inventory.iterrows():
    category = row['Category']
    current_stock = row['current_stock']
    
    # Get future predictions for this category
    cat_forecast = future[future['Category'] == category].copy()
    cat_forecast = cat_forecast.sort_values('ds')
    
    # Simulate stock depletion day by day
    stock = current_stock
    days_until_stockout = 0
    stockout_date = None
    
    for _, f_row in cat_forecast.iterrows():
        daily_sales = max(0, f_row['predicted_sales'])
        stock -= daily_sales
        days_until_stockout += 1
        
        if stock <= 0:
            stockout_date = f_row['ds'].strftime('%Y-%m-%d')
            break
    
    # If stock never runs out in forecast period
    if stock > 0:
        days_until_stockout = 30
        stockout_date = "Beyond 30 days"
    
    # Determine alert level
    if days_until_stockout <= 15:
        alert = "🔴 CRITICAL"
    elif days_until_stockout <= 30:
        alert = "🟡 WARNING"
    else:
        alert = "🟢 SAFE"
    
    # Calculate reorder quantity (14 days of predicted demand)
    avg_daily = cat_forecast['predicted_sales'].mean()
    reorder_qty = round(avg_daily * 14)
    
    results.append({
        'Category': category,
        'Current Stock': current_stock,
        'Days Until Stock-Out': days_until_stockout,
        'Stock-Out Date': stockout_date,
        'Alert': alert,
        'Reorder Quantity': reorder_qty
    })
    
    print(f"\nCategory: {category}")
    print(f"Current Stock: {current_stock} units")
    print(f"Days Until Stock-Out: {days_until_stockout} days")
    print(f"Predicted Stock-Out Date: {stockout_date}")
    print(f"Reorder Quantity Suggested: {reorder_qty} units")
    print(f"Status: {alert}")
    print("-"*50)

# ── 5. Save results to SQLite ────────────────────────────────────
results_df = pd.DataFrame(results)
conn = sqlite3.connect('inventory_twin.db')
results_df.to_sql('stockout_predictions', conn, if_exists='replace', index=False)
conn.close()
print("\nStock-out predictions saved to database!")

