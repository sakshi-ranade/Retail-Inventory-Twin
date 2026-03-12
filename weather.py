# Retail Inventory Twin - Weather Data Integration
import pandas as pd
import numpy as np
import sqlite3

def generate_historical_weather():
    """
    Generate realistic historical weather data for
    New York City from Sept 30 to Dec 30, 2017.
    Based on real NYC seasonal temperature patterns.
    """
    dates = pd.date_range(start='2017-09-30', end='2017-12-30', freq='D')
    
    np.random.seed(42)
    
    # NYC average temps by month (Fahrenheit)
    # Sept: ~65F, Oct: ~55F, Nov: ~45F, Dec: ~35F
    temps = []
    humidity = []
    rainfall = []
    
    for date in dates:
        if date.month == 9:
            base_temp = 65
        elif date.month == 10:
            base_temp = 55
        elif date.month == 11:
            base_temp = 45
        else:
            base_temp = 35
            
        # Add realistic daily variation
        temp = base_temp + np.random.normal(0, 5)
        temps.append(round(temp, 2))
        
        # Humidity between 50-90%
        humidity.append(round(np.random.uniform(50, 90), 2))
        
        # Rainfall - 30% chance of rain
        rainfall.append(round(np.random.uniform(0, 1) if np.random.random() < 0.3 else 0, 2))
    
    df = pd.DataFrame({
        'ds': dates,
        'temperature': temps,
        'humidity': humidity,
        'rainfall': rainfall
    })
    
    return df

def save_weather_to_db(df):
    """Save weather data to SQLite database"""
    conn = sqlite3.connect('inventory_twin.db')
    df.to_sql('weather_data', conn, if_exists='replace', index=False)
    print("Historical weather data saved successfully!")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"Total days: {len(df)}")
    print("\nSample data:")
    print(df.head(10))
    conn.close()

if __name__ == "__main__":
    print("Generating historical weather data for NYC (Sept-Dec 2017)...")
    weather_df = generate_historical_weather()
    save_weather_to_db(weather_df)

