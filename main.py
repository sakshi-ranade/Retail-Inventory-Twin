import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv('dataset.csv', encoding='cp1252')

# 2. Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# 3. Filter to last 3 months of data in the dataset
max_date = df['Order Date'].max()
min_date = max_date - pd.DateOffset(months=3)
df = df[df['Order Date'] >= min_date]

print(f"Date range: {df['Order Date'].min()} to {df['Order Date'].max()}")
print(f"Rows after filtering: {len(df)}")

# 4. Keep only needed columns
df = df[['Order Date', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Sales']]

# 5. Group by date and category to get daily sales
daily_sales = df.groupby(['Order Date', 'Category', 'Sub-Category'])['Quantity'].sum().reset_index()
daily_sales.columns = ['ds', 'Category', 'Sub-Category', 'units_sold']

print("\nDaily sales sample:")
print(daily_sales.head(10))
print("\nCategories:", daily_sales['Category'].unique())
print("\nShape:", daily_sales.shape)


# 6. Load daily sales into SQLite database
conn = sqlite3.connect('inventory_twin.db')

# Drop existing view if exists
conn.execute("DROP VIEW IF EXISTS daily_sales")
conn.commit()

daily_sales.to_sql('daily_sales', conn, if_exists='replace', index=False)


# 7. Add a simple inventory table with current stock levels
inventory = pd.DataFrame({
    'Category': ['Office Supplies', 'Technology', 'Furniture'],
    'Sub-Category': ['Binders', 'Phones', 'Chairs'],
    'current_stock': [500, 300, 150]
})
inventory.to_sql('inventory', conn, if_exists='replace', index=False)

print("\nData loaded into SQLite successfully!")

# 8. Verifying data 
print("\nDaily sales in DB:")
print(pd.read_sql("SELECT * FROM daily_sales LIMIT 5", conn))
print("\nInventory in DB:")
print(pd.read_sql("SELECT * FROM inventory", conn))

conn.close()

