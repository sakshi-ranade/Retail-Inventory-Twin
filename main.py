import pandas as pd
import sqlite3

df = pd.read_csv('dataset.csv', encoding='cp1252')
df['Order Date'] = pd.to_datetime(df['Order Date'])

max_date = df['Order Date'].max()
min_date = max_date - pd.DateOffset(months=3)
df = df[df['Order Date'] >= min_date]

print(f"Date range: {df['Order Date'].min()} to {df['Order Date'].max()}")
print(f"Rows after filtering: {len(df)}")

df = df[['Order Date', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Sales']]

daily_sales = df.groupby(['Order Date', 'Category', 'Sub-Category'])['Quantity'].sum().reset_index()
daily_sales.columns = ['ds', 'Category', 'Sub-Category', 'units_sold']

print("\nDaily sales sample:")
print(daily_sales.head(10))
print("\nCategories:", daily_sales['Category'].unique())
print("\nShape:", daily_sales.shape)

conn = sqlite3.connect('inventory_twin.db')
conn.execute("DROP TABLE IF EXISTS daily_sales")
conn.execute("DROP VIEW IF EXISTS daily_sales")
conn.commit()

daily_sales.to_sql('daily_sales', conn, if_exists='replace', index=False)

inventory = pd.DataFrame({
    'Category': ['Office Supplies', 'Technology', 'Furniture'],
    'Sub-Category': ['Binders', 'Phones', 'Chairs'],
    'current_stock': [1200, 400, 350]
})
inventory.to_sql('inventory', conn, if_exists='replace', index=False)

print("\nData loaded into SQLite successfully!")
print("\nInventory in DB:")
print(pd.read_sql("SELECT * FROM inventory", conn))

conn.close()
