"""
Week 1 - Setup and Data Understanding
Retail Inventory Twin Project

Goal:
    - Load the CSV file into a Pandas DataFrame
    - Clean and convert the Date column to a proper datetime format
    - Create a SQLite database and store the cleaned data in it

Requirements:
    - Python 3.8 or higher
    - pandas     (pip install pandas)
    - sqlalchemy (pip install sqlalchemy)

Run this script from the same folder where your CSV file is located:
    python week1_setup.py
"""

import pandas as pd
import sqlite3
import os


# -----------------------------------------------------------------------
# STEP 1: Define file paths
# -----------------------------------------------------------------------

# Change this to the actual name of your CSV file if it is different
CSV_FILE = "dataset_new.csv"

# This is the name of the SQLite database file that will be created
# SQLite stores everything in a single file on your disk
DATABASE_FILE = "inventory.db"


# -----------------------------------------------------------------------
# STEP 2: Load the CSV into a Pandas DataFrame
# -----------------------------------------------------------------------

# Check that the CSV file actually exists before trying to open it
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"Could not find '{CSV_FILE}'. "
        "Make sure it is in the same folder as this script."
    )

print(f"Loading data from: {CSV_FILE}")

# pd.read_csv reads a comma-separated file and returns a DataFrame.
# A DataFrame is like a spreadsheet table inside Python.
df = pd.read_csv(CSV_FILE)

print(f"Rows loaded    : {len(df)}")
print(f"Columns found  : {list(df.columns)}")
print()


# -----------------------------------------------------------------------
# STEP 3: Preview the data
# -----------------------------------------------------------------------

# df.head() shows the first 5 rows so you can do a quick sanity check
print("First 5 rows of raw data:")
print(df.head())
print()

# df.dtypes tells you what data type each column currently has.
# You will likely see that 'Date' is listed as 'object' (plain text).
print("Column data types (before cleaning):")
print(df.dtypes)
print()


# -----------------------------------------------------------------------
# STEP 4: Convert the Date column to datetime
# -----------------------------------------------------------------------

# Right now the Date column is stored as plain text, e.g. "1/1/2022".
# We need Python to understand it as a real date so we can later:
#   - sort by date
#   - filter by month or year
#   - plot time series charts
#
# pd.to_datetime() converts a text column into proper date objects.
# The format string tells pandas exactly how the date is written:
#   %m = month as a number (01-12)
#   %d = day   as a number (01-31)
#   %Y = 4-digit year
#
# If your dates look different (e.g. "2022-01-01"), change the format.

df["Date"] = pd.to_datetime(df["Date"])

print("Date column after conversion:")
print(df["Date"].head())
print(f"Date dtype is now: {df['Date'].dtype}")
print()


# -----------------------------------------------------------------------
# STEP 5: Check for missing values
# -----------------------------------------------------------------------

# isnull().sum() counts how many blank/missing cells are in each column.
# Ideally all counts should be 0. If not, we need to handle them later.

missing = df.isnull().sum()
print("Missing values per column:")
print(missing)
print()

# Quick overall check
total_missing = missing.sum()
if total_missing == 0:
    print("Good news: no missing values found.")
else:
    print(f"Warning: {total_missing} missing values found. "
          "These will need to be handled in Week 2 (EDA).")
print()


# -----------------------------------------------------------------------
# STEP 6: Basic summary statistics
# -----------------------------------------------------------------------

# df.describe() gives you count, mean, min, max, etc. for numeric columns.
# This is a quick way to spot anything obviously wrong (e.g. negative prices).

print("Summary statistics for numeric columns:")
print(df.describe())
print()


# -----------------------------------------------------------------------
# STEP 7: Create a SQLite database and load the data into it
# -----------------------------------------------------------------------

# SQLite is a lightweight database that lives in a single file.
# It lets us run SQL queries on our data, which will be useful later
# for filtering and joining with weather data.

print(f"Creating SQLite database: {DATABASE_FILE}")

# sqlite3.connect() creates the database file if it does not exist yet,
# or opens it if it already exists.
connection = sqlite3.connect(DATABASE_FILE)

# df.to_sql() writes the entire DataFrame into a table called "sales".
# if_exists="replace" means: if the table already exists, drop it and
# rebuild it from scratch. This is safe to run multiple times.
# index=False means we do NOT write the pandas row numbers as a column.

df.to_sql(
    name="sales",
    con=connection,
    if_exists="replace",
    index=False
)

print("Data written to the 'sales' table.")
print()


# -----------------------------------------------------------------------
# STEP 8: Verify the data made it into SQLite correctly
# -----------------------------------------------------------------------

# We run a simple SQL query to read the first 3 rows back from the database.
# This confirms everything was saved properly.

cursor = connection.cursor()
cursor.execute("SELECT * FROM sales LIMIT 3;")
rows = cursor.fetchall()

print("First 3 rows read back from SQLite:")
for row in rows:
    print(row)
print()

# Also check the total row count in the database matches the CSV
cursor.execute("SELECT COUNT(*) FROM sales;")
db_row_count = cursor.fetchone()[0]
print(f"Total rows in database : {db_row_count}")
print(f"Total rows in DataFrame: {len(df)}")

if db_row_count == len(df):
    print("Row counts match. Data loaded successfully.")
else:
    print("Warning: row counts do not match. Something may have gone wrong.")
print()


# -----------------------------------------------------------------------
# STEP 9: Print the database schema
# -----------------------------------------------------------------------

# This query shows the CREATE TABLE statement SQLite stored internally.
# It describes all the column names and their types in the database.

cursor.execute(
    "SELECT sql FROM sqlite_master WHERE type='table' AND name='sales';"
)
schema = cursor.fetchone()[0]
print("Database schema for the 'sales' table:")
print(schema)
print()


# -----------------------------------------------------------------------
# STEP 10: Close the database connection
# -----------------------------------------------------------------------

# Always close the connection when you are finished.
# This flushes any pending writes and releases the file lock.

connection.close()
print("Database connection closed.")
print()
print("Week 1 setup complete.")
print(f"Your database is saved at: {os.path.abspath(DATABASE_FILE)}")
print("Next step: open week2_eda.py for exploratory data analysis.")
