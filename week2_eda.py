"""
Week 2 - Exploratory Data Analysis (EDA)  [Updated]
Retail Inventory Twin Project

Goal:
    - Discover which categories sell the most under different weather
      conditions, both with and without promotions or discounts
    - Identify pre-bad-weather demand spikes (pantry-loading effect)
      in the 2-3 days leading up to rain or snow
    - Visualize correlations between sales, promotions, and weather

Charts produced:
    1a. Top categories by weather condition  - WITH promo/discount
    1b. Top categories by weather condition  - WITHOUT promo/discount
    2.  Pre-bad-weather sales spike          - avg units 3 days before
                                               bad weather vs normal days
    3.  Correlation heatmap                  - sales vs promotions
                                               on bad and good weather days

Requirements:
    - pandas     (pip install pandas)
    - matplotlib (pip install matplotlib)
    - seaborn    (pip install seaborn)
    - The SQLite database created in Week 1  (inventory.db)

Run this script from the same folder as inventory.db:
    python week2_eda.py
"""

import sqlite3
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------

DATABASE_FILE  = "inventory.db"
OUTPUT_FOLDER  = "eda_charts"

# How many top categories to show in each chart
TOP_N = 5

# Weather conditions that count as "bad weather"
BAD_WEATHER    = ["Rainy", "Snowy"]

# Column name references  -  update if your CSV uses different names
WEATHER_COL    = "Weather Condition"
SEASON_COL     = "Seasonality"
SALES_COL      = "Units Sold"
CATEGORY_COL   = "Category"
PROMO_COL      = "Holiday/Promotion"   # 1 = promotion active, 0 = none
DISCOUNT_COL   = "Discount"            # numeric discount percentage
DATE_COL       = "Date"
PRODUCT_COL    = "Product ID"
STORE_COL      = "Store ID"


# -----------------------------------------------------------------------
# HELPER: save a figure to disk and then display it
# -----------------------------------------------------------------------

def save_and_show(fig, filename):
    """Save a matplotlib figure to the output folder, then display it."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {filepath}")
    plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------
# STEP 1: Load data from SQLite
# -----------------------------------------------------------------------

print("=" * 60)
print("STEP 1: Loading data from database")
print("=" * 60)

if not os.path.exists(DATABASE_FILE):
    raise FileNotFoundError(
        f"Could not find '{DATABASE_FILE}'. "
        "Run week1_setup.py first to create the database."
    )

connection = sqlite3.connect(DATABASE_FILE)
df = pd.read_sql("SELECT * FROM sales", connection)
connection.close()

print(f"Rows loaded : {len(df)}")
print(f"Columns     : {list(df.columns)}")
print()


# -----------------------------------------------------------------------
# STEP 2: Clean and prepare columns
# -----------------------------------------------------------------------

print("=" * 60)
print("STEP 2: Cleaning and preparing data")
print("=" * 60)

# Convert Date to datetime so we can sort and shift by day
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

# Validate all required columns are present
required = [WEATHER_COL, SEASON_COL, SALES_COL, CATEGORY_COL,
            PROMO_COL, DISCOUNT_COL, DATE_COL, PRODUCT_COL, STORE_COL]
for col in required:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

# Drop rows missing values in the key columns we will analyse
df = df.dropna(subset=[SALES_COL, CATEGORY_COL, WEATHER_COL,
                        PROMO_COL, DISCOUNT_COL])

# Flag rows where a promotion OR any discount is active.
# A discount > 0 counts the same as a promotion for this analysis.
df["has_promo_or_discount"] = (
    (df[PROMO_COL] == 1) | (df[DISCOUNT_COL] > 0)
).astype(int)

# Flag bad-weather days
df["is_bad_weather"] = df[WEATHER_COL].isin(BAD_WEATHER).astype(int)

print(f"Rows after cleaning : {len(df)}")
print(f"Bad-weather rows    : {df['is_bad_weather'].sum()}")
print(f"Promo/discount rows : {df['has_promo_or_discount'].sum()}")
print()


# -----------------------------------------------------------------------
# CHARTS 1a & 1b: Top categories by weather condition
#                  split by WITH vs WITHOUT promo or discount
# -----------------------------------------------------------------------

print("=" * 60)
print("CHARTS 1a & 1b: Category sales by weather - promo split")
print("=" * 60)

# Compute average units sold per Category x Weather x Promo group
weather_promo = (
    df.groupby([CATEGORY_COL, WEATHER_COL, "has_promo_or_discount"])[SALES_COL]
    .mean()
    .round(1)
    .reset_index()
)

# Find the overall top N categories by total sales so both charts
# focus on the same set of categories
top_cats = (
    df.groupby(CATEGORY_COL)[SALES_COL]
    .sum()
    .nlargest(TOP_N)
    .index
    .tolist()
)
print(f"Top {TOP_N} categories: {top_cats}")

weather_promo_top = weather_promo[
    weather_promo[CATEGORY_COL].isin(top_cats)
].copy()

weather_order  = sorted(df[WEATHER_COL].unique())
palette        = sns.color_palette("tab10", n_colors=len(weather_order))
weather_colors = dict(zip(weather_order, palette))

for flag, label, filename in [
    (1, "WITH Promotion or Discount",    "chart1a_weather_with_promo.png"),
    (0, "WITHOUT Promotion or Discount", "chart1b_weather_no_promo.png"),
]:
    subset = weather_promo_top[
        weather_promo_top["has_promo_or_discount"] == flag
    ]

    # Build a wide pivot: rows = categories, columns = weather conditions
    pivot = (
        subset.pivot(index=CATEGORY_COL, columns=WEATHER_COL, values=SALES_COL)
        .reindex(index=top_cats, columns=weather_order)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    n_cats     = len(top_cats)
    n_weather  = len(weather_order)
    bar_width  = 0.18
    x          = range(n_cats)

    for i, weather in enumerate(weather_order):
        offsets = [xi + i * bar_width for xi in x]
        values  = [pivot.loc[cat, weather] if cat in pivot.index else 0
                   for cat in top_cats]
        ax.bar(
            offsets, values,
            width=bar_width,
            label=weather,
            color=weather_colors[weather],
            edgecolor="white"
        )

    # Centre the category labels under each group of bars
    group_center = [xi + bar_width * (n_weather - 1) / 2 for xi in x]
    ax.set_xticks(group_center)
    ax.set_xticklabels(top_cats, fontsize=11)

    ax.set_title(
        f"Average Daily Units Sold by Category and Weather\n{label}",
        fontsize=13, pad=12
    )
    ax.set_xlabel("Category")
    ax.set_ylabel("Avg Units Sold per Day")
    ax.legend(title="Weather", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(125, 150)

    # Print the pivot table to the console so you can read exact numbers
    print(f"\n{label}:")
    print(pivot.to_string())

    save_and_show(fig, filename)


# -----------------------------------------------------------------------
# CHART 2: Pre-bad-weather demand spike
#           Average sales on day -1, -2, -3 before a Rainy or Snowy day
#           compared against normal (non-pre-bad) days
# -----------------------------------------------------------------------

print()
print("=" * 60)
print("CHART 2: Pre-bad-weather sales spike")
print("=" * 60)

# Sort the dataset so we can use shift() to look one row ahead.
# We sort per Store + Product so we never bleed across product boundaries.
df = df.sort_values([STORE_COL, PRODUCT_COL, DATE_COL]).reset_index(drop=True)

# For each row, look at the weather on the NEXT 1, 2, and 3 days.
# shift(-1) moves the column up by one row, so row N gets row N+1's value.
# We group by Store+Product so the shift stays within the same product.
grp = df.groupby([STORE_COL, PRODUCT_COL])

df["weather_plus1"] = grp[WEATHER_COL].shift(-1)
df["weather_plus2"] = grp[WEATHER_COL].shift(-2)
df["weather_plus3"] = grp[WEATHER_COL].shift(-3)

# A row is "1 day before bad weather" if the NEXT day is bad
df["day_minus1"] = df["weather_plus1"].isin(BAD_WEATHER).astype(int)
df["day_minus2"] = df["weather_plus2"].isin(BAD_WEATHER).astype(int)
df["day_minus3"] = df["weather_plus3"].isin(BAD_WEATHER).astype(int)

# Build a label column for how many days before bad weather this row is.
# Priority: if it is day -1, label it -1 even if it is also day -2.
# Rows that are none of these get labelled "Normal day".
def label_pre_bad(row):
    if row["day_minus1"]:
        return "1 day before"
    if row["day_minus2"]:
        return "2 days before"
    if row["day_minus3"]:
        return "3 days before"
    return "Normal day"

df["pre_bad_label"] = df.apply(label_pre_bad, axis=1)

# Compute avg sales per category per pre-bad label
pre_bad_cat = (
    df[df[CATEGORY_COL].isin(top_cats)]
    .groupby([CATEGORY_COL, "pre_bad_label"])[SALES_COL]
    .mean()
    .round(2)
    .reset_index()
)

# Define a display order for the x-axis
label_order = ["3 days before", "2 days before", "1 day before", "Normal day"]

print("Average units sold in the days leading up to bad weather:")
pivot_pre = (
    pre_bad_cat.pivot(index=CATEGORY_COL, columns="pre_bad_label", values=SALES_COL)
    .reindex(columns=label_order)
)
print(pivot_pre.round(2).to_string())
print()

# Colour scheme: darkening greens as we approach the bad-weather day
pre_bad_colors = ["#9FE1CB", "#5DCAA5", "#1D9E75", "#888780"]

fig, ax = plt.subplots(figsize=(12, 5))

n_labels  = len(label_order)
bar_width  = 0.15
x          = range(n_labels)

for i, category in enumerate(top_cats):
    cat_data = pre_bad_cat[pre_bad_cat[CATEGORY_COL] == category]
    values   = []
    for lbl in label_order:
        row = cat_data[cat_data["pre_bad_label"] == lbl]
        values.append(float(row[SALES_COL].values[0]) if len(row) else 0)

    offsets = [xi + i * bar_width for xi in x]
    ax.bar(
        offsets, values,
        width=bar_width,
        label=category,
        color=sns.color_palette("tab10", n_colors=TOP_N)[i],
        edgecolor="white"
    )

group_center = [xi + bar_width * (TOP_N - 1) / 2 for xi in x]
ax.set_xticks(group_center)
ax.set_xticklabels(label_order, fontsize=11)

ax.set_title(
    "Avg Daily Units Sold: Days Leading Up to Bad Weather vs Normal Days\n"
    f"(Bad weather = {', '.join(BAD_WEATHER)})",
    fontsize=13, pad=12
)
ax.set_xlabel("Days Before Bad Weather")
ax.set_ylabel("Avg Units Sold per Day")
ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(125, 150)

# Draw a horizontal reference line at the overall normal-day average
normal_avg = df.loc[
    df["pre_bad_label"] == "Normal day", SALES_COL
].mean()
ax.axhline(
    normal_avg, color="gray", linestyle=":",
    linewidth=1.2, label=f"Overall normal avg ({normal_avg:.1f})"
)
ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left")

save_and_show(fig, "chart2_pre_bad_weather_spike.png")


# -----------------------------------------------------------------------
# CHART 3: Correlation heatmap
#           Sales vs Promotion on bad vs good weather days
# -----------------------------------------------------------------------

# print("=" * 60)
# print("CHART 3: Correlation heatmap - sales, promo, weather")
# print("=" * 60)

# # We want to show how sales correlate with promotions separately for
# # bad-weather days and good-weather days.
# # Strategy:
# #   - One-hot encode category, weather group (bad/good), and promo flag
# #   - Compute Pearson correlation across all numeric features vs Units Sold

# df_heat = df[df[CATEGORY_COL].isin(top_cats)].copy()

# # Binary weather group
# df_heat["Good Weather"] = (df_heat["is_bad_weather"] == 0).astype(int)
# df_heat["Bad Weather"]  = df_heat["is_bad_weather"]

# # Interaction columns: promo ON bad vs good weather days
# # These capture whether running a promotion matters differently
# # depending on the weather
# df_heat["Promo x Good Weather"] = (
#     (df_heat[PROMO_COL] == 1) & (df_heat["Good Weather"] == 1)
# ).astype(int)

# df_heat["Promo x Bad Weather"] = (
#     (df_heat[PROMO_COL] == 1) & (df_heat["Bad Weather"] == 1)
# ).astype(int)

# df_heat["Discount x Good Weather"] = df_heat[DISCOUNT_COL] * df_heat["Good Weather"]
# df_heat["Discount x Bad Weather"]  = df_heat[DISCOUNT_COL] * df_heat["Bad Weather"]

# # One-hot encode individual weather conditions
# weather_dummies  = pd.get_dummies(df_heat[WEATHER_COL], prefix="Weather")

# # One-hot encode categories
# category_dummies = pd.get_dummies(df_heat[CATEGORY_COL], prefix="Category")

# # Assemble the final correlation DataFrame
# interaction_cols = [
#     SALES_COL,
#     PROMO_COL,
#     DISCOUNT_COL,
#     "Good Weather",
#     "Bad Weather",
#     "Promo x Good Weather",
#     "Promo x Bad Weather",
#     "Discount x Good Weather",
#     "Discount x Bad Weather",
# ]

# # Combine interaction columns with one-hot weather and category dummies.
# # Reset index on dummies to prevent any index-alignment issues after the
# # sort_values() call above.
# weather_dummies  = weather_dummies.reset_index(drop=True)
# category_dummies = category_dummies.reset_index(drop=True)
# df_heat          = df_heat.reset_index(drop=True)

# corr_df = pd.concat(
#     [df_heat[interaction_cols], weather_dummies, category_dummies],
#     axis=1
# )

# # Remove any duplicate columns that may arise from the concat
# corr_df = corr_df.loc[:, ~corr_df.columns.duplicated()]

# corr_matrix = corr_df.corr()

# # Print correlations with Units Sold so you can read exact numbers
# print("Correlation of all features with Units Sold:")
# with_sales = (
#     corr_matrix[SALES_COL]
#     .drop(SALES_COL)
#     .sort_values(ascending=False)
# )
# print(with_sales.round(4).to_string())
# print()

# # Draw a focused heatmap using only the interaction variables.
# # We exclude the category dummies here to keep the chart readable.
# focus_cols = interaction_cols + list(weather_dummies.columns)

# focus_matrix = corr_df[focus_cols].corr()

# # Rename columns for cleaner labels on the chart
# rename_map = {
#     SALES_COL      : "Units Sold",
#     PROMO_COL      : "Promotion",
#     DISCOUNT_COL   : "Discount %",
# }
# focus_matrix = focus_matrix.rename(columns=rename_map, index=rename_map)

# fig, ax = plt.subplots(figsize=(12, 9))

# sns.heatmap(
#     focus_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="coolwarm",
#     center=0,
#     linewidths=0.5,
#     ax=ax,
#     annot_kws={"size": 8}
# )

# ax.set_title(
#     "Correlation Heatmap: Sales vs Promotions and Discounts\n"
#     "on Bad Weather vs Good Weather Days",
#     fontsize=13, pad=14
# )
# ax.tick_params(axis="x", rotation=45)
# ax.tick_params(axis="y", rotation=0)

# save_and_show(fig, "chart3_promo_weather_heatmap.png")

# -----------------------------------------------------------------------
# CHART 3: Sales vs Weather (Bad vs Good)
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# CHART 3: Correlation between Day -1 Sales and Bad Weather Day Sales
#           (per Category and Weather Type)
# -----------------------------------------------------------------------

print("=" * 60)
print("CHART 3: Correlation - Day -1 vs Bad Weather Day Sales")
print("=" * 60)

# Ensure data is sorted properly for shifting
df = df.sort_values([STORE_COL, PRODUCT_COL, DATE_COL]).reset_index(drop=True)

# Create next-day weather and sales columns
grp = df.groupby([STORE_COL, PRODUCT_COL])

df["weather_next"] = grp[WEATHER_COL].shift(-1)
df["sales_next"]   = grp[SALES_COL].shift(-1)

# Keep only rows where NEXT day is bad weather
df_pairs = df[df["weather_next"].isin(BAD_WEATHER)].copy()

# Focus only on top categories for consistency
df_pairs = df_pairs[df_pairs[CATEGORY_COL].isin(top_cats)]

# -----------------------------------------------------------------------
# Compute correlation per Category x Weather
# -----------------------------------------------------------------------

corr_results = []

for (cat, weather), group in df_pairs.groupby([CATEGORY_COL, "weather_next"]):
    if len(group) > 5:  # avoid unstable correlations
        corr = group[[SALES_COL, "sales_next"]].corr().iloc[0, 1]
        corr_results.append({
            "Category": cat,
            "Weather": weather,
            "Correlation": round(corr, 3)
        })

corr_df = pd.DataFrame(corr_results)

# Print results for reference
print("\nCorrelation values (Day -1 vs Bad Weather Day):")
print(corr_df.to_string(index=False))

# -----------------------------------------------------------------------
# Heatmap visualization
# -----------------------------------------------------------------------

pivot_corr = corr_df.pivot(
    index="Category",
    columns="Weather",
    values="Correlation"
)

fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(
    pivot_corr,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)

ax.set_title(
    "Correlation: Sales (1 Day Before) vs Sales (Bad Weather Day)",
    fontsize=13,
    pad=12
)
ax.set_xlabel("Bad Weather Type")
ax.set_ylabel("Category")

save_and_show(fig, "chart3_pre_vs_bad_weather_correlation.png")




# -----------------------------------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------------------------------

print("=" * 60)
print("WEEK 2 EDA SUMMARY")
print("=" * 60)

print(f"\nTop {TOP_N} categories by total units sold:")
totals = df.groupby(CATEGORY_COL)[SALES_COL].sum().nlargest(TOP_N)
for rank, (cat, total) in enumerate(totals.items(), 1):
    print(f"  {rank}. {cat:<20} {int(total):>10,} units")

print("\nPre-bad-weather pantry-loading effect (avg units, all categories):")
for lbl in label_order:
    avg = df.loc[df["pre_bad_label"] == lbl, SALES_COL].mean()
    print(f"  {lbl:<20}: {avg:.2f}")
    


# print(f"\nCorrelation of 'Promo x Bad Weather'  with Units Sold: "
#       f"{corr_matrix.loc['Promo x Bad Weather', SALES_COL]:.4f}")
# print(f"Correlation of 'Promo x Good Weather' with Units Sold: "
#       f"{corr_matrix.loc['Promo x Good Weather', SALES_COL]:.4f}")

print(f"\nAll charts saved to the '{OUTPUT_FOLDER}/' folder.")
print("Next step: run week3_weather_integration.py")
