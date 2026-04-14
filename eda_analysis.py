"""
Exploratory Data Analysis: Sales by Category (2024-2025)
=========================================================================
Focus:
1. Average sales trend over time (line graph)
2. Weather correlation scatter plots (6 categories)
3. Revenue contribution (ABC Pareto analysis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
import sys
import os

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings("ignore")

# =========================================================================
# CONFIG
# =========================================================================
CSV_PATH_2024 = "inventory_demand_data_2024.csv"
CSV_PATH_2025 = "inventory_demand_data.csv"
OUTPUT_FOLDER = "eda_charts"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================================================================
# 1. LOAD & COMBINE DATA
# =========================================================================
print("\n" + "="*70)
print("LOADING DATA (2024-2025 Combined)")
print("="*70)

df_2024 = pd.read_csv(CSV_PATH_2024, parse_dates=["ds"])
df_2024.rename(columns={"ds": "date"}, inplace=True)
print(f"[OK] Loaded 2024 data: {len(df_2024):,} rows")

df_2025 = pd.read_csv(CSV_PATH_2025, parse_dates=["Date"])
df_2025.rename(columns={"Date": "date"}, inplace=True)
print(f"[OK] Loaded 2025 data: {len(df_2025):,} rows")

df = pd.concat([df_2024, df_2025], ignore_index=True)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.sort_values("date").reset_index(drop=True)

print(f"[OK] Combined dataset: {len(df):,} rows")
print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# =========================================================================
# 2. AVERAGE SALES TREND OVER TIME (Line graph only)
# =========================================================================
print("\n" + "="*70)
print("VISUALIZATION 1: AVERAGE SALES TREND BY CATEGORY (2024-2025)")
print("="*70)

df['year_month'] = df['date'].dt.to_period('M')
ts_data = df.groupby(['year_month', 'category'])['sales_volume'].mean().reset_index()
ts_data['year_month'] = ts_data['year_month'].astype(str)

fig, ax = plt.subplots(figsize=(14, 6))

for category in sorted(df['category'].unique()):
    cat_ts = ts_data[ts_data['category'] == category]
    ax.plot(cat_ts['year_month'], cat_ts['sales_volume'], marker='o', label=category, linewidth=2)

ax.set_xlabel('Year-Month', fontweight='bold', fontsize=11)
ax.set_ylabel('Average Sales (units)', fontweight='bold', fontsize=11)
ax.set_title('Average Sales Trend by Category (2024-2025)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9, ncol=2)
ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/01_sales_trend_by_category.png", dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {OUTPUT_FOLDER}/01_sales_trend_by_category.png")
plt.show()
plt.close()

# =========================================================================
# 3. WEATHER CORRELATION SCATTER PLOTS (Temperature vs Sales)
# =========================================================================
print("\n" + "="*70)
print("VISUALIZATION 2: WEATHER CORRELATION SCATTER PLOTS")
print("="*70)

categories_to_analyze = [
    'Hot Beverages',
    'Cold Beverages',
    'Fruits - Seasonal',
    'Fruits - Berries',
    'Meat',
    'Seafood',
    'Frozen'
]

colors_map = {
    'Hot Beverages': 'coral',
    'Cold Beverages': 'skyblue',
    'Fruits - Seasonal': 'orange',
    'Fruits - Berries': 'purple',
    'Meat': 'brown',
    'Seafood': 'teal',
    'Frozen': 'lightblue'
}

print(f"\nGenerating {len(categories_to_analyze)} scatter plots...")

for category in categories_to_analyze:
    cat_data = df[df['category'] == category].copy()
    
    if len(cat_data) > 0:
        print(f"\n   Processing: {category}")
        
        corr_temp, p_temp = pearsonr(cat_data['temperature_f'], 
                                     cat_data['sales_volume'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        color = colors_map.get(category, 'steelblue')
        ax.scatter(cat_data['temperature_f'], cat_data['sales_volume'],
                  alpha=0.6, s=40, color=color, edgecolor='black', linewidth=0.5)
        
        # Add regression line
        z = np.polyfit(cat_data['temperature_f'], 
                       cat_data['sales_volume'], 1)
        p = np.poly1d(z)
        x_line = np.array([cat_data['temperature_f'].min(), 
                          cat_data['temperature_f'].max()])
        ax.plot(x_line, p(x_line), "r--", linewidth=2.5, 
               label=f'Trend (r={corr_temp:.3f})')
        
        ax.set_xlabel('Temperature (degF)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Sales Volume (units)', fontweight='bold', fontsize=11)
        ax.set_title(f'{category} Sales vs Temperature\nCorrelation: r={corr_temp:.4f}, p={p_temp:.6f}',
                    fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        filename = category.replace(' - ', '_').replace(' ', '_').lower()
        filepath = f"{OUTPUT_FOLDER}/02_weather_scatter_{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"      Saved: {filepath}")
        plt.close()
        
        sig_text = "SIGNIFICANT" if p_temp < 0.05 else "NOT SIGNIFICANT"
        print(f"      Temp: {cat_data['temperature_f'].min():.1f}-{cat_data['temperature_f'].max():.1f}F, Records: {len(cat_data)}, r={corr_temp:.4f}, {sig_text}")

# =========================================================================
# 4. REVENUE CONTRIBUTION: ABC PARETO ANALYSIS
# =========================================================================
print("\n" + "="*70)
print("VISUALIZATION 3: REVENUE CONTRIBUTION - ABC PARETO ANALYSIS")
print("="*70)

df['revenue'] = df['unit_price'] * df['sales_volume']

revenue_by_category = df.groupby('category').agg({
    'revenue': 'sum',
    'sales_volume': 'sum',
    'unit_price': 'mean'
}).sort_values('revenue', ascending=False)

revenue_by_category['cumulative_revenue'] = revenue_by_category['revenue'].cumsum()
revenue_by_category['cumulative_pct'] = (revenue_by_category['cumulative_revenue'] / 
                                         revenue_by_category['revenue'].sum()) * 100
revenue_by_category['revenue_pct'] = (revenue_by_category['revenue'] / 
                                      revenue_by_category['revenue'].sum()) * 100

print("\nRevenue Contribution by Category:")
print(revenue_by_category[['revenue', 'revenue_pct', 'cumulative_pct']].round(2).to_string())

# Visualization: Pareto chart
fig, ax1 = plt.subplots(figsize=(12, 6))

categories = revenue_by_category.index
x_pos = np.arange(len(categories))
bars = ax1.bar(x_pos, revenue_by_category['revenue'], color='steelblue', 
              edgecolor='black', alpha=0.7, label='Revenue')

# Assign colors based on cumulative percentage (80% rule)
for i, (idx, row) in enumerate(revenue_by_category.iterrows()):
    if row['cumulative_pct'] <= 80:
        bars[i].set_color('darkgreen')
    elif row['cumulative_pct'] <= 95:
        bars[i].set_color('orange')
    else:
        bars[i].set_color('red')

ax1.set_xlabel('Category', fontweight='bold', fontsize=11)
ax1.set_ylabel('Total Revenue ($)', fontweight='bold', fontsize=11)
ax1.set_title('Pareto Analysis: Revenue Contribution by Category (ABC Classification)',
             fontsize=13, fontweight='bold', pad=20)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(categories, rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'${height/1000:.1f}K\n({revenue_by_category["revenue_pct"].iloc[i]:.1f}%)',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

# Line plot (cumulative %)
ax2 = ax1.twinx()
ax2.plot(x_pos, revenue_by_category['cumulative_pct'], color='red', marker='o', 
        linewidth=2.5, markersize=6, label='Cumulative %')
ax2.set_ylabel('Cumulative Revenue %', fontweight='bold', fontsize=11, color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Add 80% line reference
ax2.axhline(80, color='green', linestyle='--', linewidth=2, alpha=0.7, label='80% Threshold')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/03_revenue_contribution_abc_pareto.png", dpi=300, bbox_inches='tight')
print(f"\n[OK] Saved: {OUTPUT_FOLDER}/03_revenue_contribution_abc_pareto.png")
plt.show()
plt.close()

# =========================================================================
# 5. STOCK-OUT RISK: DAYS OF COVER HISTOGRAM
# =========================================================================
print("\n" + "="*70)
print("VISUALIZATION 4: STOCK-OUT RISK - DAYS OF COVER HISTOGRAM")
print("="*70)

# Calculate average daily sales per category
avg_daily_sales = df.groupby('category')['sales_volume'].mean()

# Create Days of Cover metric
df['avg_daily_sales_cat'] = df['category'].map(avg_daily_sales)
df['days_of_cover'] = df['stock_quantity'] / (df['avg_daily_sales_cat'].clip(lower=0.1))

print(f"\nDays of Cover Statistics:")
print(f"  Mean: {df['days_of_cover'].mean():.2f} days")
print(f"  Median: {df['days_of_cover'].median():.2f} days")
print(f"  Min: {df['days_of_cover'].min():.2f} days")
print(f"  Max: {df['days_of_cover'].max():.2f} days")
print(f"  Std Dev: {df['days_of_cover'].std():.2f} days")

# Risk categories
critical = len(df[df['days_of_cover'] < 2])
high_risk = len(df[(df['days_of_cover'] >= 2) & (df['days_of_cover'] < 5)])
medium = len(df[(df['days_of_cover'] >= 5) & (df['days_of_cover'] < 10)])
healthy = len(df[df['days_of_cover'] >= 10])

print(f"\nRisk Distribution:")
print(f"  Critical (< 2 days): {critical:,} ({100*critical/len(df):.1f}%)")
print(f"  High (2-5 days): {high_risk:,} ({100*high_risk/len(df):.1f}%)")
print(f"  Medium (5-10 days): {medium:,} ({100*medium/len(df):.1f}%)")
print(f"  Healthy (>= 10 days): {healthy:,} ({100*healthy/len(df):.1f}%)")

# Create histogram
fig, ax = plt.subplots(figsize=(12, 6))

counts, bins, patches = ax.hist(df['days_of_cover'], bins=50, color='steelblue', 
                                edgecolor='black', alpha=0.7)

# Color-code bins by risk level
for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i+1]) / 2
    if bin_center < 2:
        patch.set_facecolor('red')
        patch.set_alpha(0.8)
    elif bin_center < 5:
        patch.set_facecolor('orange')
        patch.set_alpha(0.8)
    elif bin_center < 10:
        patch.set_facecolor('yellow')
        patch.set_alpha(0.8)
    else:
        patch.set_facecolor('green')
        patch.set_alpha(0.8)

# Add reference lines
ax.axvline(2, color='red', linestyle='--', linewidth=2.5, label='Critical threshold (2 days)', alpha=0.8)
ax.axvline(5, color='orange', linestyle='--', linewidth=2.5, label='High-risk threshold (5 days)', alpha=0.8)
ax.axvline(df['days_of_cover'].mean(), color='purple', linestyle='-', linewidth=2.5, 
          label=f'Mean ({df["days_of_cover"].mean():.1f} days)', alpha=0.8)

ax.set_xlabel('Days of Cover', fontweight='bold', fontsize=11)
ax.set_ylabel('Frequency (# of records)', fontweight='bold', fontsize=11)
ax.set_title('Stock-Out Risk Assessment: Days of Cover Distribution\n(Red=Critical, Orange=High, Yellow=Medium, Green=Healthy)',
            fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/04_stock_out_risk_days_of_cover.png", dpi=300, bbox_inches='tight')
print(f"\n[OK] Saved: {OUTPUT_FOLDER}/04_stock_out_risk_days_of_cover.png")
plt.show()
plt.close()

# =========================================================================
# 6. WEATHER IMPACT ON TOP 3 CATEGORIES (With & Without Discounts/Promotions)
# =========================================================================
print("\n" + "="*70)
print("VISUALIZATION 5: TOP 3 CATEGORIES SALES - WEATHER IMPACT (WITH/WITHOUT DISCOUNTS)")
print("="*70)

# Get top 3 categories by average sales
top_3_categories = df.groupby('category')['sales_volume'].mean().nlargest(3).index.tolist()
print(f"\nTop 3 Categories by Average Sales: {', '.join(top_3_categories)}")

# Filter data: with and without discounts
df_no_discount = df[df['discount_pct'] == 0].copy()
df_with_discount = df[df['discount_pct'] > 0].copy()

print(f"  Data without discounts: {len(df_no_discount):,} records")
print(f"  Data with discounts: {len(df_with_discount):,} records")

# ===== GRAPH 1: WITHOUT DISCOUNTS =====
print(f"\nGenerating Graph 1: Without Discounts/Promotions")

fig, ax = plt.subplots(figsize=(12, 6))

weather_conditions = sorted(df_no_discount['weather_condition'].unique())
x_pos = np.arange(len(top_3_categories))
width = 0.12  # Reduced bar thickness
spacing = 0.08  # Increased spacing between category groups

for i, weather in enumerate(weather_conditions):
    weather_data = df_no_discount[df_no_discount['weather_condition'] == weather]
    avg_sales = [weather_data[weather_data['category'] == cat]['sales_volume'].mean() 
                 for cat in top_3_categories]
    offset = (i - len(weather_conditions)/2) * width + (i * spacing / len(weather_conditions))
    ax.bar(x_pos + offset, avg_sales, width, label=weather, edgecolor='black', linewidth=0.7, alpha=0.85)

ax.set_xlabel('Category', fontweight='bold', fontsize=12)
ax.set_ylabel('Avg Units Sold per Day', fontweight='bold', fontsize=12)
ax.set_title('Average Daily Units Sold by Category and Weather\nWITHOUT Promotion or Discount', 
            fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(top_3_categories, fontsize=11)
ax.legend(title='Weather', fontsize=10, title_fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/05_top3_weather_no_discount.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_FOLDER}/05_top3_weather_no_discount.png")
plt.show()
plt.close()

# ===== GRAPH 2: WITH DISCOUNTS =====
print(f"Generating Graph 2: With Discounts/Promotions")

fig, ax = plt.subplots(figsize=(12, 6))

weather_conditions_promo = sorted(df_with_discount['weather_condition'].unique())
x_pos = np.arange(len(top_3_categories))
width = 0.12  # Reduced bar thickness
spacing = 0.08  # Increased spacing between category groups

for i, weather in enumerate(weather_conditions_promo):
    weather_data = df_with_discount[df_with_discount['weather_condition'] == weather]
    avg_sales = [weather_data[weather_data['category'] == cat]['sales_volume'].mean() 
                 for cat in top_3_categories]
    offset = (i - len(weather_conditions_promo)/2) * width + (i * spacing / len(weather_conditions_promo))
    ax.bar(x_pos + offset, avg_sales, width, label=weather, edgecolor='black', linewidth=0.7, alpha=0.85)

ax.set_xlabel('Category', fontweight='bold', fontsize=12)
ax.set_ylabel('Avg Units Sold per Day', fontweight='bold', fontsize=12)
ax.set_title('Average Daily Units Sold by Category and Weather\nWITH Promotion or Discount', 
            fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(top_3_categories, fontsize=11)
ax.legend(title='Weather', fontsize=10, title_fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/05_top3_weather_with_discount.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_FOLDER}/05_top3_weather_with_discount.png")
plt.show()
plt.close()

# ===== COMBINED PANEL: WITH AND WITHOUT DISCOUNTS =====
print(f"Generating Combined Panel: Without and With Discounts")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

x_pos = np.arange(len(top_3_categories))
width = 0.12
spacing = 0.08

# Panel 1: WITHOUT DISCOUNTS
for i, weather in enumerate(weather_conditions):
    weather_data = df_no_discount[df_no_discount['weather_condition'] == weather]
    avg_sales = [weather_data[weather_data['category'] == cat]['sales_volume'].mean() 
                 for cat in top_3_categories]
    offset = (i - len(weather_conditions)/2) * width + (i * spacing / len(weather_conditions))
    ax1.bar(x_pos + offset, avg_sales, width, label=weather, edgecolor='black', linewidth=0.7, alpha=0.85)

ax1.set_xlabel('Category', fontweight='bold', fontsize=12)
ax1.set_ylabel('Avg Units Sold per Day', fontweight='bold', fontsize=12)
ax1.set_title('WITHOUT Promotion or Discount', fontsize=13, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_ylim([70, 130])
ax1.set_xticklabels(top_3_categories, fontsize=11)
ax1.legend(title='Weather', fontsize=10, title_fontsize=11, loc='upper right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# Panel 2: WITH DISCOUNTS
for i, weather in enumerate(weather_conditions_promo):
    weather_data = df_with_discount[df_with_discount['weather_condition'] == weather]
    avg_sales = [weather_data[weather_data['category'] == cat]['sales_volume'].mean() 
                 for cat in top_3_categories]
    offset = (i - len(weather_conditions_promo)/2) * width + (i * spacing / len(weather_conditions_promo))
    ax2.bar(x_pos + offset, avg_sales, width, label=weather, edgecolor='black', linewidth=0.7, alpha=0.85)

ax2.set_xlabel('Category', fontweight='bold', fontsize=12)
ax2.set_ylabel('Avg Units Sold per Day', fontweight='bold', fontsize=12)
ax2.set_title('WITH Promotion or Discount', fontsize=13, fontweight='bold', pad=15)
ax2.set_xticks(x_pos)
ax2.set_ylim([70, 130])
ax2.set_xticklabels(top_3_categories, fontsize=11)
ax2.legend(title='Weather', fontsize=10, title_fontsize=11, loc='upper right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/05_top3_weather_comparison_panel.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_FOLDER}/05_top3_weather_comparison_panel.png")
plt.show()
plt.close()

# Print summary statistics
print(f"\nSales Comparison - Top 3 Categories:")
for category in top_3_categories:
    cat_no_disc = df_no_discount[df_no_discount['category'] == category]['sales_volume'].mean()
    cat_with_disc = df_with_discount[df_with_discount['category'] == category]['sales_volume'].mean()
    lift = ((cat_with_disc - cat_no_disc) / cat_no_disc * 100) if cat_no_disc > 0 else 0
    print(f"  {category}:")
    print(f"    No Discount: {cat_no_disc:.2f} units/day | With Discount: {cat_with_disc:.2f} units/day | Lift: {lift:+.1f}%")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "="*70)
print("ENHANCED EDA COMPLETE")
print("="*70)
print("\nGenerated Visualizations:")
print("  1. 01_sales_trend_by_category.png - Line graph of avg sales over time")
print("  2. 02_weather_scatter_*.png - 7 temperature correlation scatter plots")
print("  3. 03_revenue_contribution_abc_pareto.png - Pareto revenue analysis")
print("  4. 04_stock_out_risk_days_of_cover.png - Days of Cover histogram")
print("  5. 05_top3_weather_no_discount.png - Top 3 categories weather impact (no discounts)")
print("  6. 05_top3_weather_with_discount.png - Top 3 categories weather impact (with discounts)")
print("  7. 05_top3_weather_comparison_panel.png - Side-by-side comparison panel")
print("\n[OK] All visualizations saved to 'eda_charts' folder")
print("="*70 + "\n")
