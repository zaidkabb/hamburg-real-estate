import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('hamburg_properties.csv')

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"\nTotal properties: {len(df)}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

print("\n" + "=" * 50)
print("BASIC STATISTICS")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("PRICE ANALYSIS")
print("=" * 50)
print(f"Average price: €{df['price'].mean():,.0f}")
print(f"Median price: €{df['price'].median():,.0f}")
print(f"Cheapest property: €{df['price'].min():,.0f}")
print(f"Most expensive property: €{df['price'].max():,.0f}")

print("\n" + "=" * 50)
print("TOP 10 MOST EXPENSIVE NEIGHBORHOODS (avg price)")
print("=" * 50)
neighborhood_avg = df.groupby('neighborhood')['price'].mean().sort_values(ascending=False).head(10)
for hood, price in neighborhood_avg.items():
    print(f"{hood}: €{price:,.0f}")

print("\n" + "=" * 50)
print("PROPERTY SIZE DISTRIBUTION")
print("=" * 50)
print(df['size_sqm'].describe())

print("\n" + "=" * 50)
print("ROOMS DISTRIBUTION")
print("=" * 50)
print(df['rooms'].value_counts().sort_index())

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Price distribution
axes[0, 0].hist(df['price'], bins=50, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Price Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Price (€)')
axes[0, 0].set_ylabel('Frequency')

# 2. Price vs Size
axes[0, 1].scatter(df['size_sqm'], df['price'], alpha=0.5, color='coral')
axes[0, 1].set_title('Price vs Size', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Size (sqm)')
axes[0, 1].set_ylabel('Price (€)')

# 3. Top 10 neighborhoods by average price
top_neighborhoods = df.groupby('neighborhood')['price'].mean().sort_values(ascending=False).head(10)
axes[1, 0].barh(range(len(top_neighborhoods)), top_neighborhoods.values, color='lightgreen')
axes[1, 0].set_yticks(range(len(top_neighborhoods)))
axes[1, 0].set_yticklabels(top_neighborhoods.index)
axes[1, 0].set_title('Top 10 Most Expensive Neighborhoods', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Average Price (€)')

# 4. Rooms distribution
room_counts = df['rooms'].value_counts().sort_index()
axes[1, 1].bar(room_counts.index, room_counts.values, color='plum', edgecolor='black')
axes[1, 1].set_title('Number of Rooms Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Number of Rooms')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved as 'data_exploration.png'")
print("\nCheck your folder for the image!")

plt.show()