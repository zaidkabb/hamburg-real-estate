import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of properties
n_properties = 1000

# Hamburg neighborhoods with base rental prices (€ per m² per month)
neighborhoods = {
    'Altona': 15, 'Eimsbüttel': 16, 'Hamburg-Mitte': 14,
    'Hamburg-Nord': 13, 'Wandsbek': 12, 'Bergedorf': 10,
    'Harburg': 9, 'Rotherbaum': 18, 'Winterhude': 17,
    'Eppendorf': 17.5, 'St. Pauli': 16, 'Blankenese': 20,
    'Ottensen': 16.5, 'Uhlenhorst': 17, 'St. Georg': 15,
    'Barmbek': 13.5, 'Lokstedt': 14, 'Niendorf': 12,
    'Poppenbüttel': 13.5, 'Rahlstedt': 11, 'Volksdorf': 14.5,
    'Sasel': 13, 'Farmsen': 11.5, 'Billstedt': 9.5,
    'Horn': 10.5, 'Hamm': 12, 'Dulsberg': 11.5,
    'Bahrenfeld': 13.5, 'Stellingen': 12.5, 'Schnelsen': 12,
    'Bramfeld': 12, 'Steilshoop': 10.5, 'Jenfeld': 10,
    'Tonndorf': 11, 'Marienthal': 15.5, 'Wellingsbüttel': 16,
    'Hoheluft': 16.5, 'Langenhorn': 11.5, 'Fuhlsbüttel': 12,
    'Ohlsdorf': 12.5, 'Alsterdorf': 15, 'Groß Borstel': 13
}

print("=" * 60)
print("🏠 CREATING RENTAL DATASET FOR HAMBURG")
print("=" * 60)

# Generate data
data = {
    'neighborhood': np.random.choice(list(neighborhoods.keys()), n_properties),
    'size_sqm': np.random.randint(30, 200, n_properties),
    'rooms': np.random.randint(1, 6, n_properties),
    'year_built': np.random.randint(1950, 2024, n_properties),
    'has_balcony': np.random.choice([0, 1], n_properties),
    'has_parking': np.random.choice([0, 1], n_properties),
    'floor': np.random.randint(0, 10, n_properties),
    'has_elevator': np.random.choice([0, 1], n_properties),
    'is_furnished': np.random.choice([0, 1], n_properties, p=[0.7, 0.3])  # 30% furnished
}

df = pd.DataFrame(data)

# Calculate rental price based on features
df['rent_per_sqm'] = df['neighborhood'].map(neighborhoods)
df['monthly_rent'] = df['rent_per_sqm'] * df['size_sqm']

# Add realistic variations
df['monthly_rent'] = df['monthly_rent'] * (1 + np.random.uniform(-0.12, 0.12, n_properties))

# Add premiums for features
df['monthly_rent'] = df['monthly_rent'] + (df['has_balcony'] * 50)
df['monthly_rent'] = df['monthly_rent'] + (df['has_parking'] * 80)
df['monthly_rent'] = df['monthly_rent'] + (df['has_elevator'] * 30)
df['monthly_rent'] = df['monthly_rent'] + (df['is_furnished'] * 200)  # Furnished premium

# Newer buildings cost more
df['monthly_rent'] = df['monthly_rent'] + ((df['year_built'] - 1950) * 0.5)

# Round prices
df['monthly_rent'] = df['monthly_rent'].round(0)

# Add utilities estimate (Nebenkosten) - typically €2-3.5 per sqm
df['utilities'] = (df['size_sqm'] * np.random.uniform(2, 3.5, n_properties)).round(0)
df['total_rent'] = df['monthly_rent'] + df['utilities']

# Save to CSV
df.to_csv('hamburg_rentals.csv', index=False)

print(f"\n✅ Created rental dataset with {len(df)} properties")
print(f"📊 Monthly rent range: €{df['monthly_rent'].min():,.0f} - €{df['monthly_rent'].max():,.0f}")
print(f"📊 Total rent (incl. utilities) range: €{df['total_rent'].min():,.0f} - €{df['total_rent'].max():,.0f}")
print(f"💡 Average monthly rent: €{df['monthly_rent'].mean():,.0f}")
print(f"💡 Average total rent: €{df['total_rent'].mean():,.0f}")

print("\n" + "=" * 60)
print("TOP 10 MOST EXPENSIVE NEIGHBORHOODS (avg monthly rent)")
print("=" * 60)
neighborhood_avg = df.groupby('neighborhood')['monthly_rent'].mean().sort_values(ascending=False).head(10)
for hood, rent in neighborhood_avg.items():
    print(f"{hood}: €{rent:,.0f}/month")

print("\n" + "=" * 60)
print("FURNISHED vs UNFURNISHED")
print("=" * 60)
furnished_avg = df.groupby('is_furnished')['monthly_rent'].mean()
print(f"Unfurnished: €{furnished_avg[0]:,.0f}/month")
print(f"Furnished: €{furnished_avg[1]:,.0f}/month")
print(f"Furnished premium: €{furnished_avg[1] - furnished_avg[0]:,.0f}/month")

print("\n✅ Dataset saved as 'hamburg_rentals.csv'")
print("\nFirst 5 properties:")
print(df[['neighborhood', 'size_sqm', 'rooms', 'monthly_rent', 'utilities', 'total_rent']].head())