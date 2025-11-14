import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of properties
n_properties = 1000

# Hamburg neighborhoods with base prices (€ per m²)
neighborhoods = {
    'Altona': 5500, 'Eimsbüttel': 6000, 'Hamburg-Mitte': 5000,
    'Hamburg-Nord': 4500, 'Wandsbek': 4000, 'Bergedorf': 3500,
    'Harburg': 3200, 'Rotherbaum': 7000, 'Winterhude': 6500,
    'Eppendorf': 6800, 'St. Pauli': 5800, 'Blankenese': 8000,
    'Ottensen': 6200, 'Uhlenhorst': 6400, 'St. Georg': 5300,
    'Barmbek': 4800, 'Lokstedt': 5200, 'Niendorf': 4300,
    'Poppenbüttel': 5000, 'Rahlstedt': 3800, 'Volksdorf': 5500,
    'Sasel': 4700, 'Farmsen': 4000, 'Billstedt': 3300,
    'Horn': 3600, 'Hamm': 4200, 'Dulsberg': 4100,
    'Bahrenfeld': 4900, 'Stellingen': 4400, 'Schnelsen': 4200,
    'Bramfeld': 4300, 'Steilshoop': 3700, 'Jenfeld': 3400,
    'Tonndorf': 3900, 'Marienthal': 5800, 'Wellingsbüttel': 6000,
    'Hoheluft': 6300, 'Langenhorn': 4100, 'Fuhlsbüttel': 4200,
    'Ohlsdorf': 4500, 'Alsterdorf': 5700, 'Groß Borstel': 4600
}

# Generate data
data = {
    'neighborhood': np.random.choice(list(neighborhoods.keys()), n_properties),
    'size_sqm': np.random.randint(30, 200, n_properties),
    'rooms': np.random.randint(1, 6, n_properties),
    'year_built': np.random.randint(1950, 2024, n_properties),
    'has_balcony': np.random.choice([0, 1], n_properties),
    'has_parking': np.random.choice([0, 1], n_properties),
    'floor': np.random.randint(0, 10, n_properties),
    'has_elevator': np.random.choice([0, 1], n_properties)
}

df = pd.DataFrame(data)

# Calculate price based on features
df['price_per_sqm'] = df['neighborhood'].map(neighborhoods)
df['price'] = df['price_per_sqm'] * df['size_sqm']

# Add realistic variations
df['price'] = df['price'] * (1 + np.random.uniform(-0.15, 0.15, n_properties))
df['price'] = df['price'] + (df['has_balcony'] * 15000)
df['price'] = df['price'] + (df['has_parking'] * 20000)
df['price'] = df['price'] + (df['has_elevator'] * 10000)
df['price'] = df['price'] - ((2024 - df['year_built']) * 500)

# Round prices
df['price'] = df['price'].round(0)

# Save to CSV
df.to_csv('hamburg_properties.csv', index=False)
print(f"✅ Created dataset with {len(df)} properties")
print(f"📊 Price range: €{df['price'].min():,.0f} - €{df['price'].max():,.0f}")
print(f"🏘️  Number of neighborhoods: {df['neighborhood'].nunique()}")
print("\nFirst 5 properties:")
print(df.head())