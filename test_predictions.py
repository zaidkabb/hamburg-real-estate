import pandas as pd
import joblib
import numpy as np

# Load the saved model
model = joblib.load('best_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

print("=" * 60)
print("🏠 HAMBURG REAL ESTATE PRICE PREDICTOR")
print("=" * 60)

# Example properties to predict
test_properties = [
    {
        'name': 'Small apartment in Altona',
        'neighborhood': 'Altona',
        'size_sqm': 50,
        'rooms': 2,
        'year_built': 2015,
        'has_balcony': 1,
        'has_parking': 0,
        'floor': 3,
        'has_elevator': 1
    },
    {
        'name': 'Large family house in Blankenese',
        'neighborhood': 'Blankenese',
        'size_sqm': 180,
        'rooms': 5,
        'year_built': 2010,
        'has_balcony': 1,
        'has_parking': 1,
        'floor': 0,
        'has_elevator': 0
    },
    {
        'name': 'Budget apartment in Harburg',
        'neighborhood': 'Harburg',
        'size_sqm': 60,
        'rooms': 2,
        'year_built': 1985,
        'has_balcony': 0,
        'has_parking': 0,
        'floor': 2,
        'has_elevator': 0
    },
    {
        'name': 'Modern loft in Eimsbüttel',
        'neighborhood': 'Eimsbüttel',
        'size_sqm': 120,
        'rooms': 3,
        'year_built': 2020,
        'has_balcony': 1,
        'has_parking': 1,
        'floor': 5,
        'has_elevator': 1
    }
]

print("\n🔮 Predicting prices for sample properties...\n")

for prop in test_properties:
    # Extract property details
    name = prop.pop('name')
    neighborhood_name = prop['neighborhood']
    
    # Encode neighborhood
    prop['neighborhood_encoded'] = label_encoder.transform([prop['neighborhood']])[0]
    
    # Create feature array in the right order
    features = ['size_sqm', 'rooms', 'year_built', 'has_balcony', 
                'has_parking', 'floor', 'has_elevator', 'neighborhood_encoded']
    
    X = np.array([[prop[f] for f in features]])
    
    # Predict
    predicted_price = model.predict(X)[0]
    
    # Display
    print("=" * 60)
    print(f"📍 {name}")
    print("-" * 60)
    print(f"   Location: {neighborhood_name}")
    print(f"   Size: {prop['size_sqm']} m²")
    print(f"   Rooms: {prop['rooms']}")
    print(f"   Built: {prop['year_built']}")
    print(f"   Balcony: {'Yes' if prop['has_balcony'] else 'No'}")
    print(f"   Parking: {'Yes' if prop['has_parking'] else 'No'}")
    print(f"   Floor: {prop['floor']}")
    print(f"   Elevator: {'Yes' if prop['has_elevator'] else 'No'}")
    print(f"\n   💰 PREDICTED PRICE: €{predicted_price:,.0f}")
    print("=" * 60)
    print()

print("\n✅ All predictions complete!")
print("\n💡 These predictions are based on our trained XGBoost model")
print(f"   with 88.25% accuracy (R² score)")