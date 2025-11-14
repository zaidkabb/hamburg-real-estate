import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

print("=" * 60)
print("🏠 RENTAL PRICE MODEL TRAINING")
print("=" * 60)

# Load data
df = pd.read_csv('hamburg_rentals.csv')
print(f"✅ Loaded {len(df)} rental properties")

# Encode neighborhood
le = LabelEncoder()
df['neighborhood_encoded'] = le.fit_transform(df['neighborhood'])
print(f"✅ Encoded {len(le.classes_)} neighborhoods")

# Select features (we'll predict monthly_rent, not including utilities)
features = ['size_sqm', 'rooms', 'year_built', 'has_balcony', 
            'has_parking', 'floor', 'has_elevator', 'is_furnished', 
            'neighborhood_encoded']
target = 'monthly_rent'

X = df[features]
y = df[target]

print(f"\n📊 Features for predicting MONTHLY RENT:")
for i, feature in enumerate(features, 1):
    print(f"  {i}. {feature}")

print("\n" + "=" * 60)
print("STEP 1: SPLIT DATA (80% TRAIN / 20% TEST)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Training set: {len(X_train)} properties")
print(f"✅ Testing set: {len(X_test)} properties")

print("\n" + "=" * 60)
print("STEP 2: TRAIN MODELS")
print("=" * 60)

# Store models and predictions
models = {}
predictions = {}
scores = {}

# Model 1: Linear Regression
print("\n🔷 Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
models['Linear Regression'] = lr_model
predictions['Linear Regression'] = lr_pred

# Model 2: Random Forest
print("🔷 Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
models['Random Forest'] = rf_model
predictions['Random Forest'] = rf_pred

# Model 3: XGBoost
print("🔷 Training XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
models['XGBoost'] = xgb_model
predictions['XGBoost'] = xgb_pred

print("\n" + "=" * 60)
print("STEP 3: EVALUATE MODELS")
print("=" * 60)

# Evaluate each model
for name, pred in predictions.items():
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    scores[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    print(f"\n📊 {name}:")
    print(f"   MAE (Mean Absolute Error): €{mae:,.0f}/month")
    print(f"   RMSE (Root Mean Squared Error): €{rmse:,.0f}/month")
    print(f"   R² Score: {r2:.4f} ({r2*100:.2f}%)")

# Find best model
best_model_name = max(scores, key=lambda x: scores[x]['R2'])
print(f"\n🏆 BEST MODEL: {best_model_name} (R² = {scores[best_model_name]['R2']:.4f})")

# Save the best model
best_model = models[best_model_name]
joblib.dump(best_model, 'rental_model.pkl')
joblib.dump(le, 'rental_label_encoder.pkl')
print(f"\n💾 Saved model as 'rental_model.pkl'")
print(f"💾 Saved encoder as 'rental_label_encoder.pkl'")

print("\n" + "=" * 60)
print("STEP 4: VISUALIZE PREDICTIONS")
print("=" * 60)

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, pred) in enumerate(predictions.items()):
    axes[idx].scatter(y_test, pred, alpha=0.5, color=['blue', 'green', 'red'][idx])
    axes[idx].plot([y_test.min(), y_test.max()], 
                   [y_test.min(), y_test.max()], 
                   'k--', lw=2, label='Perfect Prediction')
    axes[idx].set_xlabel('Actual Rent (€/month)', fontsize=12)
    axes[idx].set_ylabel('Predicted Rent (€/month)', fontsize=12)
    axes[idx].set_title(f'{name}\nR² = {scores[name]["R2"]:.4f}', 
                       fontsize=14, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rental_model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved comparison plot as 'rental_model_comparison.png'")

plt.show()

print("\n" + "=" * 60)
print("🎉 RENTAL MODEL TRAINING COMPLETE!")
print("=" * 60)

# Show some example predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

sample_indices = np.random.choice(len(X_test), 5, replace=False)
for idx in sample_indices:
    actual = y_test.iloc[idx]
    predicted = best_model.predict(X_test.iloc[idx:idx+1])[0]
    diff = abs(actual - predicted)
    
    print(f"\n🏠 Property #{idx}")
    print(f"   Actual rent: €{actual:,.0f}/month")
    print(f"   Predicted: €{predicted:,.0f}/month")
    print(f"   Difference: €{diff:,.0f}")