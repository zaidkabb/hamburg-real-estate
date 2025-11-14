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
print("STEP 1: LOAD AND PREPARE DATA")
print("=" * 60)

# Load data
df = pd.read_csv('hamburg_properties.csv')
print(f"✅ Loaded {len(df)} properties")

# Prepare features
print("\n📋 Original columns:", list(df.columns))

# Encode neighborhood (convert text to numbers)
le = LabelEncoder()
df['neighborhood_encoded'] = le.fit_transform(df['neighborhood'])
print(f"✅ Encoded {len(le.classes_)} neighborhoods into numbers")

# Select features for the model
features = ['size_sqm', 'rooms', 'year_built', 'has_balcony', 
            'has_parking', 'floor', 'has_elevator', 'neighborhood_encoded']
target = 'price'

X = df[features]
y = df[target]

print(f"\n📊 Features we're using to predict price:")
for i, feature in enumerate(features, 1):
    print(f"  {i}. {feature}")

print("=" * 60)
print("STEP 2: SPLIT DATA (TRAIN/TEST)")
print("=" * 60)

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Training set: {len(X_train)} properties")
print(f"✅ Testing set: {len(X_test)} properties")
print(f"\n💡 We'll train on {len(X_train)} properties and test on {len(X_test)}")

print("\n" + "=" * 60)
print("STEP 3: TRAIN MODELS")
print("=" * 60)

# Dictionary to store models and their predictions
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
print("STEP 4: EVALUATE MODELS")
print("=" * 60)

# Evaluate each model
for name, pred in predictions.items():
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    scores[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    print(f"\n📊 {name}:")
    print(f"   MAE (Mean Absolute Error): €{mae:,.0f}")
    print(f"   RMSE (Root Mean Squared Error): €{rmse:,.0f}")
    print(f"   R² Score: {r2:.4f} ({r2*100:.2f}%)")

# Find best model
best_model_name = max(scores, key=lambda x: scores[x]['R2'])
print(f"\n🏆 BEST MODEL: {best_model_name} (R² = {scores[best_model_name]['R2']:.4f})")

# Save the best model
best_model = models[best_model_name]
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print(f"\n💾 Saved best model as 'best_model.pkl'")

print("\n" + "=" * 60)
print("STEP 5: VISUALIZE PREDICTIONS")
print("=" * 60)

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, pred) in enumerate(predictions.items()):
    axes[idx].scatter(y_test, pred, alpha=0.5, color=['blue', 'green', 'red'][idx])
    axes[idx].plot([y_test.min(), y_test.max()], 
                   [y_test.min(), y_test.max()], 
                   'k--', lw=2, label='Perfect Prediction')
    axes[idx].set_xlabel('Actual Price (€)', fontsize=12)
    axes[idx].set_ylabel('Predicted Price (€)', fontsize=12)
    axes[idx].set_title(f'{name}\nR² = {scores[name]["R2"]:.4f}', 
                       fontsize=14, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved comparison plot as 'model_comparison.png'")

plt.show()

print("\n" + "=" * 60)
print("🎉 MODEL TRAINING COMPLETE!")
print("=" * 60)