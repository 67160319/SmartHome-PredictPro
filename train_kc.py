import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import os

# 1. Load Data
df = pd.read_csv('kc_house_data.csv')

# 2. Data Cleaning
df = df.dropna()

# 3. Feature Selection
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']
X = df[features]
y = df['price']

# 4. Data Splitting (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Comparison
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_r2 = r2_score(y_test, lr.predict(X_test))

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_r2 = r2_score(y_test, rf.predict(X_test))

# 6. Select Best Model
if rf_r2 > lr_r2:
    best_model = rf
    best_name = "Random Forest"
else:
    best_model = lr
    best_name = "Linear Regression"

# 7. Save Model
if not os.path.exists('model'):
    os.makedirs('model')
joblib.dump(best_model, 'model/house_model.pkl')

print("--- Result ---")
print(f"Linear Regression R2: {lr_r2:.4f}")
print(f"Random Forest R2: {rf_r2:.4f}")
print(f"Selected: {best_name}")
print("SUCCESS: Model saved successfully!")