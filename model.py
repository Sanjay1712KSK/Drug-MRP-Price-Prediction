import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 1. Fetch data from API
response = requests.get('http://localhost:8000/products')
data = response.json()['products']
df = pd.DataFrame(data)

# 2. Clean missing values
df['name'] = df['name'].fillna('unknown')
df['pack_size'] = df['pack_size'].fillna('unknown')
df['rating'] = df['rating'].fillna(0)
df['rating_count'] = df['rating_count'].fillna(0)
df['discounted_price'] = df['discounted_price'].fillna(df['discounted_price'].mean())
df['mrp'] = df['mrp'].fillna(df['mrp'].mean())

# 3. Encode categorical columns
le_name = LabelEncoder()
le_packsize = LabelEncoder()
df['name_encoded'] = le_name.fit_transform(df['name'])
df['pack_size_encoded'] = le_packsize.fit_transform(df['pack_size'])

# 4. Prepare features and target
features = ['name_encoded', 'pack_size_encoded', 'rating', 'rating_count', 'discounted_price']
target = 'mrp'
X = df[features]
y = df[target]

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
score = model.score(X_test, y_test)
print(f"Test R² score: {score:.2f}")

# 8. Predict (optional)
predictions = model.predict(X_test[:10000])
print("Sample predictions:", predictions)

y_pred = model.predict(X_test)

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual MRP')
plt.ylabel('Predicted MRP')
plt.title('Actual vs Predicted MRP')
plt.grid(True)
plt.show()