import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Fetch data from API
response = requests.get('http://localhost:8000/products')
data = response.json()['products']
df = pd.DataFrame(data)

# Confirm columns present
print("Columns:", df.columns)
# Example columns: ['name', 'pack_size', 'rating', 'discounted_price', 'sales_quantity', 'date']

# Handle missing values
df['rating'] = df['rating'].fillna(0)
df['discounted_price'] = df['discounted_price'].fillna(df['discounted_price'].mean())
df['pack_size'] = df['pack_size'].fillna('unknown')
df['name'] = df['name'].fillna('unknown')
df['sales_quantity'] = df['sales_quantity'].fillna(0)

# Encode categorical features
for col in ['name', 'pack_size']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Handle time column if present
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Optional: extract date features
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    # Fill missing date features
    df['month'] = df['month'].fillna(-1)
    df['day'] = df['day'].fillna(-1)
    df['weekday'] = df['weekday'].fillna(-1)

# Final features for model
features = ['name', 'pack_size', 'rating', 'discounted_price']
if 'month' in df.columns:
    features += ['month', 'day', 'weekday']

X = df[features]
y = df['sales_quantity']

print(X.head(), y.head())