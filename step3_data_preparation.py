from sklearn.preprocessing import LabelEncoder

# Fill missing values
df['rating'] = df['rating'].fillna(0)
df['discounted_price'] = df['discounted_price'].fillna(df['discounted_price'].mean())
df['pack_size'] = df['pack_size'].fillna('unknown')
df['name'] = df['name'].fillna('unknown')

# Encode categorical columns
le_name = LabelEncoder()
le_packsize = LabelEncoder()
df['name_encoded'] = le_name.fit_transform(df['name'])
df['pack_size_encoded'] = le_packsize.fit_transform(df['pack_size'])

# Handle time column if present
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month.fillna(-1)
    df['day'] = df['date'].dt.day.fillna(-1)
    df['weekday'] = df['date'].dt.weekday.fillna(-1)

# Choose features and target
features = ['name_encoded', 'pack_size_encoded', 'rating', 'discounted_price']
if 'month' in df.columns:
    features += ['month', 'day', 'weekday']

# Confirm target
target = 'sales_quantity' if 'sales_quantity' in df.columns else 'demand'
X = df[features]
y = df[target]

print("Features for training:", features)
print("Target:", target)
print(X.head())
print(y.head())