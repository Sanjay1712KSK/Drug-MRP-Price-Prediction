import pandas as pd

# Replace with your actual CSV filename
df = pd.read_csv('your_file.csv')

print("Shape:", df.shape)
print("Columns:", df.columns)
print("Sample rows:")
print(df.head())

print("Missing values per column:")
print(df.isnull().sum())
print("Data types:")
print(df.dtypes)