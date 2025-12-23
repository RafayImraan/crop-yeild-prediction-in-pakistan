import pandas as pd

# Load CSV
df = pd.read_csv("data/crop_yield_dataset.csv")

# Strip column spaces
df.columns = df.columns.str.strip()

print("Initial shape:", df.shape)
print("Value types:")
print(df.dtypes)

# Convert Year to int
df['Year'] = df['Year'].astype(int)

# Convert Value to float
df['Value'] = df['Value'].astype(float)

print("After conversion:")
print(df.dtypes)

# Filter only Yield element
df = df[df['Element'] == 'Yield']

print("After filter Yield:", df.shape)

# Keep only relevant columns
df = df[['Area', 'Item', 'Year', 'Value']]

print("After keep columns:", df.shape)

# Drop missing
df.dropna(inplace=True)

print("After dropna:", df.shape)

# Encode
df_encoded = pd.get_dummies(df, columns=['Area', 'Item'], drop_first=True)

print("Encoded shape:", df_encoded.shape)
print("Encoded dtypes:")
print(df_encoded.dtypes)

# Features and target
y = df_encoded['Value']
X = df_encoded.drop('Value', axis=1)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X dtypes:")
print(X.dtypes)
