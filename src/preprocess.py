import pandas as pd

def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    
    # Filter only Yield element
    df = df[df['Element'] == 'Yield']
    
    # Keep only relevant columns
    df = df[['Area', 'Item', 'Year', 'Value']]
    
    # Drop missing values
    df.dropna(inplace=True)
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['Area', 'Item'], drop_first=True)
    
    return df_encoded
