import pandas as pd
from preprocess import load_and_clean
from train_model import rf, scaler, X  # Assuming Random Forest trained in train_model.py

def predict_yield(area, crop, year):
    # Create a dataframe with the same columns as X
    input_df = pd.DataFrame(columns=X.columns)
    
    # Set all zeros
    input_df.loc[0] = 0
    
    # Set year
    input_df.loc[0, 'Year'] = year
    
    # Set one-hot encoding for area
    area_col = [col for col in X.columns if col.startswith('Area_') and area in col]
    if area_col:
        input_df.loc[0, area_col[0]] = 1
    
    # Set one-hot encoding for crop
    crop_col = [col for col in X.columns if col.startswith('Item_') and crop in col]
    if crop_col:
        input_df.loc[0, crop_col[0]] = 1
    
    # Predict
    pred = rf.predict(input_df)
    return pred[0]

# Example
predicted_yield = predict_yield('Pakistan', 'Maize', 2025)
print(f"Predicted yield: {predicted_yield:.2f} kg/ha")
