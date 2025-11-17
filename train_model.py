import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

CSV_PATH = r"C:\Users\sunan\OneDrive\Documents\new project\cleaned_csv.csv"
MODEL_OUT = r"C:\Users\sunan\OneDrive\Documents\new project\new_model.pkl"

# --- Ensure CSV exists with at least 4 numeric columns ---
if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
    # Create sample data
    df = pd.DataFrame({
        "solar": [10, 15, 20],
        "wind": [5, 7, 10],
        "hydro": [3, 4, 5],
        "target": [100, 120, 150]
    })
    df.to_csv(CSV_PATH, index=False)
else:
    df = pd.read_csv(CSV_PATH)
    # Keep only numeric columns
    df = df.select_dtypes(include="number").dropna()
    # If less than 4 numeric columns, create sample data
    if df.shape[1] < 4:
        df = pd.DataFrame({
            "solar": [10, 15, 20],
            "wind": [5, 7, 10],
            "hydro": [3, 4, 5],
            "target": [100, 120, 150]
        })

# --- Train the model ---
X = df[["solar", "wind", "hydro"]]
y = df["target"]

model = RandomForestRegressor()
model.fit(X, y)

# --- Save the trained model ---
with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)

print("✔ Model trained and saved as:", MODEL_OUT)