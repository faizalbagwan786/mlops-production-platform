import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "churn_model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model
