from fastapi import FastAPI, HTTPException
from app.schema import ChurnRequest
from app.model_loader import load_model

app = FastAPI(title="Churn Prediction API")

# Load model once at startup
try:
    model = load_model()
except Exception as e:
    model = None
    print(f"Model load failed: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: ChurnRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Prepare data for prediction
    import pandas as pd

    input_data = pd.DataFrame([{
    "age": data.age,
    "monthly_charges": data.monthly_charges,
    "tenure": data.tenure,
    "contract_type": data.contract_type
    }])

    
    try:
        prediction = int(model.predict(input_data)[0])
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
