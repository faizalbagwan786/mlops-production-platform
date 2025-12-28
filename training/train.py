import pandas as pd
import joblib
import os

from evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


print("Step 1: Loading data")

# Load raw data
data_path = "data/raw/churn.csv"
df = pd.read_csv(data_path)

print("Data loaded successfully")
print(df.head())


print("Step 2: Separate features and target")

X = df.drop("churn", axis=1)
y = df["churn"]


print("Step 3: Define columns")

categorical_cols = ["contract_type"]
numeric_cols = ["age", "monthly_charges", "tenure"]


print("Step 4: Preprocessing")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)


print("Step 5: Create ML pipeline")

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ]
)


print("Step 6: Train-test split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("Step 7: Train model")

model.fit(X_train, y_train)


print("Step 8: Evaluate model")

passed, accuracy = evaluate_model(model, X_test, y_test)

if not passed:
    raise ValueError("Model did not meet evaluation criteria. Aborting save.")


print("Step 9: Save model")

model_dir = "app/model"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "churn_model.pkl")
joblib.dump(model, model_path)

print(f"Model saved at {model_path}")
