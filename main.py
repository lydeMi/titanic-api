from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Charger le modèle et les colonnes attendues
model = joblib.load("titanic_model.pkl")
expected_cols = joblib.load("model_columns.pkl")

# Définir les données attendues
class Passenger(BaseModel):
    Pclass: int
    Age: float
    Fare: float
    Sex_male: int
    Embarked_S: int
    FamilySize: int

# Créer l'application
app = FastAPI(title="Titanic Survival API")

@app.get("/")
def welcome():
    return {"message": "Bienvenue sur l'API Titanic"}

@app.post("/predict")
def predict_survival(data: Passenger):
    try:
        input_data = pd.DataFrame([data.model_dump()])
        input_data = input_data[expected_cols]  # Forcer l'ordre exact

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][prediction]

        return {
            "prediction": int(prediction),
            "survival": "Survivant" if prediction == 1 else "Non survivant",
            "probability": round(probability, 4)
        }

    except Exception as e:
        return {"error": str(e)}
