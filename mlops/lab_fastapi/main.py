from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import logging
from typing import List, Optional
import uvicorn
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели (замените путь на свой)
try:
    with open("lr_salary.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None  

app = FastAPI(title="Salary")


class SalaryFeatures(BaseModel):
    ExperienceYears: float = Field(..., alias="Experience Years")


@app.post("/predict", summary="Predict salary")
async def predict(salary: SalaryFeatures):
    """
    Фугкция для предсказания зарплаты
    """
    if model is None:
        return {"error": "Model is not loaded"}

    try:
        input_data = pd.DataFrame([salary.dict()])
        
        # Переименовываем столбцы, если необходимо
        input_data.columns = ['Experience Years']  # Это имя должно совпадать с тем, что использовалось при обучении
        
        # Предсказание
        predict = model.predict(input_data)[0]
        return {"predicted_salary": round(float(predict), 2)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
