from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse
from starter.model.model import load_model, predict_single
import logging
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

app = FastAPI()


class Data(BaseModel):
    workclass: str = None
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str
    age: int
    fnlwgt: int
    education_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int

    class Config:
        schema_extra = {
            "example": {
                "workclass": "state_gov",
                "education": "bachelors",
                "marital_status": "never_married",
                "occupation": "adm_clerical",
                "relationship": "not_in_family",
                "race": "white",
                "sex": "male",
                "native_country": "united_states",
                "age": 39,
                "fnlwgt": 77516,
                "education_num": 13,
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40
            }
        }


@app.on_event("startup")
async def startup_event():
    logging.info("Loading model")
    global model, encoder, lb
    model, encoder, lb = load_model('./starter/model')
    logging.info("Model loaded")


# welcome message on the root
@app.get("/")
def read_root():
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Welcome to the Adult Income Prediction API"}
    )


@app.post("/predict")
def predict(data: Data):
    if 'string' in data.dict().values():
        return {
            "error": "Please enter all the data correctly"
        }

    y_pred = predict_single(data, './starter/model')
    pred = list(y_pred)[0]

    # konsistentes Format sicherstellen
    if pred in [1, ">50K", ">50k"]:
        prediction = ">50K"
    else:
        prediction = "<=50K"

    return {
        "prediction": prediction
    }
