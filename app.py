from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from pathlib import Path
import logging
import pandas as pd

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier
from src.pipline.training_pipeline import TrainPipeline

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

class DataForm:
    def __init__(self, request: Request):
        self.request = request
        self.Gender = None
        self.Age = None
        self.Driving_License = None
        self.Region_Code = None
        self.Previously_Insured = None
        self.Annual_Premium = None
        self.Policy_Sales_Channel = None
        self.Vintage = None
        self.Vehicle_Age_lt_1_Year = None
        self.Vehicle_Age_gt_2_Years = None
        self.Vehicle_Damage_Yes = None

    async def get_vehicle_data(self):
        form = await self.request.form()
        self.Gender = form.get("Gender")
        self.Age = form.get("Age")
        self.Driving_License = form.get("Driving_License")
        self.Region_Code = form.get("Region_Code")
        self.Previously_Insured = form.get("Previously_Insured")
        self.Annual_Premium = form.get("Annual_Premium")
        self.Policy_Sales_Channel = form.get("Policy_Sales_Channel")
        self.Vintage = form.get("Vintage")
        self.Vehicle_Age_lt_1_Year = form.get("Vehicle_Age_lt_1_Year")
        self.Vehicle_Age_gt_2_Years = form.get("Vehicle_Age_gt_2_Years")
        self.Vehicle_Damage_Yes = form.get("Vehicle_Damage_Yes")


@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="vehicledata.html",
        context={"context": "Rendering", "status": None, "error": None},
    )


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return {"status": "Training successful!!!"}
    except Exception as e:
        logging.exception("Training failed")
        return {"status": False, "error": str(e)}


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_vehicle_data()

        payload = {
            "Gender": int(form.Gender),
            "Age": int(form.Age),
            "Driving_License": int(form.Driving_License),
            "Region_Code": int(form.Region_Code),
            "Previously_Insured": int(form.Previously_Insured),
            "Annual_Premium": float(form.Annual_Premium),
            "Policy_Sales_Channel": float(form.Policy_Sales_Channel),
            "Vintage": int(form.Vintage),
            "Vehicle_Age_lt_1_Year": int(form.Vehicle_Age_lt_1_Year),
            "Vehicle_Age_gt_2_Years": int(form.Vehicle_Age_gt_2_Years),
            "Vehicle_Damage_Yes": int(form.Vehicle_Damage_Yes),
        }

        vehicle_df = pd.DataFrame([payload])

        expected_columns = [
            "Gender",
            "Age",
            "Driving_License",
            "Region_Code",
            "Previously_Insured",
            "Annual_Premium",
            "Policy_Sales_Channel",
            "Vintage",
            "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_gt_2_Years",
            "Vehicle_Damage_Yes",
        ]

        vehicle_df = vehicle_df[expected_columns]

        model_predictor = VehicleDataClassifier()
        value = model_predictor.predict(dataframe=vehicle_df)[0]
        prediction = int(value)
        status = "Response-Yes" if prediction == 1 else "Response-No"

        return templates.TemplateResponse(
            request=request,
            name="vehicledata.html",
            context={
                "context": "Rendering",
                "status": status,
                "error": None,
            },
        )

    except Exception as e:
        logging.exception("Prediction failed")
        return templates.TemplateResponse(
            request=request,
            name="vehicledata.html",
            context={
                "context": "Rendering",
                "status": None,
                "error": f"Prediction failed: {e}",
            },
            status_code=400,
        )


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)