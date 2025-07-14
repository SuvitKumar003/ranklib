from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
from topsisx.topsis import topsis
from topsisx.utils import validate_inputs

app = FastAPI(
    title="TOPSISX API",
    version="0.1.0",
    description="API for multi-criteria decision making using TOPSIS"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to TOPSISX API! Go to /docs to try the endpoints."}

@app.post("/topsis/")
async def run_topsis(
    file: UploadFile,
    weights: str = Form(...),
    impacts: str = Form(...)
):
    try:
        # Read uploaded file
        df = pd.read_csv(file.file)
        
        # Parse weights and impacts
        weights_list = [float(w.strip()) for w in weights.split(",")]
        impacts_list = [i.strip() for i in impacts.split(",")]

        # Validate inputs
        validate_inputs(df, weights_list, impacts_list)

        # Run TOPSIS
        result_df = topsis(df, weights_list, impacts_list)

        # Convert DataFrame to dict for JSON response
        response = result_df.to_dict(orient="records")

        return JSONResponse(content={"result": response})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
