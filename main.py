from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import uuid
import os
from rembg import remove

from model_utils import predict_species, predict_gender, remove_background_and_save

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Step 1: Save uploaded file temporarily
        temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
        temp_path = os.path.join("temp", temp_filename)
        os.makedirs("temp", exist_ok=True)

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 2: Create path for cleaned image and process
        cleaned_filename = f"{uuid.uuid4().hex}_cleaned.jpg"
        cleaned_path = os.path.join(UPLOAD_DIR, cleaned_filename)
        remove_background_and_save(temp_path, cleaned_path)

        # Step 3: Predict using the cleaned image only
        species = predict_species(cleaned_path)
        gender = predict_gender(cleaned_path)

        # Step 4: Delete original uploaded temp file
        os.remove(temp_path)

        return {"species": species, "gender": gender}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
