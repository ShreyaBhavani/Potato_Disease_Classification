from fastapi import FastAPI,File,UploadFile,HTTPException
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8605/v1/models/potato_disease_model:predict"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
async def root():
    return {"message": "Potato disease API is running"}

@app.get("/ping")
async def ping():
    return "Hello,I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)    
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    json_data ={
        "instances": img_batch.tolist()
    }

    try:
        response = requests.post(endpoint, json=json_data, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # If TF Serving is not reachable or returns error, show a clear API error
        raise HTTPException(status_code=502, detail=f"Error calling TF Serving: {e}")

    result = response.json()

    if "predictions" not in result:
        raise HTTPException(status_code=502, detail="TF Serving response missing 'predictions' field")

    predictions = np.array(result["predictions"])

    predicted_class = CLASS_NAMES[int(np.argmax(predictions[0]))]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": confidence,
    }

    
    # print("Filename:", file.filename)
    # print("Content-Type:", file.content_type)

    # data = await file.read()
    # print("File size (bytes):", len(data))

    # return {
    #     "filename": file.filename,
    #     "content_type": file.content_type,
    #     "size": len(data)
    # }



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)