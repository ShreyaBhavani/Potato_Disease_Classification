from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
import tensorflow as tf
import keras

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load SavedModel exported at E:\\Potato_disease\\saved_models\\potato_disease_model
MODEL_PATH = Path(r"E:\Potato_disease\saved_models\potato_disease_model\1")
MODEL = keras.layers.TFSMLayer(str(MODEL_PATH), call_endpoint="serve")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
async def root():
    return {"message": "Potato disease API is running"}

@app.get("/ping")
async def ping():
    return "Hello,I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))  # SAME AS TRAINING
    # The training pipeline already contained Resizing + Rescaling layers,
    # so the exported model expects raw pixel ranges (0-255 float32).
    image = np.array(image).astype("float32")
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)    
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    print("Image shape:", img_batch.shape)
    print("Max pixel value:", np.max(img_batch))
    # Run inference directly against SavedModel via TFSMLayer
    try:
        preds = MODEL(img_batch, training=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running model: {e}")

    # TFSMLayer can return a dict or tensor; normalize to numpy array
    if isinstance(preds, dict):
        preds = list(preds.values())[0]

    predictions = np.array(preds)

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