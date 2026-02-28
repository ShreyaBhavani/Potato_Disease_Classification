from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# prod_model = tf.keras.models.load_model(r"E:\Potato_disease\saved_models\1")
# beta_model = tf.keras.models.load_model(r"E:\Potato_disease\saved_models\2")
MODEL = tf.keras.models.load_model(r"E:\Potato_disease\saved_models\1")

# endpoint = "http://localhost:8605/v1/models/potato_disease_model:predict"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

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
    predictions = MODEL.predict(img_batch)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return{
        'class': predicted_class,
        'confidence': float(confidence)
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