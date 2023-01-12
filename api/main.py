from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

origins = [
   "http://localhost",
   "http://localhost:3000",
]
app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

endpoint = "http://localhost:8601/v1/models/potatoes_model:predict"
endpoint_1 = "http://localhost:8601/v1/models/pepper_model:predict"
endpoint_2 = "http://localhost:8601/v1/models/tomato_model:predict"

CLASS_NAMES = ["Potato Early Blight", "Potato Late Blight", "Potato Healthy"]
PEPPER_CLASS_NAMES= ['Pepper Bacterial spot', 'Pepper Healthy']
Tomato_CLASS_NAMES=['Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Leaf Mold',
                    'Tomato Septoria Leaf Spot', 'Tomato Spider Mites Two Spotted Spider Mite',
                    'Tomato Target Spot', 'Tomato YellowLeaf Curl Virus', 'Tomato Mosaic Virus', 'Tomato Healthy']
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }
@app.post("/predict_pepper")
async def predict_pepper(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint_1, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = PEPPER_CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

@app.post("/predict_tomato")
async def predict_tomato(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint_2, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = Tomato_CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)