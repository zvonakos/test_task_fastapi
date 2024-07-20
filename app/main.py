import io

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from keras.src.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from keras.src.utils import load_img, img_to_array

import numpy as np
import uvicorn
import logging

from utils import allowed_file
from schemas import PredictionResponse

app = FastAPI()

model = MobileNetV2(weights='mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/proceed", response_model=PredictionResponse)
async def proceed(file: UploadFile = File(), k: int = 5):
    # if not allowed_file(file.filename):
    #     logger.error(f"Unsupported file type: {file.filename}")
    #     raise HTTPException(status_code=400, detail="Unsupported file type")
    # try:
    contents = await file.read()
    # img = load_img(file, target_size=(224, 224))
    image = Image.open(io.BytesIO(contents))
    img = image.resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=k)[0]

    response = [
        {"class": pred[1], "probability": float(pred[2])} for pred in decoded_preds
    ]

    logger.info(f"Predictions: {response}")

    return JSONResponse(content={"predictions": response})

    # except Exception as e:
    #     logger.error(f"Error during prediction: {e}")
    #     raise HTTPException(status_code=500, detail="Error during prediction")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
