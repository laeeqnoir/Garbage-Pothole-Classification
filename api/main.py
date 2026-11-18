from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
import os

# Load model and labels once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/civicfix_mobilenetv2.h5")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "../src/civic_classifier/labels.json")

# Helper functions (mirroring src/civic_classifier/model_utils.py)
def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Compatibility shim for older models that used a 'groups' argument on DepthwiseConv2D
    class DepthwiseConv2DCompat(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, groups=1, **kwargs):
            # 'groups' argument is ignored – newer TF/Keras does not support it
            super().__init__(*args, **kwargs)
    # Some older checkpoints also reference the generic Conv2D with a groups kwarg.
    class Conv2DCompat(tf.keras.layers.Conv2D):
        def __init__(self, *args, groups=1, **kwargs):
            # Drop the groups param for compatibility
            super().__init__(*args, **kwargs)
    custom_objects = {
        "DepthwiseConv2D": DepthwiseConv2DCompat,
        "Conv2D": Conv2DCompat,
    }
    # Load without compiling (not needed for inference) to avoid potential optimizer issues.
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

def load_labels(label_path=LABELS_PATH):
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Labels file not found: {label_path}")
    with open(label_path, "r") as f:
        return json.load(f)

def preprocess_image(file_path, target_size=(224, 224)):
    img = image.load_img(file_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

model = load_model()
labels = load_labels()

app = FastAPI()
# Allow requests from the HTML page (served from file or any origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as out_file:
        content = await file.read()
        out_file.write(content)
    # Preprocess and run prediction
    img_arr = preprocess_image(temp_path)
    preds = model.predict(img_arr)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_label = labels.get(str(pred_idx), "unknown")
    confidence = float(preds[0][pred_idx]) * 100.0
    # Clean up temp file
    try:
        os.remove(temp_path)
    except Exception:
        pass
    return JSONResponse(content={"label": pred_label, "confidence": round(confidence, 2)})
