from fastapi import FastAPI, UploadFile, File, Form
from typing import Dict, Any
from pydantic import BaseModel
import base64
import sys;sys.path.append('../')
from listener_effort_api.lepm import predict_le

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World!"}

@app.post("/predict")
def predict(input_dict: Dict[str, Dict]):
    """
    Predict the label for the given input dictionary.
    """
    listener_effort = predict_le(input_dict)
    return {"listener_effort": listener_effort}

class AudioItem(BaseModel):
    wav_b64: str
    metadata: Dict[str, Any]
    
class PredictRequest(BaseModel):
    files: Dict[str, AudioItem]

@app.post("/predict_from_bytes")
async def predict_from_bytes(req: PredictRequest):
    input_dict = {}
    for filename, item in req.files.items():
        wav_bytes = base64.b64decode(item.wav_b64)
        input_dict[filename] = {
            "wav_path": filename,
            "wav_bytes": wav_bytes,
            "metadata": item.metadata
        }
    results = predict_le(input_dict, from_bytes=True)
    return results