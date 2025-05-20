from fastapi import FastAPI, UploadFile, File, Form
from typing import Dict, List
import sys;sys.path.append('../')
from lepm_api.lepm import predict_le

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

@app.post("/predict_from_bytes")
async def predict_from_bytes(
    files: List[UploadFile] = File(...),
    metadata: List[str]     = Form(...)
):
    print("Received files:", len(files))
    input_dict = {}
    for i, (file_obj, meta) in enumerate(zip(files, metadata)):
        wav = await file_obj.read()
        input_dict[f'{i+1}_{file_obj.filename}'] = {
            "wav_path":  file_obj.filename,
            "wav_bytes": wav,
            "metadata":  meta
        }
    print("Input dictionary:", input_dict.keys())
    results = predict_le(input_dict, from_bytes=True)
    return results