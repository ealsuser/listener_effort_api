from fastapi import FastAPI
import base64
import sys;sys.path.append('../')
from listener_effort_api.items import PredictRequest
from listener_effort_api.lepm import predict_le, batch_predict_le

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World!"}

@app.post("/v1/listener-effort")
async def predict_from_bytes(req: PredictRequest):
    # Decode the base64-encoded audio data in the request
    for session in req.input:
        for audio in session.audios:
            audio.wav = base64.b64decode(audio.wav)
    return batch_predict_le(req)