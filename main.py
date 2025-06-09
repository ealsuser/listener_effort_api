from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import base64
import os
from dotenv import load_dotenv
import sys;sys.path.append('../')
from listener_effort_api.items import PredictRequest
from listener_effort_api.lepm import batch_predict_le
load_dotenv()

app = FastAPI()
bearer_scheme = HTTPBearer()

def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """
    Dependency that raises 401 unless the incoming Bearer token
    matches the one in $EALS_LE_API_KEY.
    """
    expected = os.getenv("EALS_LE_API_KEY")
    if credentials.scheme.lower() != "bearer" or credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/")
def read_root():
    return {"Hello": "World!"}

@app.post("/v1/listener-effort")
async def predict_from_bytes(
    req: PredictRequest,
    _: HTTPAuthorizationCredentials = Depends(verify_token),  # enforce auth
):
    # Decode the base64-encoded audio data in the request
    for session in req.input:
        for audio in session.audios:
            audio.wav = base64.b64decode(audio.wav)
    return batch_predict_le(req)