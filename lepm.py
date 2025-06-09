from typing import List
import pandas as pd # type: ignore
import numpy as np # type: ignore
import json
import random
from joblib import load # type: ignore
from listener_effort_api.items import PredictRequest, SessionItem, WhisperTranscript, WhisperFeatures, PredictResponse, SessionResult, AudioResult
from listener_effort_api.whisper_transcripts import get_transcript_from_bytes
from listener_effort_api.whisper_features import get_whisper_features, get_WER
from listener_effort_api import config
from listener_effort_api.utils import get_logger
logger = get_logger()

def get_features_for_model(
        session: SessionItem,
        whisper_large: List[WhisperTranscript],
        whisper_base: List[WhisperTranscript],
    ) -> List[WhisperFeatures]:

    # Get all whisper features
    all_whisper_features = []
    for audio_i, audio in enumerate(session.audios):

        # Get this whisper transcript
        w_large_i = whisper_large[audio_i]
        w_base_i = whisper_base[audio_i]

        # Get whisper features
        # wer = get_WER(audio.transcript, w_large_i.whisper_result["text"])
        # if wer > .8:
        #     logger.info(f"WER too high: {wer} for audio number {audio_i+1}")
        #     continue
        whisper_features = get_whisper_features(w_large_i.whisper_result, w_base_i.whisper_result)
        all_whisper_features.append(whisper_features)
        logger.info(f"Whisper features for audio {audio_i+1} out of {len(session.audios)}: {whisper_features}")
    
    if len(all_whisper_features) == 0:
        logger.info("No whisper features to predict")
        return []
    return all_whisper_features

def load_model() -> tuple:
    training_study = 'Speech_study' # 'Radcliff' # 'Speech_study' # 'Prilenia'
    model_name = 'LinearRegression05'

    # Load model's metadata
    model_metadata_save_name = f'{training_study}_{model_name}_model_metadata'
    with open(f'{config.Models.models_path}/{model_metadata_save_name}.json', 'r') as f:
        model_metadata = json.load(f)

    # Load model
    model = load(f'{config.Models.models_path}/{training_study}_{model_name}_model.joblib')
    
    return model, model_metadata

def predict_le(
        session: SessionItem,
    ) -> SessionResult:

    # Get all Whisper transcripts
    whisper_large = get_transcript_from_bytes(session, model_size='large-v2')
    whisper_base = get_transcript_from_bytes(session, model_size='base')

    # Get features for model
    features_for_model = get_features_for_model(session, whisper_large, whisper_base)

    # Get model predictions
    model, model_metadata = load_model()
    features_train = model_metadata['features_train']

    # Audio predictions
    audio_results = []
    for audio_features in features_for_model:
        try:
            audio_prediction = model.predict(audio_features.to_dataframe()[features_train])[0]
            audio_prediction = np.clip(audio_prediction, 0, 100)
            audio_status = "ok"
        except Exception as e:
            logger.error(f"Error in predicting audio: {e}")
            audio_prediction = np.nan
            audio_status = str(e)
        audio_result = AudioResult(
            status=audio_status,
            listener_effort=audio_prediction, 
        )
        audio_results.append(audio_result)

    # Session prediction
    try:
        session_features = WhisperFeatures.mean(features_for_model).to_dataframe()
        session_prediction = model.predict(session_features[features_train])[0]
        session_prediction = np.clip(session_prediction, 0, 100)
        session_status = "ok"
    except Exception as e:
        logger.error(f"Error in predicting session: {e}")
        session_prediction = np.nan
        session_status = str(e)
    if session_status == "ok":
        session_status = "ok" if all(a.status == "ok" for a in audio_results) else "error"
    listener_effort_stddev = float(np.nanstd([audio.listener_effort for audio in audio_results]))
    
    return SessionResult(
                status=session_status,
                listener_effort=session_prediction, 
                listener_effort_stddev=listener_effort_stddev,
                audio_results=audio_results,
            )

def predict_le_mock(
        session: SessionItem,
    ) -> SessionResult:

    # Audio predictions
    audio_results = []
    for _ in range(len(session.audios)):
        if random.uniform(1, 100)>5:
            audio_prediction = random.uniform(0, 100)
            audio_status = "ok"
        else:
            audio_prediction = -1
            audio_status = "mock error occurred"
        audio_result = AudioResult(
            status=audio_status,
            listener_effort=audio_prediction, 
        )
        audio_results.append(audio_result)

    # Session prediction
    if random.uniform(1, 100)>10:
        session_prediction = random.uniform(0, 100)
        listener_effort_stddev = random.uniform(0, 100)
        session_status = "ok"
    else:
        session_prediction = -1
        listener_effort_stddev = -1
        session_status = "mock error occurred"
    if session_status == "ok":
        session_status = "ok" if all(a.status == "ok" for a in audio_results) else "mock error occurred"

    return SessionResult(
                status=session_status,
                listener_effort=session_prediction, 
                listener_effort_stddev=listener_effort_stddev,
                audio_results=audio_results,
                )

def batch_predict_le(
        req: PredictRequest,
    ) -> PredictResponse:

    """
    Batch predict listener effort for multiple sessions.
    """
    results = []
    for session in req.input:
        assert len(session.audios)<=3, f"Only up to 3 audios per session are supported, received {len(session.audios)}"
        # result = predict_le(session)
        result = predict_le_mock(session)
        results.append(result)
    status = "ok" if all(res.status == "ok" for res in results) else "error"
    return PredictResponse(status=status, result=results)