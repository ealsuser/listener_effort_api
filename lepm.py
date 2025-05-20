from typing import Dict, Any, Optional, List
import pandas as pd # type: ignore
import json
from joblib import load # type: ignore

from lepm_api.whisper_transcripts import get_transcript, get_transcript_from_bytes
from lepm_api.whisper_features import get_whisper_features, WhisperFeatures
from lepm_api.whisper_features import get_WER
from lepm_api import config
from lepm_api.utils import get_logger
logger = get_logger()

def get_features_for_model(
        input_dict: Dict[str, Dict],
        whisper_large: Dict[str, Any],
        whisper_base: Dict[str, Any],
    ) -> pd.DataFrame:

    # Get all whisper features
    all_whisper_features = []
    for audio_i, audio_i_data in input_dict.items():

        # Get this whisper transcript
        wr_large_i = whisper_large[audio_i_data['wav_path']]
        wr_base_i = whisper_base[audio_i_data['wav_path']]

        # Get whisper features
        # wer = get_WER(audio_i_data['task_prompt'], wr_large_i['whisper_result']["text"])
        # if wer > .8:
        #     logger.info(f"WER too high: {wer} for {audio_i_data['wav_path']}")
        #     continue
        whisper_features = get_whisper_features(wr_large_i['whisper_result'], wr_base_i['whisper_result'])
        all_whisper_features.append(whisper_features)
        logger.info(f"Whisper features for {audio_i_data['wav_path']}: {whisper_features}")
    
    if len(all_whisper_features) == 0:
        logger.info("No whisper features to predict")
        return []
    return WhisperFeatures.mean(all_whisper_features)

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
        input_dict: Dict[str, Dict],
        from_bytes: Optional[bool] = False,
    ) -> Dict[str, Any]:

    # Get all Whisper transcripts
    if from_bytes:
        wavs_to_transcribe = input_dict.copy()
        whisper_large = get_transcript_from_bytes(wavs_to_transcribe, model_size='large-v2')
        whisper_base = get_transcript_from_bytes(wavs_to_transcribe, model_size='base')
    else:
        wavs_to_transcribe = [task_data['wav_path'] for task_data in input_dict.values()]
        whisper_large = get_transcript(wavs_to_transcribe, model_size='large-v2')
        whisper_base = get_transcript(wavs_to_transcribe, model_size='base')

    # Get features for model
    feaures_for_model = get_features_for_model(input_dict, whisper_large, whisper_base)

    # Get model predictions
    model, model_metadata = load_model()
    features_train = model_metadata['features_train']
    prediction = model.predict(feaures_for_model.to_dataframe()[features_train])[0]
    
    return {'prediction': prediction, 'features': feaures_for_model.dict(), 'transcripts': whisper_large}