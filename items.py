from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, conlist
import pandas as pd # type: ignore

class AudioItem(BaseModel):
    wav: bytes
    transcript: Optional[str] = None
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioItem":
        return cls(
            wav=data['wav'],
            transcript=data['transcript']
        )

class SessionItem(BaseModel):
    audios: List[AudioItem]
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionItem":
        audios = [AudioItem.from_dict(audio) for audio in data['audios']]
        return cls(audios=audios)

class PredictRequest(BaseModel):
    input: List[SessionItem]
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictRequest":
        sessions = [SessionItem.from_dict(sess) for sess in data["input"]]
        return cls(input=sessions)

class WhisperTranscript(BaseModel):
    whisper_result: Dict[str, Any]
    model_size: str
    language: Union[str, None]
    params: Dict[str, Any]

class WhisperFeatures(BaseModel):
    speaking_rate_large_v2: float
    articulation_rate: float
    whisper_confidence_base: float
    whisper_probs: float

    @classmethod
    def mean(cls, features_list: List["WhisperFeatures"]) -> "WhisperFeatures":
        df = pd.DataFrame([f.dict() for f in features_list])
        means = df.mean(axis=0).to_dict()
        return cls(**means)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.dict()])


class AudioResult(BaseModel):
    status: str
    listener_effort: float

class SessionResult(BaseModel):
    status: str
    listener_effort: float
    listener_effort_stddev: float
    audio_results: List[AudioResult]

class PredictResponse(BaseModel):
    status: str
    result: List[SessionResult]