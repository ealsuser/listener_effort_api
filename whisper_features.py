import numpy as np
import pandas as pd # type: ignore
import jiwer
import re
from pydantic import BaseModel
from typing import List, Dict, Any, Union
from listener_effort_api.preprocess import clean_text
from listener_effort_api.utils import get_logger
logger = get_logger()

### WER
def get_WER(
        string1: str, 
        string2: str,
        ) -> float:
    """
    Calculates the Word Error Rate (WER) between two strings.
    """
    try:
        return jiwer.wer(clean_text(string1), clean_text(string2))
    except Exception as e:
        logger.info(f"Error in calculate_WER: {e}")
        return np.nan

### Speaking/Articulation Rate
def get_whisper_duration(
        transcript: Union[Dict[str, Any], float],
        first_word: int = 1, 
        last_word: int = -2
    ) -> float:
    """
    Calculates the duration of the audio segment based on the timestamps of the first and last words.
    """
    try:
        if isinstance(transcript, dict):
            time_start = transcript["timestamps"][first_word][1]
            time_end = transcript["timestamps"][last_word][-1]
            duration = time_end - time_start
            if duration > 0:
                return duration
            else:
                logger.info(f"Duration is negative: {duration}")
                return np.nan
        else:
            logger.info(f"Transcript is not a dictionary: {transcript}")
            return np.nan
    except Exception as e:
        logger.info(f"Error in get_whisper_duration: {e}")
        return np.nan

def get_n_words(
        text: str, 
        first_word: int = 1, 
        last_word:int = -2
    )-> Union[int, float]:
    """
    Counts the number of words in a given text.
    """
    try:
        text_split = text.split()
        return len(text_split[first_word : len(text_split) + last_word + 1])
    except Exception as e:
        logger.info(f"Error in length_words: {e}")
        return np.nan

def get_timestamps(
        transcript_raw: dict,
    ) -> Union[Dict[str, Any], float]:
    """
    Extracts words and their corresponding timestamps from the transcript.
    """
    try:
        words_and_ts = []
        for segment in transcript_raw["segments"]:
            for word in segment["words"]:
                words_and_ts.append([word["text"], word["start"], word["end"]])

        return {"text": transcript_raw["text"], "timestamps": words_and_ts}
    except Exception as e:
        logger.info(f"Error in get_timestamps: {e}")
        return np.nan

def count_syllables(
        word: str
        ) -> int:
    """
    Counts the number of syllables in a given word.
    """
    count = 0
    vowels = "aeiouy"
    word = word.lower().strip(".:;?!")
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le"):
        count += 1
    if count == 0:
        count += 1
    return count

def count_syllables_in_text(
        text: str,
    )-> Union[int, float]:
    """
    Counts the total number of syllables in a given text.
    """
    try:
        words = text.split()
        total_syllables = sum(list(map(count_syllables, words)))
        return total_syllables
    except Exception as e:
        logger.info(f"Error in count_syllables_in_text: {e}")
        return np.nan

### Whisper probs and confidences
def get_whisper_probs(
        segments: List[Dict[str, Any]]
        ) -> Union[np.ndarray, float]:
    """
    Extracts the average log probabilities from the segments of the Whisper result.
    """
    try:
        avg_logprobs = np.array([segment["avg_logprob"] for segment in segments])
        probs = np.exp(avg_logprobs)
        return probs
    except Exception as e:
        logger.info(f"Error in get_whisper_probs: {e}")
        return np.nan

def get_whisper_confidences(
        segments: List[Dict[str, Any]]
        ) -> Union[np.ndarray, float]:
    """
    Extracts and averages the confidence scores from the words of the Whisper result.
    """
    try:
        all_confidences = []
        for segment in segments:
            for word in segment["words"]:
                all_confidences.append(word["confidence"])
        return np.array(all_confidences)
    except Exception as e:
        logger.info(f"Error in get_whisper_confidences: {e}")
        return np.nan

### Get all features

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

def get_whisper_features(
        whisper_result_large: Dict[str, Any],
        whisper_result_base: Dict[str, Any]
    ) -> WhisperFeatures:

    timestamps = get_timestamps(whisper_result_large)
    duration = get_whisper_duration(timestamps)

    # Speaking rate
    n_words = get_n_words(whisper_result_large['text'])
    speaking_rate = n_words / duration
    
    # Articulation rate
    n_syllables = count_syllables_in_text(whisper_result_large['text'])
    articulation_rate = n_syllables / duration

    # Whisper confidence
    whisper_confidences = get_whisper_confidences(whisper_result_base['segments'])
    mean_confidence = float(np.mean(whisper_confidences))

    # Whisper probs
    whisper_probs = get_whisper_probs(whisper_result_large['segments'])
    mean_probs = float(np.mean(whisper_probs))

    return WhisperFeatures(
        speaking_rate_large_v2=speaking_rate,
        articulation_rate=articulation_rate,
        whisper_confidence_base=mean_confidence,
        whisper_probs=mean_probs
    )