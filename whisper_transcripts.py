from typing import Dict, Any, Optional, List
import inspect
import tempfile
import whisper_timestamped # type: ignore
from listener_effort_api.utils import get_logger
logger = get_logger()

def get_transcript(
    wav_paths: List[str],
    model_size: Optional[str] = "large-v2",
    device: Optional[str] = "cpu",
    initial_prompt: Optional[str] = None,
    fp16: Optional[bool] = False,
    language: Optional[str] = "en"
) -> Dict[str, Any]:

    """
    Get whisper transcription.
    """
    model = whisper_timestamped.load_model(model_size, device=device)

    all_results = {}
    for wav_path in wav_paths:

        # Do transcript
        logger.info(f'Transcribing -> {wav_path}')
        audio = whisper_timestamped.load_audio(wav_path)
        result = whisper_timestamped.transcribe(
            model, audio, language=language, initial_prompt=initial_prompt, fp16=fp16,
        )

        # Collect args and results
        signature = inspect.signature(whisper_timestamped.transcribe)
        default_args = {
            param.name: param.default
            for param in signature.parameters.values()
            if param.default is not inspect.Parameter.empty
        }

        all_results[wav_path] = {
                "whisper_result": result,
                "model_size": model_size,
                "language": language,
                "params": default_args,
            }
        
    return all_results

def get_transcript_from_bytes(
    wavs: Dict[str, Dict[str, bytes]],
    model_size: Optional[str] = "large-v2",
    device: Optional[str] = "cpu",
    initial_prompt: Optional[str] = None,
    fp16: Optional[bool] = False,
    language: Optional[str] = "en"
) -> Dict[str, Any]:

    """
    Get whisper transcription.
    """
    wavs = wavs.copy()
    model = whisper_timestamped.load_model(model_size, device=device)

    all_results = {}
    for wav_name, task_data in wavs.items():
        wav_bytes = task_data['wav_bytes']

        # Do transcript
        logger.info(f'Transcribing -> {wav_name}')
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
            tmpfile.write(wav_bytes)
            tmpfile.flush()
            audio = whisper_timestamped.load_audio(tmpfile.name)
            result = whisper_timestamped.transcribe(
                model, audio, language=language, initial_prompt=initial_prompt, fp16=fp16,
            )

        # Collect args and results
        signature = inspect.signature(whisper_timestamped.transcribe)
        default_args = {
            param.name: param.default
            for param in signature.parameters.values()
            if param.default is not inspect.Parameter.empty
        }

        all_results[wav_name] = {
                "whisper_result": result,
                "model_size": model_size,
                "language": language,
                "params": default_args,
            }
        
    return all_results