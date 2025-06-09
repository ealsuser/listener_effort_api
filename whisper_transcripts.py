from typing import Dict, Any, Optional, List
from listener_effort_api.items import SessionItem, WhisperTranscript
import inspect
import tempfile
import whisper_timestamped # type: ignore
from listener_effort_api.utils import get_logger
logger = get_logger()

def get_transcript_from_bytes(
    session: SessionItem,
    model_size: str = "large-v2",
    device: str = "cpu",
    initial_prompt: Optional[str] = None,
    fp16: bool = False,
    language: Optional[str] = "en"
) -> List[WhisperTranscript]:
    
    """
    Get whisper transcription.
    """
    session = session.copy()
    model = whisper_timestamped.load_model(model_size, device=device)

    all_results = []
    for i, audio in enumerate(session.audios):

        # Do transcript
        logger.info(f'Transcribing audio {i+1} of {len(session.audios)}')
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
            tmpfile.write(audio.wav)
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
        
        # Create WhisperTranscript object
        whisper_result = WhisperTranscript(
            whisper_result=result,
            model_size=model_size,
            language=language,
            params=default_args
        )

        all_results.append(whisper_result)
        
    return all_results


# def get_transcript(
#     wav_paths: List[str],
#     model_size: Optional[str] = "large-v2",
#     device: Optional[str] = "cpu",
#     initial_prompt: Optional[str] = None,
#     fp16: Optional[bool] = False,
#     language: Optional[str] = "en"
# ) -> Dict[str, Any]:

#     """
#     Get whisper transcription.
#     """
#     model = whisper_timestamped.load_model(model_size, device=device)

#     all_results = {}
#     for wav_path in wav_paths:

#         # Do transcript
#         logger.info(f'Transcribing -> {wav_path}')
#         audio = whisper_timestamped.load_audio(wav_path)
#         result = whisper_timestamped.transcribe(
#             model, audio, language=language, initial_prompt=initial_prompt, fp16=fp16,
#         )

#         # Collect args and results
#         signature = inspect.signature(whisper_timestamped.transcribe)
#         default_args = {
#             param.name: param.default
#             for param in signature.parameters.values()
#             if param.default is not inspect.Parameter.empty
#         }

#         all_results[wav_path] = {
#                 "whisper_result": result,
#                 "model_size": model_size,
#                 "language": language,
#                 "params": default_args,
#             }
        
#     return all_results