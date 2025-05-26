import requests #type: ignore
import base64
import sys;sys.path.append('../')
from listener_effort_api.utils import get_logger
logger = get_logger()

def load_wav_file(file_path):
    with open(file_path, 'rb') as f:
        return f.read()

payload = {
    "files": {
        "file_1": {
            "wav_b64": base64.b64encode(load_wav_file('/data-output/radcliff_data/2024-05-03/raw/aural/03cd72ae-fbf6-484b-9d86-09f7230726bf/20230927-162950-A91472/14.wav')).decode(),
            "metadata": {"user_id": '03cd72ae-fbf6-484b-9d86-09f7230726bf'}
        },
        "file_2": {
            "wav_b64": base64.b64encode(load_wav_file('/data-output/radcliff_data/2024-05-03/raw/aural/03cd72ae-fbf6-484b-9d86-09f7230726bf/20230927-162950-A91472/15.wav')).decode(),
            "metadata": {"user_id": '03cd72ae-fbf6-484b-9d86-09f7230726bf'}
        }
    }
}

def test_predict():
    """
    Test the predict function.
    """
    try:
        response = requests.post('http://localhost:8000/predict_from_bytes', json=payload)
        logger.info(f"Response: {response.json()}")
        assert response.status_code == 200
        assert response.json()['prediction'] == 8.764874644468387
        logger.info("Test passed")
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    test_predict()