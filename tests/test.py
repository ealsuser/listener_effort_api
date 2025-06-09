import requests #type: ignore
import base64
import os
from dotenv import load_dotenv
import sys;sys.path.append('../')
from listener_effort_api.utils import get_logger
logger = get_logger()

# load .env into os.environ
load_dotenv()

def load_wav_file(file_path):
    with open(file_path, 'rb') as f:
        return f.read()

payload = {
    "input": [
        {
            "audios": [
                {
                   "wav": base64.b64encode(load_wav_file('/data-output/radcliff_data/2024-05-03/raw/aural/03cd72ae-fbf6-484b-9d86-09f7230726bf/20230927-162950-A91472/19.wav')).decode(),
                    "transcript": "The forest near my grandpa's cabin is said to contain mythical creatures."
                },
                {
                    "wav": base64.b64encode(load_wav_file('/data-output/radcliff_data/2024-05-03/raw/aural/03cd72ae-fbf6-484b-9d86-09f7230726bf/20230927-162950-A91472/20.wav')).decode(),
                    "transcript": None
                },
                {
                    "wav": base64.b64encode(load_wav_file('/data-output/radcliff_data/2024-05-03/raw/aural/03cd72ae-fbf6-484b-9d86-09f7230726bf/20230927-162950-A91472/21.wav')).decode(),
                    "transcript": None
                }
            ]
        },
        {
            "audios": [
                {
                   "wav": base64.b64encode(load_wav_file('/data-output/radcliff_data/2024-05-03/raw/aural/8430613e-9cfe-464c-b657-f24ea1e3b5f1/20230922-222930-A73246/19.wav')).decode(),
                    "transcript": 'Why did the critics rate this book so favorably?'
                },
                {
                    "wav": base64.b64encode(load_wav_file('/data-output/radcliff_data/2024-05-03/raw/aural/8430613e-9cfe-464c-b657-f24ea1e3b5f1/20230922-222930-A73246/20.wav')).decode(),
                    "transcript": 'Many of his songs are the products of collaborations with other musicians.'
                },
            ]
        }
    ]
}


def test_predict():
    """
    Test the predict function.
    """
    try:
        token = os.getenv("EALS_LE_API_KEY")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        response = requests.post('http://localhost:8000/v1/listener-effort', json=payload, headers=headers)
        logger.info(f"Response: {response.json()}")
        assert response.status_code == 200
        # assert response.json()['status'] == "ok"
        # assert round(response.json()['result'][0]['listener_effort'], 3) == 0
        # assert round(response.json()['result'][0]['listener_effort_stddev'], 3) == 5.326
        # assert round(response.json()['result'][1]['listener_effort'], 3) == 80.617
        # assert round(response.json()['result'][1]['listener_effort_stddev'], 3) == 13.652
        logger.info("Test passed")
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    test_predict()