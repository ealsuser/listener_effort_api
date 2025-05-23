import requests #type: ignore
import sys;sys.path.append('../')
from listener_effort_api.utils import get_logger
logger = get_logger()

input_dict = {
    'audio_1': {
        'wav_path': '/data-output/radcliff_data/2024-05-03/raw/aural/03cd72ae-fbf6-484b-9d86-09f7230726bf/20230927-162950-A91472/14.wav', 
        # 'task_prompt': 'The supermarket chain shut down because of poor management.'
        },
    'audio_2': {
        'wav_path': '/data-output/radcliff_data/2024-05-03/raw/aural/03cd72ae-fbf6-484b-9d86-09f7230726bf/20230927-162950-A91472/15.wav',
        # 'task_prompt': 'Much more money must be donated to make this department succeed.'
        },
}

def test_predict():
    """
    Test the predict function.
    """
    try:
        response = requests.post('http://localhost:8000/predict', json=input_dict)
        logger.info(f"Response: {response.json()}")
        assert response.status_code == 200
        assert response.json() == {"listener_effort": 8.764874644468387}
        logger.info("Test passed")
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    test_predict()