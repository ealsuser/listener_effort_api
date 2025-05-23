# LEPM API

### Setup
1. `poetry install`

### Run app
1. `poetry run uvicorn main:app --reload`
2. Open port: `poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000`

### Test app
1. Run app
2. `poetry run python tests/test.py`

### Streamlit
1. Run app
2. `poetry shell`
2. `streamlit run app.py`

### For commits
Check if everything compiles with:
1. `cd listener_effort_api`
2. `poetry run mypy .`