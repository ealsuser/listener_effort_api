# LEPM API

### Setup
1. `poetry install`

### Run app
1. `poetry run uvicorn main:app --reload`

### Test app
1. Run app
2. `poetry run python tests/test.py`

### Streamlit
1. Run app
2. `poetry shell`
2. `streamlit run app.py`

### Jupyter Notebooks
Create a conda env with poetry-specified dependencies.
To update conda env in case of new dependencies, run steps 1, 3, and 4.
1. `poetry export -f requirements.txt --without-hashes -o requirements.txt`
2. `conda create -n listener-effort-api-env python=3.10`
3. `conda activate listener-effort-api-env`
4. `pip install -r requirements.txt`
5. `pip install ipykernel`
6. `python -m ipykernel install --user --name=listener-effort-api-env`

### For commits
Check if everything compiles with:
1. `cd listener_effort_api`
2. `poetry run mypy .`