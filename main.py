from fastapi import FastAPI

from prepare_dataframe import *
from heuristic import *
from baseline_models import *
from neural_network import *

app = FastAPI()

models = {
    HEURISTIC_MODEL_NAME: Heuristic_model_runner(),
    LOGISTIC_REGR_MODEL_NAME: Logistic_regression_runner(),
    KNN_MODEL_NAME: K_nearest_neighbors_runner(),
    NEURAL_NETWORK_MODEL_NAME: Neural_network_runner()
}


@app.get("/model")
def get_all_models_info():
    """
    Returns name, example paramethers and model type about all models.
    """
    return models


@app.get("/model/{model_id}")
def get_specific_model_info(model_id: str):
    """"Returns name of selected model, example paramethers and description."""
    return models[model_id]


@app.post(f"/model/{HEURISTIC_MODEL_NAME}")
def run_heuristic_model():
    """Returns accuracy and classification report of heuristic model."""
    return models[HEURISTIC_MODEL_NAME].run()


@app.post(f"/model/{LOGISTIC_REGR_MODEL_NAME}")
def run_log_regr_model(params: Logistic_regression_params):
    """Returns accuracy and classification report of logistic regression model trained on specified paramethers."""
    return models[LOGISTIC_REGR_MODEL_NAME].run(params)


@app.post(f"/model/{KNN_MODEL_NAME}")
def run_log_regr_model(params: K_nearest_neighbors_params):
    """Returns accuracy and classification report of K-Nearest neighbors model trained on specified paramethers."""
    return models[KNN_MODEL_NAME].run(params)


@app.post(f"/model/{NEURAL_NETWORK_MODEL_NAME}")
def run_log_regr_model(params: Neural_network_params):
    """Returns accuracy and classification report of neural network model trained on specified paramethers."""
    return models[NEURAL_NETWORK_MODEL_NAME].run(params)
