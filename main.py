from fastapi import Body, FastAPI
from typing import Union

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
    return models


@app.get("/model/{model_id}")
def get_specific_model_info(model_id: str):
    return models[model_id]


# https://stackoverflow.com/questions/71539448/using-different-pydantic-models-depending-on-the-value-of-fields
@app.post("/run_model")
def run_model(params: Union[Heuristic_model_params, Logistic_regression_params,
                            K_nearest_neighbors_params,
                            Neural_network_params] = Body(
                                ..., discriminator='model_type')):
    model = models[params.model_type]

    return model.run(params)