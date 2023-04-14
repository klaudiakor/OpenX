# OpenX
ML Engineer - internship task

Dataset: https://archive.ics.uci.edu/ml/datasets/Covertype

Files:
-----
prepare_dataframe.py: \
    It contains functions that prepare the data to be used in models.

heuristic.py: \
    Implementation of a simple heuristic that classifies the data.

baseline_models.py: \
    Implementation of 2 basic models from the Scikit-learn library

neural_network.py: \
    Neural network implemented based on TensorFlow library and function that find a good set of hyperparameters.

models_evaluation.py: \
    Evaluation of models from previous files.


API:
-----
To use api run commend:
    uvicorn main:app

And open:
    http://127.0.0.1:8000/docs

