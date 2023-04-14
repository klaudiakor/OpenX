import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from prepare_dataframe import get_columns_names, prepare_data_for_model_with_selected_features, print_results

LOGISTIC_REGR_MODEL_NAME = "logistic"
KNN_MODEL_NAME = "knn"
FEATURES_NAMES = get_columns_names()[:-1]


class Logistic_regression_params(BaseModel):
    penalty = 'l2'
    max_iter = 10000
    features_names = FEATURES_NAMES


class K_nearest_neighbors_params(BaseModel):
    n_neighbors = 5
    weights = 'uniform'
    features_names = FEATURES_NAMES


def logistic_regression(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
    y_test: pd.Series, params: Logistic_regression_params
) -> tuple[tuple[np.float64, str], np.ndarray]:
    """
    Trains a logistic regression model on the provided training data.
    
    params: Logistic_regression_params
        penalty: {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
            Specify the norm of the penalty
        max_iter:
            Maximum number of iterations taken for the solvers to converge
            
    Returns the tuple with accuracy score and classification report and vector of prediction.
    """

    model = LogisticRegression(max_iter=params.max_iter,
                               penalty=params.penalty)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    score = (accuracy_score(y_test,
                            y_pred), classification_report(y_test, y_pred))

    return (score, y_pred)


def k_nearest_neighbors(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
    y_test: pd.Series, params: K_nearest_neighbors_params
) -> tuple[tuple[np.float64, str], np.ndarray]:
    """
    Trains a k-nearest neighbors classifier on the provided training data.
    
    params: K_nearest_neighbors_params   
        n_neighbors: int, default=5
        Number of neighbors to use by default for kneighbors queries.

        weights : {'uniform', 'distance'} , default='uniform'
            Weight function used in prediction. 
        
    Returns the tuple with accuracy score and classification report and vector of prediction.
    """

    model = KNeighborsClassifier(n_neighbors=params.n_neighbors,
                                 weights=params.weights)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = (accuracy_score(y_test,
                            y_pred), classification_report(y_test, y_pred))

    return (score, y_pred)


class Logistic_regression_runner(BaseModel):
    name = "Logistic Regression"
    description = """ Trains the logistic regression model | Params: | penalty: {'l1', 'l2', 'elasticnet', 'none'} - Specify the norm of the penalty
    max_iter: int - Maximum number of iterations taken for the solvers to converge"""

    def run(self, param: Logistic_regression_params) -> dict:

        X_train, X_test, y_train, y_test = prepare_data_for_model_with_selected_features(
            param.features_names)
        score, y_pred = logistic_regression(X_train, X_test, y_train, y_test,
                                            param)

        return {"accuracy": score[0], "prediction": y_pred.tolist()}


class K_nearest_neighbors_runner(BaseModel):
    name = "K-Nearest Neighbors"
    description = "Trains a k-nearest neighbors classifier. | Params: |n_neighbors: int, default=5 Number of neighbors to use by default for kneighbors queries. weights : {'uniform', 'distance'} , default='uniform' Weight function used in prediction."

    def run(self, param: K_nearest_neighbors_params) -> dict:

        X_train, X_test, y_train, y_test = prepare_data_for_model_with_selected_features(
            param.features_names)

        score, y_pred = k_nearest_neighbors(X_train, X_test, y_train, y_test,
                                            param)

        return {"accuracy": score[0], "prediction": y_pred.tolist()}


if __name__ == "__main__":

    log_regr_params = Logistic_regression_params()
    X_train, X_test, y_train, y_test = prepare_data_for_model_with_selected_features(
        log_regr_params.features_names)

    log_reg_score = logistic_regression(X_train, X_test, y_train, y_test,
                                        log_regr_params)[0]
    knn_score = k_nearest_neighbors(X_train, X_test, y_train, y_test,
                                    K_nearest_neighbors_params())[0]

    print_results("Logistic Regression", log_reg_score)
    print_results("K-Nearest Neigbours", knn_score)
