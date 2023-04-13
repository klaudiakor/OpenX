from typing import Literal
import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from prepare_dataframe import *


def logistic_regression(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series,
                        penalty='l2',
                        max_iter=10000) -> np.float64:
    """
    Trains a logistic regression model on the provided training data.
    Paramethers
    ----
    penalty: {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
        Specify the norm of the penalty
    max_iter:
        Maximum number of iterations taken for the solvers to converge
        
    Returns the tuple with accuracy score and classification report.
    """

    model = LogisticRegression(max_iter=max_iter, penalty=penalty)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test,
                          y_pred), classification_report(y_test, y_pred)


def k_nearest_neighbors(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series,
                        n_neighbors=5,
                        weights='uniform') -> np.float64:
    """
    Trains a k-nearest neighbors classifier on the provided training data.
    Paramethers
    ---
    n_neighbors: int, default=5
    Number of neighbors to use by default for kneighbors queries.

    weights : {'uniform', 'distance'} , default='uniform'
        Weight function used in prediction. 
    
    Returns the tuple with accuracy score and classification report.
    """

    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test,
                          y_pred), classification_report(y_test, y_pred)


LOGISTIC_REGR_MODEL_NAME = "logistic"


class Logistic_regression_params(BaseModel):
    model_type: Literal['logistic'] = LOGISTIC_REGR_MODEL_NAME
    penalty = 'l2'
    max_iter = 10000


KNN_MODEL_NAME = "knn"


class K_nearest_neighbors_params(BaseModel):
    model_type: Literal['knn'] = KNN_MODEL_NAME
    n_neighbors = 5
    weights = 'uniform'


class Logistic_regression_runner(BaseModel):
    name = "Logistic Regression"
    example_params = Logistic_regression_params()
    description = """ Trains the logistic regression model | Params: | penalty: {'l1', 'l2', 'elasticnet', 'none'} - Specify the norm of the penalty
    max_iter: int - Maximum number of iterations taken for the solvers to converge"""

    def run(param: Logistic_regression_params):

        df = prepare_data_frame()
        X, y = split_df(df)

        X_train, X_test, y_train, y_test = preprocessing(X, y)
        return logistic_regression(X_train,
                                   X_test,
                                   y_train,
                                   y_test,
                                   penalty=param.penalty,
                                   max_iter=param.max_iter)


class K_nearest_neighbors_runner(BaseModel):
    name = "K-Nearest Neighbors"
    example_params = K_nearest_neighbors_params()
    description = "Trains a k-nearest neighbors classifier. | Params: |n_neighbors: int, default=5 Number of neighbors to use by default for kneighbors queries. weights : {'uniform', 'distance'} , default='uniform' Weight function used in prediction."  #TODO

    def run(self, param: K_nearest_neighbors_params):

        df = prepare_data_frame()
        X, y = split_df(df)

        X_train, X_test, y_train, y_test = preprocessing(X, y)
        return k_nearest_neighbors(X_train,
                                   X_test,
                                   y_train,
                                   y_test,
                                   n_neighbors=param.n_neighbors,
                                   weights=param.weights)  #TODO: params in one


if __name__ == "__main__":
    df = prepare_data_frame()
    X, y = split_df(df)
    X_train, X_test, y_train, y_test = preprocessing(X, y)

    log_reg_score = logistic_regression(X_train, X_test, y_train, y_test)
    knn_score = k_nearest_neighbors(X_train, X_test, y_train, y_test)

    print_results("Logistic Regression", log_reg_score)
    print_results("K-Nearest Neigbours", knn_score)
