import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from prepare_dataframe import *

LOGISTIC_REGR_MODEL_NAME = "logistic"


class Logistic_regression_params(BaseModel):
    penalty = 'l2'
    max_iter = 10000


KNN_MODEL_NAME = "knn"


class K_nearest_neighbors_params(BaseModel):
    n_neighbors = 5
    weights = 'uniform'


def logistic_regression(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series,
                        params: Logistic_regression_params) -> np.float64:
    """
    Trains a logistic regression model on the provided training data.
    
    params: Logistic_regression_params
        penalty: {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
            Specify the norm of the penalty
        max_iter:
            Maximum number of iterations taken for the solvers to converge
            
    Returns the tuple with accuracy score and classification report.
    """

    model = LogisticRegression(max_iter=params.max_iter,
                               penalty=params.penalty)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test,
                          y_pred), classification_report(y_test, y_pred)


def k_nearest_neighbors(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series,
                        params: K_nearest_neighbors_params) -> np.float64:
    """
    Trains a k-nearest neighbors classifier on the provided training data.
    
    params: K_nearest_neighbors_params   
        n_neighbors: int, default=5
        Number of neighbors to use by default for kneighbors queries.

        weights : {'uniform', 'distance'} , default='uniform'
            Weight function used in prediction. 
        
    Returns the tuple with accuracy score and classification report.
    """

    model = KNeighborsClassifier(n_neighbors=params.n_neighbors,
                                 weights=params.weights)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test,
                          y_pred), classification_report(y_test, y_pred)


class Logistic_regression_runner(BaseModel):
    name = "Logistic Regression"
    description = """ Trains the logistic regression model | Params: | penalty: {'l1', 'l2', 'elasticnet', 'none'} - Specify the norm of the penalty
    max_iter: int - Maximum number of iterations taken for the solvers to converge"""

    def run(self, param: Logistic_regression_params):

        df = prepare_data_frame()
        X, y = split_df(df)

        X_train, X_test, y_train, y_test = preprocessing(X, y)
        return logistic_regression(X_train, X_test, y_train, y_test, param)


class K_nearest_neighbors_runner(BaseModel):
    name = "K-Nearest Neighbors"
    description = "Trains a k-nearest neighbors classifier. | Params: |n_neighbors: int, default=5 Number of neighbors to use by default for kneighbors queries. weights : {'uniform', 'distance'} , default='uniform' Weight function used in prediction."

    def run(self, param: K_nearest_neighbors_params):

        df = prepare_data_frame()
        X, y = split_df(df)

        X_train, X_test, y_train, y_test = preprocessing(X, y)
        return k_nearest_neighbors(X_train, X_test, y_train, y_test, param)


if __name__ == "__main__":
    df = prepare_data_frame()
    X, y = split_df(df)
    X_train, X_test, y_train, y_test = preprocessing(X, y)

    log_reg_score = logistic_regression(X_train, X_test, y_train, y_test,
                                        Logistic_regression_params())
    knn_score = k_nearest_neighbors(X_train, X_test, y_train, y_test,
                                    K_nearest_neighbors_params())

    print_results("Logistic Regression", log_reg_score)
    print_results("K-Nearest Neigbours", knn_score)
