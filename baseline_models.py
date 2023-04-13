import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from prepare_dataframe import *


def logistic_regression(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series) -> np.float64:
    """
    Trains a logistic regression model on the provided training data.
    Returns the tuple with accuracy score and classification report.
    """

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test,
                          y_pred), classification_report(y_test, y_pred)


def k_nearest_neighbors(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series) -> np.float64:
    """
    Trains a k-nearest neighbors classifier on the provided training data.
    Returns the tuple with accuracy score and classification report.
    """

    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test,
                          y_pred), classification_report(y_test, y_pred)


if __name__ == "__main__":
    df = prepare_data_frame()
    X, y = split_df(df)
    X_train, X_test, y_train, y_test = preprocessing(X, y)

    log_reg_score = logistic_regression(X_train, X_test, y_train, y_test)
    knn_score = k_nearest_neighbors(X_train, X_test, y_train, y_test)

    print_results("Logistic Regression", log_reg_score)
    print_results("K-Nearest Neigbours", knn_score)
