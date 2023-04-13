from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

from prepare_dataframe import *


def logistic_regression(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series) -> np.float64:
    """
    Trains a logistic regression model on the provided training data 
    and returns the accuracy score on the test data.
    """

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    return score


def k_nearest_neighbors(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series) -> np.float64:
    """
    Trains a k-nearest neighbors classifier on the provided training data 
    and returns the accuracy score on the test data.
    """

    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    return score


if __name__ == "__main__":
    df = prepare_data_frame()
    X, y = split_df(df)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    log_reg_score = logistic_regression(X_train, X_test, y_train, y_test)
    knn_score = k_nearest_neighbors(X_train, X_test, y_train, y_test)

    print(f"Logistic Regression accuracy score: {log_reg_score}")  #TODO: 72%
    print(f"K-Nearest Neigbours accuracy score: {knn_score}")  #TODO: 92%
