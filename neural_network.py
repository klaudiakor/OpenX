import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, classification_report

from prepare_dataframe import *


def prepare_sets(
    X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    This function prepares the data sets needed to train a neural network. 
    
    It splits the test set into a validation set 
    and converts the target vectors into a binary matrix representation using one-hot encoding.
    """

    X_test, X_valid, y_test, y_valid = train_test_split(X_test,
                                                        y_test,
                                                        test_size=0.50,
                                                        random_state=1)

    #one hot encoding
    TARGET_CATEGORIES_NUMBER = 7
    y_train = np_utils.to_categorical(y_train - 1, TARGET_CATEGORIES_NUMBER)
    y_test = np_utils.to_categorical(y_test - 1, TARGET_CATEGORIES_NUMBER)
    y_valid = np_utils.to_categorical(y_valid - 1, TARGET_CATEGORIES_NUMBER)

    return X_test, X_valid, y_train, y_test, y_valid


def reverse_one_hot_encoding(
        y_test: np.ndarray,
        y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert target vectors one-hot encoding to label encoding 
    by taking the index of the maximum value along the second dimension of the array.
    """

    y_label = np.argmax(y_test, axis=1)
    y_pred_label = np.argmax(y_pred, axis=1)
    print((y_label))

    return y_label, y_pred_label


def calculate_metrics(y_test: np.ndarray,
                      y_pred: np.ndarray) -> tuple[np.float64, str]:
    """
    Calculate accuracy score and classification report for a classification model.
    """

    target_names = [
        "1 -- Spruce/Fir", "2 -- Lodgepole Pine", "3 -- Ponderosa Pine",
        "4 -- Cottonwood/Willow", "5 -- Aspen", "6 -- Douglas-fir",
        "7 -- Krummholz"
    ]

    y_label, y_pred_label = reverse_one_hot_encoding(y_test, y_pred)

    nn_report = (classification_report(y_label,
                                       y_pred_label,
                                       target_names=target_names))
    nn_score = accuracy_score(y_label, y_pred_label)

    return nn_score, nn_report


def neural_network(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series,
                   y_test: pd.Series) -> tuple[np.float64, str]:
    """
    Trains a neural network model on the provided training data 
    and returns the accuracy score and classification report on the test data.
    """

    FEATURES_NUMBER = (X_train.shape[1])

    X_test, X_valid, y_train, y_test, y_valid = prepare_sets(
        X_test, y_train, y_test)

    model = Sequential()

    model.add(
        Dense(2 * FEATURES_NUMBER + 1,
              input_dim=FEATURES_NUMBER,
              activation='relu'))
    model.add(Dropout(0.2))  #TODO: ochrona przed przetrenowaniem
    model.add(Dense(7, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train,
              y_train,
              validation_data=(X_valid, y_valid),
              epochs=1,
              batch_size=32)

    y_pred = model.predict(X_test)

    return calculate_metrics(y_test, y_pred)


if __name__ == "__main__":

    df = prepare_data_frame()
    X, y = split_df(df)

    #scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.60,
                                                        random_state=1)

    nn_score, nn_report = neural_network(X_train, X_test, y_train, y_test)

    print(f"Neural Network accuracy {nn_score}")
