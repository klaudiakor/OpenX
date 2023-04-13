import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, classification_report
from itertools import product

from prepare_dataframe import *

FEATURES_NUMBER = 54
TARGET_CATEGORIES_NUMBER = 7


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

    return y_label, y_pred_label


def calculate_metrics(y_test: np.ndarray,
                      y_pred: np.ndarray) -> tuple[np.float64, str]:
    """
    Calculate accuracy score and classification report for a classification model.
    """

    y_label, y_pred_label = reverse_one_hot_encoding(y_test, y_pred)

    nn_report = (classification_report(
        y_label,
        y_pred_label,
    ))
    nn_score = accuracy_score(y_label, y_pred_label)

    return nn_score, nn_report


def create_model(hidden_layers_units: list[int], activation_function: str,
                 loss: str, optimizer: str,
                 metrics: str) -> keras.engine.sequential.Sequential:

    model = Sequential()

    model.add(
        Dense(2 * FEATURES_NUMBER + 1,
              input_dim=FEATURES_NUMBER,
              activation=activation_function))

    for i in range(len(hidden_layers_units)):
        model.add(Dense(hidden_layers_units[i],
                        activation=activation_function))

    model.add(Dense(TARGET_CATEGORIES_NUMBER, activation='sigmoid'))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def plot_training_curves(history: keras.callbacks.History):
    """
    Plots the training and validation loss for a given Keras history object.
    
    Parameters:
    ---
        history: A Keras history object, containing the training and validation loss
            values for each epoch.
    """
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("nn_training_curves.png")
    # plt.show()


def neural_network(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: pd.Series,
                   y_test: pd.Series,
                   hidden_layers_units: list[int],
                   activation_function: str,
                   loss: str,
                   optimizer: str,
                   metrics: str,
                   epochs: int,
                   batch_size: int,
                   visualization=False) -> tuple[np.float64, str]:
    """
    Trains a neural network model on the provided training data 
    and returns the accuracy score and classification report on the test data.

    Parameters:
    -----------
    X_train : pandas.DataFrame
        A DataFrame containing the training data.
    X_test : pandas.DataFrame
        A DataFrame containing the testing data.
    y_train : pandas.Series
        A Series containing the target values for the training data.
    y_test : pandas.Series
        A Series containing the target values for the testing data.
    hidden_layers_units: list[int]
        A number of neurons in each hidden layer.
    activation_function: str
        Eg. 'linear', 'relu', 'tanh', 'sigmoid'
    loss: str
        Loss function. Eg. 'binary_crossentropy'
    optimizer: str
        Eg. 'Adam', 'sgd'
    metrics: str
        Eg. 'accuracy', 'mse'
    epochs: int
        Number of epochs to train the model.
    batch_size: int
        Number of samples per gradient update.
    visualization: bool
        If set to True (default is False), the function will display the training curves.
        
    Returns:
    -----------
    Tuple with accuracy of model and classification report
    """

    X_test, X_valid, y_train, y_test, y_valid = prepare_sets(
        X_test, y_train, y_test)

    model = create_model(hidden_layers_units=hidden_layers_units,
                         activation_function=activation_function,
                         loss=loss,
                         optimizer=optimizer,
                         metrics=metrics)

    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_valid, y_valid),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0)

    y_pred = model.predict(X_test)

    if visualization:
        plot_training_curves(history)

    return calculate_metrics(y_test, y_pred)


def find_best_params(param_grid: dict, X_train: pd.DataFrame,
                     X_test: pd.DataFrame, y_train: pd.Series,
                     y_test: pd.Series) -> tuple:
    """
    This function searches for the combination of hyperparameters 
    that yields the highest accuracy on the testing data. 

    Parameters:
    -----------
    param_grid : dict
        A dictionary containing all the hyperparameters to be checked. The keys of the dictionary 
        should be the names of the hyperparameters, and the values should be a list of possible values 
        to search for that hyperparameter.

        Example:
            param_grid = {
                    'batch_size': [64, 128],
                    'epochs': [10, 20],
                    'optimizer': ['adam', 'sgd'],
                    'activation_function':['relu', 'tanh']
                    'hidden_layers_units': [[FEATURES_NUMBER // 2, FEATURES_NUMBER // 4], [FEATURES_NUMBER // 2], []]
                }

    X_train : pandas.DataFrame
        A DataFrame containing the training data.
    X_test : pandas.DataFrame
        A DataFrame containing the testing data.
    y_train : pandas.Series
        A Series containing the target values for the training data.
    y_test : pandas.Series
        A Series containing the target values for the testing data.
        
    Returns
    -----------
    A tuple containing the hyperparameters with the highest accuracy in the following order: 
        batch_size, epochs, optimizer, activation_function, and hidden_layers_units.
    """

    hyperparam_results = [
    ]  # list of accuracies for each set of hyperparameters

    all_hyperparams = list(product(*param_grid.values()))
    for hyperparams in all_hyperparams:
        batch_size, epochs, optimizer, activation_function, hidden_layers_units = hyperparams

        score = neural_network(X_train,
                               X_test,
                               y_train,
                               y_test,
                               hidden_layers_units=hidden_layers_units,
                               activation_function=activation_function,
                               loss='binary_crossentropy',
                               optimizer=optimizer,
                               metrics='accuracy',
                               epochs=epochs,
                               batch_size=batch_size)

        print('Hyperparameters:', hyperparams)
        print('Test accuracy:', score[0])
        hyperparam_results.append((score[0], hyperparams))

    return max(hyperparam_results, key=lambda x: x[0])[1]


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

    result = find_best_params(
        {
            'batch_size': [64, 128],
            'epochs': [10, 20],
            'optimizer': ['adam', 'sgd'],
            'activation_function': ['relu', 'tanh'],
            'hidden_layers_units': [[
                FEATURES_NUMBER // 2, FEATURES_NUMBER // 4
            ], [FEATURES_NUMBER // 2], []]
        }, X_train, X_test, y_train, y_test)

    print(result)
