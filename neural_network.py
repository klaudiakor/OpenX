import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from itertools import product

from prepare_dataframe import *

TARGET_CATEGORIES_NUMBER = 7
NEURAL_NETWORK_MODEL_NAME = "nn"
FEATURES_NAMES = get_columns_names()[:-1]  # without target name


class Neural_network_params(BaseModel):
    hidden_layers_units = [27]
    activation_function = 'relu'
    loss = 'binary_crossentropy'
    optimizer = 'adam'
    metrics = 'accuracy'
    epochs = 10
    batch_size = 32
    features_names = FEATURES_NAMES


def prepare_sets(
    X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    This function prepares the data sets needed to train a neural network. 
    
    It splits the test set to create a validation set 
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
    Convert target vectors one-hot encoded to label encode 
    by taking the index of the maximum value along the second dimension of the array.
    """

    y_label = np.argmax(y_test, axis=1)
    y_pred_label = np.argmax(y_pred, axis=1)

    return y_label, y_pred_label


def calculate_metrics(y_test: np.ndarray,
                      y_pred: np.ndarray) -> tuple[np.float64, str]:
    """
    Calculate the accuracy score and classification report for a classification model, that used one hot encoding.
    """

    y_label, y_pred_label = reverse_one_hot_encoding(y_test, y_pred)

    nn_report = (classification_report(
        y_label,
        y_pred_label,
    ))
    nn_score = accuracy_score(y_label, y_pred_label)

    return nn_score, nn_report


def create_model(hidden_layers_units: list[int], activation_function: str,
                 loss: str, optimizer: str, metrics: str,
                 features_num: int) -> keras.engine.sequential.Sequential:
    """
    Creates a neural network model using Keras with the specified hyperparameters.

    Parameters:
    ---

    hidden_layers_units: list[int]
        A list of integers representing the number of neurons for each hidden layer 
        in the neural network. The first layer is always created with (2 * features_num + 1) neurons.
    activation_function, loss, optimizer, metrics
        standard hyperparameters for neural network (check Keras library)
    features_num: int
        Number of features which are used to create the model (input dimension)
    
    Returns:
    ----

    model: keras.engine.sequential.Sequential
        created model
    """

    model = Sequential()

    model.add(
        Dense(2 * features_num + 1,
              input_dim=features_num,
              activation=activation_function))

    for i in range(len(hidden_layers_units)):
        model.add(Dense(hidden_layers_units[i],
                        activation=activation_function))

    model.add(Dense(TARGET_CATEGORIES_NUMBER, activation='sigmoid'))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def plot_training_curves(history: keras.callbacks.History):
    """
    Save plot of the training and validation loss for a given Keras history object.
    
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


def neural_network(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        params: Neural_network_params,
        visualization=False) -> tuple[tuple[np.float64, str], np.ndarray]:
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
    params: Neural_network_params
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
    Tuple with accuracy of model and classification report and vector of prediction.
    """

    X_test, X_valid, y_train, y_test, y_valid = prepare_sets(
        X_test, y_train, y_test)

    model = create_model(hidden_layers_units=params.hidden_layers_units,
                         activation_function=params.activation_function,
                         loss=params.loss,
                         optimizer=params.optimizer,
                         metrics=params.metrics,
                         features_num=len(params.features_names))

    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_valid, y_valid),
                        epochs=params.epochs,
                        batch_size=params.batch_size,
                        verbose=0)

    y_pred = model.predict(X_test)

    if visualization:
        plot_training_curves(history)

    score = calculate_metrics(y_test, y_pred)
    y_pred_label = np.argmax(y_test, axis=1)
    return (score, y_pred_label)


def find_best_params(param_grid: dict, features: list[str]) -> tuple:
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
                    'hidden_layers_units': [[FEATURES_NUMBER // 2, FEATURES_NUMBER // 4], [FEATURES_NUMBER // 2], []],
                    'features_names': [['"Wilderness_Area_1", "Wilderness_Area_2", "Wilderness_Area_3", "Wilderness_Area_4"']]
                }

    features: list[str]
        A list of feature names that will be used to create model.
        
    Returns
    -----------
    A tuple containing the hyperparameters with the highest accuracy in the following order: 
        batch_size, epochs, optimizer, activation_function, and hidden_layers_units.
    """
    params = Neural_network_params(features_names=features)

    X_train, X_test, y_train, y_test = prepare_data_for_model_with_selected_features(
        features)

    hyperparam_results = [
    ]  # list of accuracies for each set of hyperparameters

    all_hyperparams = list(product(*param_grid.values()))
    for hyperparams in all_hyperparams:
        batch_size, epochs, optimizer, activation_function, hidden_layers_units = hyperparams

        params.hidden_layers_units = hidden_layers_units
        params.activation_function = activation_function
        params.optimizer = optimizer
        params.epochs = epochs
        params.batch_size = batch_size

        score = neural_network(X_train, X_test, y_train, y_test, params)

        print('Hyperparameters:', hyperparams)
        print('Test accuracy:', score[0][0])
        hyperparam_results.append((score[0][0], hyperparams))

    return max(hyperparam_results, key=lambda x: x[0])[1]


class Neural_network_runner(BaseModel):
    name = "Neural network"
    description = """Trains a neural network model. 
    Params: hidden_layers_units: list[int] - A number of neurons in each hidden layer. | activation_function: str - Eg. 'linear', 'relu', 'tanh', 'sigmoid' | loss: str - Loss function. Eg. 'binary_crossentropy' | optimizer: str - Eg. 'Adam', 'sgd' | metrics: str - Eg. 'accuracy', 'mse' | epochs: int - Number of epochs to train the model. | batch_size: int - Number of samples per gradient update."""

    def run(self, param: Neural_network_params):

        X_train, X_test, y_train, y_test = prepare_data_for_model_with_selected_features(
            param.features_names)

        score, y_pred = neural_network(X_train, X_test, y_train, y_test, param)

        return {"accuracy": score[0], "prediction": y_pred.tolist()}


if __name__ == "__main__":

    result = find_best_params(
        {
            'batch_size': [64, 128],
            'epochs': [10, 20],
            'optimizer': ['adam', 'sgd'],
            'activation_function': ['relu', 'tanh'],
            'hidden_layers_units': [[54 // 2, 54 // 4], [54 // 2], []],
        },
        features=FEATURES_NAMES)

    print(result)
