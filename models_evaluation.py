from prepare_dataframe import *
from heuristic import *
from baseline_models import *
from neural_network import *

#LOAD DATA
df = prepare_data_frame()
# print(df.isna().sum()) # there is no lacking data
X, y = split_df(df)

print(
    "Target classes:\n 1 -- Spruce/Fir \n 2 -- Lodgepole Pine \n 3 -- Ponderosa Pine \n 4 -- Cottonwood/Willow \n 5 -- Aspen \n 6 -- Douglas-fir \n 7 -- Krummholz"
)

# Heuristic classifier
heuristic_score = heuristic_classifier(X, y)
print_results("Heuristic Classifier", heuristic_score)

X_train, X_test, y_train, y_test = preprocessing(X, y)

# Baseline models
log_reg_score = logistic_regression(X_train,
                                    X_test,
                                    y_train,
                                    y_test,
                                    params=Logistic_regression_params())[0]
print_results("Logistic Regression", log_reg_score)

knn_score = k_nearest_neighbors(X_train,
                                X_test,
                                y_train,
                                y_test,
                                params=K_nearest_neighbors_params())[0]
print_results("K-Nearest Neighbors", knn_score)

# Neural network

# Best hyperparameters found by "find_best_params" function with the following possibilities:
# 'batch_size': [64, 128],
# 'epochs': [10, 20],
# 'optimizer': ['adam', 'sgd'],
# 'activation_function': ['relu', 'tanh'],
# 'hidden_layers_units': [[
#     FEATURES_NUMBER // 2, FEATURES_NUMBER // 4
# ], [FEATURES_NUMBER // 2], []]

# (batch_size: 64, epochs: 20, optimizer: 'adam', activation_function: 'tanh', hidden_layers_units: [27, 13])

params = Neural_network_params(batch_size=64,
                               epochs=20,
                               optimizer='adam',
                               activation_function='tanh',
                               hidden_layers_units=[27, 13])

nn_score = neural_network(X_train,
                          X_test,
                          y_train,
                          y_test,
                          params,
                          visualization=True)[0]
#   hidden_layers_units=[27, 13],
#   activation_function='tanh',
#   loss='binary_crossentropy',
#   optimizer='adam',
#   metrics='accuracy',
#   epochs=20,
#   batch_size=64,
#   visualization=True)
#TODO

print_results("Neural Network", nn_score)
