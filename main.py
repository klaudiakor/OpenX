import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from prepare_dataframe import *

#PREPROCESSING
df = prepare_data_frame()
# print(df.shape)
# print(df.isna().sum())

X, y = split_df(df)

# train data - 70%
# test data - 15%
# validation data - 15%
X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                    y,
                                                    train_size=0.70,
                                                    random_state=1)
X_test, X_valid, y_test, y_valid = train_test_split(X_temp,
                                                    y_temp,
                                                    test_size=0.50,
                                                    random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# (64, 2, 'adam', 'tanh', [27, 13])