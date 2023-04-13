import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from prepare_dataframe import *


def heuristic_classifier(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    A simple heuristic classifier that uses wilderness area features to predict cover type labels.

    Parameters:
    -----------
    X : pandas DataFrame
        A DataFrame containing feature data to be used for prediction.
    y : pandas Series
        A Series containing target data to be predicted.

    Returns:
    --------
    (accuracy, report) 
        A tuple containing the accuracy and classification report.

    Idea:
    -----
    Dependencies between wilderness areas and cover types from coctype.info file:

    WILDERNESS AREA     -   COVER TYPE 

    Rawah (1)           -   lodgepole pine (2)
    Neouta (2)          -   spruce/fir (1)
    Comanche Peak (3)   -   lodgepole pine (2)
    Cache la Poudre (4) -   Ponderosa pine (3)
    """

    X_heuristic = X[[
        'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3',
        'Wilderness_Area_4'
    ]]

    y_pred = []
    for i in range(len(X_heuristic)):
        row = X_heuristic.iloc[i]
        if row['Wilderness_Area_1'] or row['Wilderness_Area_3']:
            y_pred.append(2)
        elif row['Wilderness_Area_2']:
            y_pred.append(1)
        elif row['Wilderness_Area_4']:
            y_pred.append(3)

    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    return accuracy, report


if __name__ == "__main__":
    df = prepare_data_frame()
    X, y = split_df(df)

    accuracy, report = heuristic_classifier(X, y)
    print(f"Accuracy score: {accuracy}")
    print(report)