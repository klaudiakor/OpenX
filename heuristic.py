import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, classification_report

from prepare_dataframe import prepare_data_frame, split_df, print_results

HEURISTIC_MODEL_NAME = "heuristic"


def heuristic_classifier(
        X: pd.DataFrame,
        y: pd.Series) -> tuple[tuple[np.float64, str], np.ndarray]:
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
    (accuracy, report), y_pred
        A tuple containing the accuracy and classification report and prediction.

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
    return (accuracy, report), y_pred


class Heuristic_model_runner(BaseModel):
    name = "Heuristic"
    description = "A simple heuristic classifier that uses wilderness area features to predict cover type labels. | Gets no params."

    def run(self) -> dict:
        df = prepare_data_frame()
        X, y = split_df(df)
        score, y_pred = heuristic_classifier(X, y)
        return {"accuracy": score[0], "prediction": y_pred}


if __name__ == "__main__":
    df = prepare_data_frame()
    X, y = split_df(df)

    score = heuristic_classifier(X, y)[0]
    print_results("Heuristic classifier", score)
