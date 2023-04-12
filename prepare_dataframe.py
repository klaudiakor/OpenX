import pandas as pd


def get_item_from_txt(path: str, column_index: int) -> list:
    """
    Reads a text file from the given path and returns a list of items
    extracted from the specified column.
    """
    items = []
    with open(path, "r") as f:
        for line in f:
            fields = line.split()
            items.append(fields[column_index])
    return items


def get_columns_names() -> list[str]:
    """
    return list of columns names
    """
    ATTRIBUTE_NAMES_COLUMN_INDEX = 0
    SOIL_TYPES_COLUMN_INDEX = 1

    attribute_names = get_item_from_txt(r"./data/column_names.txt",
                                        ATTRIBUTE_NAMES_COLUMN_INDEX)

    wilderness_areas = list(range(1, 5))

    soil_types = get_item_from_txt(r"./data/soil_types.txt",
                                   SOIL_TYPES_COLUMN_INDEX)

    # conctenate lists
    columns = []

    for name in attribute_names:
        if name == 'Wilderness_Area':
            for area in wilderness_areas:
                columns.append(name + '_' + str(area))
        elif name == 'Soil_Type':
            for soil_type in soil_types:
                columns.append(name + '_' + str(soil_type))
        else:
            columns.append(name)

    return columns


def prepare_data_frame():
    """
    Load the Covertype Data Set and returns dataframe with column names.
    """
    df = pd.read_csv('data/covtype.data')
    df.columns = get_columns_names()
    return df


def split_df(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    returns:

    X - dataframe with all features 

    y - vector with response
    """

    X = dataframe.iloc[:, 0:-1]
    y = dataframe.iloc[:, -1]

    return X, y


if __name__ == "__main__":
    print(get_columns_names())