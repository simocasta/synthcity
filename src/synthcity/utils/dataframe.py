# third party
import pandas as pd
from typing import List, Tuple, Union


def constant_columns(dataframe: pd.DataFrame) -> list:
    """
    Find constant value columns in a pandas dataframe.
    """
    return discrete_columns(dataframe, 1)


def discrete_columns(
    dataframe: pd.DataFrame, return_counts: bool = False
) -> List[Union[str, Tuple[str, int]]]:
    """
    Identify columns in a pandas DataFrame that contain discrete (integer or boolean) values.
    """
    discrete_cols = []
    for col in dataframe.columns:
        if pd.api.types.is_integer_dtype(dataframe[col]) or pd.api.types.is_bool_dtype(dataframe[col]):
            if return_counts:
                unique_count = dataframe[col].nunique()
                discrete_cols.append((col, unique_count))
            else:
                discrete_cols.append(col)
    return discrete_cols

