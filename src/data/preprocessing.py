import pandas as pd
import numpy as np
from src.utils.config import load_config

config = load_config()

# Handle missing values
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing value in dataframe

    Args:
        df(pd.DataFrame): Dataframe in which have missing values.
        
    returns:
        df(pd.DataFrame): DataFrame without missing value.
    """

    for feature in df.columns:
        if df[feature].isnull().sum() > 0:
            if df[feature].dtype in [int, float]:
                df[feature] = df[feature].fillna(df[feature].mean())
            else:
                df[feature] = df[feature].fillna("missing")

    
    return df