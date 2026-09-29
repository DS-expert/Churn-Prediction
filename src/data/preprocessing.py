import pandas as pd
import numpy as np
from src.utils.config import load_config
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from typing import Literal

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
    df = df.copy()

    for feature in df.columns:
        if df[feature].isnull().any():
            if df[feature].dtype in [int, float]:
                df[feature] = df[feature].fillna(df[feature].median())
            else:
                df[feature] = df[feature].fillna("missing")
            
    return df

def drop_feature(df, features: list):
    """
    Drop the feature from dataset

    Args:
        df(pd.DataFrame): Input dataframe to drop feature from it.
        features(list): list of Feature which have to drop.
    returns:
        df(pd.DataFrame): Return dataframe without that feature.
    """

    df_drop = df.drop(columns=features)
    return df_drop

def Encoder(X, X_test, all_categorical=Literal[True, False], features=[]):
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    if all_categorical == True:
        cat_feature = X.select_dtypes(exclude='number').columns
        encoded_df = encoder.fit_transform(X[cat_feature])
        test_encoded_df = encoder.transform(X_test[cat_feature])
    elif all_categorical == False:
        if features is []:
            raise ValueError(f"The 'features' required when all_categorical is {all_categorical}")
        
        encoded_df = encoder.fit_transform(X[features])
        test_encoded_df = encoder.transform(X_test[features])

    return encoded_df, test_encoded_df


