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

def Encoder(X, X_test, all_categorical: bool = True, features=None):
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore')

    if all_categorical:
        cat_features = X.select_dtypes(exclude='number').columns
    else:
        if not features:
            raise ValueError("Please provide the list of categorical features to encode.")
        cat_features = features

    encoded_arr = encoder.fit_transform(X[cat_features])
    test_encoded_arr = encoder.transform(X_test[cat_features])

    encoded_cols = encoder.get_feature_names_out(cat_features)
    encoded_df = pd.DataFrame(encoded_arr, columns=encoded_cols, index=X.index)
    encoded_df = pd.concat([X.drop(columns=cat_features), encoded_df], axis=1)
    test_encoded_df = pd.DataFrame(test_encoded_arr, columns=encoded_cols, index=X_test.index) # type: ignore
    test_encoded_df = pd.concat([X_test.drop(columns=cat_features), test_encoded_df], axis=1)
    
    return encoded_df, test_encoded_df


