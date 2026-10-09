import pandas as pd
import numpy as np
from src.utils.config import load_config
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Literal
from sklearn.model_selection import train_test_split
from feature_engine.outliers import ArbitraryOutlierCapper

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

def scaling(X, X_test):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    return X_scaled, X_test_scaled


# Function for Outliers Removal
# Function for Validation set and transform in preprocesser() function.

def remove_outliers(df: pd.DataFrame, features=None):
    """
    Remove outliers from the dataframe using the IQR method.
    
    Args:
        df(pd.DataFrame): Input dataframe to remove outliers from.
        features(list): list of features to check for outliers.
    Returns:
        df(pd.DataFrame): Dataframe without outliers.
    """

    def find_lower_upper_bound(feature):
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound

    if features is not None:
        features = [col for col in features if col in df.columns]
    else:
        features = df.select_dtypes(include='number').columns.tolist()
    
    if not features:
        return df

    bound_dictionary = {}
    for feature in features:
        lower_bound, upper_bound = find_lower_upper_bound(feature)
        bound_dictionary[f"{feature}_bounds"] = (lower_bound, upper_bound)

    for feature in features:
        lower_bound, upper_bound = bound_dictionary[f"{feature}_bounds"]
        capper = ArbitraryOutlierCapper(max_capping_dict={feature: upper_bound}, min_capping_dict={feature: lower_bound})
        df[[feature]] = capper.fit_transform(df[[feature]])

    return df

def validation_set(X_train, y_train):
    """
    Validation set for the training data.
    args:
        X_train: training features.
        y_train: training target labels.
    return:
        X_train: training features
        X_val: validation features
        y_train: training target labels
        y_val: validation target labels
     """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"])

    return X_train, X_val, y_train, y_val

# have error with remove_outliers() func when using in preprocesser() function. It is not working properly. need to check it.