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

def preprocesser(df: pd.DataFrame, target="Churn", all_categorical: bool = True, features=None):
    """
    Preprocess the data by handling missing values, encoding_categorical features, and scaling numercal features.
    Args:
        df(pd.DataFrame): Input dataframe to preprocess.
        target(str): Name of the target variable.
        all_categorical(bool): If true all categorical features will be encoded, otherwise only the features provided in the list will be encoded.
        features(list): List of features to encode if all_categorical is False.
    
    Returns:
        X_train(np.ndarray): Preprocessed training features.
        X_test(np.ndarray): Preprocessed testing features.
        y_train(np.ndarray): Training target variable.
        y_test(np.ndarray): Testing target variable.
    """

    if all_categorical:
        cat_features = df.select_dtypes(exclude='number').columns
        if target in cat_features:
            cat_features = cat_features.drop(target)
    else:
        if not features:
            raise ValueError("Please provide the list of categorical features to encode.")
        cat_features = features

    num_features = df.select_dtypes(include='number').columns

    if target in num_features:
        num_features = num_features.drop(target)
    
    handle_missing_transformer = FunctionTransformer(handle_missing_values)

    categoircal_pipeline = Pipeline(steps=[
        ('missing_handle_values', handle_missing_transformer),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    numerical_pipeline = Pipeline(steps=[
        ("missing_handle_values", handle_missing_transformer),
        ("Scaler", StandardScaler())
    ])

    preprocesser = ColumnTransformer(
        transformers=[
            ("categoircal_pipeline", categoircal_pipeline, cat_features),
            ("Numerical pipeline", numerical_pipeline, num_features)
        ], remainder='passthrough'
    )

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = preprocesser.fit_transform(X_train)
    X_test = preprocesser.transform(X_test)

    return X_train, X_test, y_train, y_test

# Function for Outliers Removal
# Function for Validation set and transform in preprocesser() function.

def remove_outliers(df: pd.DataFrame, features: list):
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

    bound_dictionary = {}
    for feature in features:
        lower_bound, upper_bound = find_lower_upper_bound(feature)
        bound_dictionary[f"{feature}_bounds"] = (lower_bound, upper_bound)

    for feature in features:
        lower_bound, upper_bound = bound_dictionary[f"{feature}_bounds"]
        capper = ArbitraryOutlierCapper(max_capping_dict={feature: upper_bound}, min_capping_dict={feature: lower_bound})
        df[[feature]] = capper.fit_transform(df[[feature]])

    return df

