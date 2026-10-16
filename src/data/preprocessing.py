import pandas as pd
import numpy as np
from src.utils.config import load_config
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Literal
from sklearn.model_selection import train_test_split
from feature_engine.outliers import ArbitraryOutlierCapper
import joblib

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

def Encoder(X, X_test, all_categorical: bool = True, X_val=None, features=None):
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore')

    if all_categorical:
        cat_features = X.select_dtypes(exclude='number').columns
    else:
        if not features:
            raise ValueError("Please provide the list of categorical features to encode.")
        cat_features = features

    encoded_arr = encoder.fit_transform(X[cat_features])
    test_encoded_arr = encoder.transform(X_test[cat_features])
    if X_val is not None:
        val_encoded_arr = encoder.transform(X_val[cat_features])

    encoded_cols = encoder.get_feature_names_out(cat_features)
    encoded_df = pd.DataFrame(encoded_arr, columns=encoded_cols, index=X.index)
    encoded_df = pd.concat([X.drop(columns=cat_features), encoded_df], axis=1)
    test_encoded_df = pd.DataFrame(test_encoded_arr, columns=encoded_cols, index=X_test.index) # type: ignore
    test_encoded_df = pd.concat([X_test.drop(columns=cat_features), test_encoded_df], axis=1)
    if X_val is not None:
        val_encoded_df = pd.DataFrame(val_encoded_arr, columns=encoded_cols, index=X_val.index)
        val_encoded_df = pd.concat([X_val.drop(columns=cat_features), val_encoded_df], axis=1)
        X_val_encoded = val_encoded_df.to_numpy()
    X_encoded = encoded_df.to_numpy()
    X_test_encoded = test_encoded_df.to_numpy()
    if X_val is not None:
        return X_encoded, X_val_encoded, X_test_encoded
    else:
        return X_encoded, X_test_encoded

def scaling(X, X_test, X_val=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_scaled, X_val_scaled, X_test_scaled
    else:
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

def preprocessing_pipeline(df: pd.DataFrame, test_dataset=None, drop_features=None, target_feature="Churn"):
    """
    Preprocessing pipeline will preprocess the dataframe.

    args:
        df(pd.DataFrame): Input dataframe.
        testing_dataframe: If you have test dataset or dataframe.
    
    returns:
        X_train(np.ndarray): return train dataset numpy array.
        X_test(np.ndarray): return test dataset numpy array.
        X_val(np.ndarray): return validation dataset.
        y_train(np.ndarray): return training labels of dataset.
        y_test(np.ndarray): return testing labels of datasets.
        y_val(np.ndarray): return validation label of dataset
    """

    # Handle missing values
    df_semi_prep = handle_missing_values(df=df)
    if test_dataset is not None:
        test_df_semi_prep = handle_missing_values(test_dataset)
    # drop features from dataset if have
    if drop_features is not None:
        df_semi_prep = drop_feature(df, features=drop_features)
        test_df_semi_prep = drop_feature(df, features=drop_features)

    cat_features = df.select_dtypes(exclude='number').columns.tolist()
    num_features = df.select_dtypes(include='number').columns.tolist()

    # remove outliers from dataset
    df_semi_prep_without_outliers = remove_outliers(df_semi_prep)

    # Split the data into train, VAl and testing
    if test_dataset is not None:
        X_train = df.drop(columns=[target_feature])
        y_train = df[target_feature]
        X_test = test_dataset
    else:
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_feature]), df[target_feature], test_size=config["data"]["test_size"], random_state=config["data"]["random_state"])

    # then split into train and validation set
    X_train, X_val, y_train, y_val = validation_set(X_train=X_train, y_train=y_train)

    #Encode the Split data
    X_train_encoded, X_val_encoded, X_test_encoded = Encoder(X_train, X_test, X_val=X_val)

    # Scaling
    X_train_scaled, X_val_scaled, X_test_scaled = scaling(X_train_encoded, X_test_encoded, X_val=X_val_encoded)

    if test_dataset is not None:
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val
    else:
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


# Complete preprocessing_pipeline(), Add Encoding and Scaling and spliting also Imbalanced Dataset(optional).

def save_preprocessed_data(path: str, X, X_test, y_train, y_test, X_val=None, y_val=None):
    joblib.dump(X, config["paths"]["X_train_preprocessed"])
    joblib.dump(X_test, config["paths"]["X_test_preprocessed"])
    joblib.dump(y_train, config["paths"]["y_train_preprocessed"])
    joblib.dump(y_test, config["paths"]["y_test_preprocessed"])
    if X_val is not None:
        joblib.dump(X_val, config["paths"]["X_val_preprocessed"])

    if y_val is not None:
        joblib.dump(y_val, config["paths"]["X_val_preprocessed"])
