import pandas as pd
import numpy as np
from src.utils.config import load_config
from src.data.preprocessing import preprocessing_pipeline, save_preprocessed_data

def main():

    # Setup 
    config = load_config()

    # Access data
    train_raw_data = config["paths"]["train_raw_data"]
    test_raw_data = config["paths"]["test_raw_data"]

    # Read that data
    train_df = pd.read_csv(train_raw_data)
    test_df = pd.read_csv(test_raw_data)

    # Give these dataframes to preprocessing pipeline.
    X_train_encoded, X_val_encoded, X_test_encoded, y_train_encoded, y_val_encoded, y_test_encoded = preprocessing_pipeline(df=train_df, test_dataset=test_df, drop_features=["UserRating", "CustomerID"])

    #Save the preprocessed data
    save_preprocessed_data(X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded, X_val=X_val_encoded, y_val=y_val_encoded)


if __name__ == "__main__":
    main()