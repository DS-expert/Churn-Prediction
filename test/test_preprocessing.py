import pytest
from src.utils.config import load_config
from src.data.preprocessing import handle_missing_values, drop_feature, Encoder, scaling, remove_outliers, preprocessing_pipeline, save_preprocessed_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pytest_check import check
import numpy as np
from pathlib import Path

config = load_config()

def test_handle_missing_values():
    # Arrange
    data = {
        "Tenure": [12, np.nan, 24, 5, np.nan, 36],
        "MonthlyCharges": [70.5, 55.0, np.nan, 90.2, 30.0, np.nan],
        "TotalCharges": [800.0, 200.5, 1500.0, np.nan, 400.0, 2200.0],
        "Contract": ["Month-to-month", "One year", np.nan, "Two year", "Month-to-month", np.nan],
        "InternetService": ["DSL", np.nan, "Fiber optic", "DSL", np.nan, "Fiber optic"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", np.nan, "Credit card", "Electronic check"],
        "Churn": ["Yes", "No", "No", "Yes", "No", "Yes"]
    }

    df = pd.DataFrame(data)

    # Act
    result = handle_missing_values(df)

    

    # Assert
    assert result.isnull().sum().sum() == 0, f"There have some missing value still in dataframe {result}"

    # Check result type
    assert isinstance(result, pd.DataFrame), "Your result in don't have in pd.DataFrame format"

    # 'missing' in categorical features
    cat_col = df.select_dtypes(exclude='number').columns

    has_missing_value = (result[cat_col] == "missing").any().any()

    assert has_missing_value, "Categorical features doens't hold 'missing' placeholder."

def test_drop_feature():
    # arrange
    data = {
            "Tenure": [12, np.nan, 24, 5, np.nan, 36],
            "MonthlyCharges": [70.5, 55.0, np.nan, 90.2, 30.0, np.nan],
            "TotalCharges": [800.0, 200.5, 1500.0, np.nan, 400.0, 2200.0],
            "Contract": ["Month-to-month", "One year", np.nan, "Two year", "Month-to-month", np.nan],
            "InternetService": ["DSL", np.nan, "Fiber optic", "DSL", np.nan, "Fiber optic"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", np.nan, "Credit card", "Electronic check"],
            "Churn": ["Yes", "No", "No", "Yes", "No", "Yes"]
        }

    df = pd.DataFrame(data)

    # act
    result = drop_feature(df, features=["Tenure", "MonthlyCharges"])

    #assert
    assert result.shape[1] < df.shape[1], "Features still not drop from the dataframe."

def test_encoder():
    #act
    data = {
                "Tenure": [12, 20, 24, 5, 23, 36],
                "MonthlyCharges": [70.5, 55.0, 45.0, 90.2, 30.0, 28.2],
                "TotalCharges": [800.0, 200.5, 1500.0, 520.0, 400.0, 2200.0],
                "Contract": ["Month-to-month", "One year", "Three year", "Two year", "Month-to-month", "Month-to-month"],
                "InternetService": ["DSL", "DSL", "Fiber optic", "DSL", "Fiber optic", "Fiber optic"],
                "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Debit card", "Credit card", "Electronic check"],
                "Churn": ["Yes", "No", "No", "Yes", "No", "Yes"]
            }

    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["Churn"]), df["Churn"], test_size=0.2, random_state=42)

    #act # Comment code is tested when the return result of Encoder() was in Dataframe(pd.DataFrame)
    result1, result1_test = Encoder(X_train, X_test, all_categorical=True)
    check.equal(result1.shape[1], result1_test.shape[1], "Encoded train and test data should have the same number of columns.")
    check.is_true(isinstance(result1, np.ndarray), "Encoded return value should be a DataFrame.")
    check.is_true(isinstance(result1_test, np.ndarray), "Encoded return value should be a DataFrame")
    check.greater(result1.shape[1], X_train.shape[1], "Encoded data should have more columns than the original data due to one-hot encoding.")
    # cat_features = X_train.select_dtypes(exclude='number') # This comment was tested when return was in dataframe.
    # result_cat = result1.select_dtypes(exclude='number')
    # check.greater(cat_features.shape[1], result_cat.shape[1], "Encoded categorical features should have more columns than the original category features")

    with pytest.raises(ValueError):
        Encoder(X_train, X_test, all_categorical=False)

    result3, result3_test = Encoder(X_train, X_test, all_categorical=False, features=["Contract", "InternetService"])
    check.equal(result3.shape[1], result3_test.shape[1], "Encoded train and test data should have the same number of columns.")
    check.is_true(isinstance(result3, np.ndarray), "Encoded return value should be a DataFrame.")
    check.is_true(isinstance(result3_test, np.ndarray), "Encoded return value should be a DataFrame")
    # check.is_in("PaymentMethod", result3.columns, f"Encoded data should contain PaymentMethod columns, but it is missing")
    # check.equal(X_train["PaymentMethod"].dtype, result3["PaymentMethod"].dtype, "Encoded data should have the same data types as the original data")
    # forbidden_cols = [col for col in result3.columns if col.startswith("PaymentMethod_")]
    # check.equal(len(forbidden_cols), 0, "Encoded data should not contain extra columns which isn't provided.")
    # pd.testing.assert_series_equal(result3["PaymentMethod"], X_train["PaymentMethod"])

def test_scaling():
    #arrange
    data = {
                "Tenure": [12, 20, 24, 5, 23, 36],
                "MonthlyCharges": [70.5, 55.0, 45.0, 90.2, 30.0, 28.2],
                "TotalCharges": [800.0, 200.5, 1500.0, 520.0, 400.0, 2200.0]
            }

    # dataframe
    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(df, np.random.randint(0, 2, size=len(df)), test_size=0.2, random_state=42)

    #act
    result1, result1_test = scaling(X_train, X_test)

    #assert
    check.is_instance(result1, np.ndarray, "Scaled return value should be a numpy array.")
    check.is_instance(result1_test, np.ndarray, "Scaled return value should be a numpy array.")
    check.equal(result1.shape[1], result1_test.shape[1], "Scaled train and test data should have the same number of features.")
    check.equal(result1.shape[1], X_train.shape[1], "Scaled data should have the same number of features as the original data.")

def test_remove_outliers():
    #arrange
    test_features_data = {
    "feature_A": [10.2, 11.1, 10.5, 10.8, 11.0],  # Normal data (around 10-11)
    "feature_B": [20.5, 19.8, 21.0, 20.2, 19.9],  # Normal data (around 20)
    "feature_C": [10.1, 10.5, 950.0, 10.3, 10.9], # OUTLIER: 950.0 is abnormally high
    "feature_D": [15.2, 14.8, 15.1, 15.5, 14.9],  # Normal data (around 15)
    "feature_E": [50.1, 49.8, -800.0, 50.5, 50.2] # OUTLIER: -800.0 is abnormally low
    }
    df = pd.DataFrame(test_features_data)
    print(df.shape)
    #act
    result = remove_outliers(df, features=["feature_C", "feature_E"])
    print(result)
    print(result.shape)

    #assert
    check.is_instance(result, pd.DataFrame, "Result should be a DataFrame after removing outliers")
    check.equal(df.shape[1], result.shape[1], "Result should have same features as original data.")

    Q1 = df["feature_C"].quantile(0.25)
    Q3 = df["feature_C"].quantile(0.75)

    IQR = Q3 - Q1
    lower_bound_c = Q1 - 1.5 * IQR
    upper_bound_c = Q3 + 1.5 * IQR

    check.equal((result["feature_C"] > upper_bound_c).any(), False, "Result dataframe should not have upper_bound outliers in feature")
    check.equal((result["feature_C"] < lower_bound_c).any(), False, "Result should have not value lower than lower_bound in feature")

def test_preprocessing_pipeline():
    #arrange
    train_features_data = {
        "feature_A": [10.2, 11.1, 10.5, 10.8, 11.0, 10.5],  # Normal data (around 10-11)
        "feature_B": [20.5, 19.8, 21.0, 20.2, 19.9, 22.0],  # Normal data (around 20)
        "feature_C": [10.1, 10.5, 950.0, 10.3, 10.9, 11.1], # OUTLIER: 950.0 is abnormally high
        "feature_D": [15.2, 14.8, 15.1, 15.5, 14.9, 15.6],  # Normal data (around 15)
        "feature_E": [50.1, 49.8, -800.0, 50.5, 50.2, 50.4],
        "Contract": ["Month-to-month", "One year", np.nan, "Two year", "Month-to-month", np.nan],
        "InternetService": ["DSL", np.nan, "Fiber optic", "DSL", np.nan, "Fiber optic"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", np.nan, "Credit card", "Electronic check"],
        "Churn": ["Yes", "No", "No", "Yes", "No", "Yes"] # OUTLIER: -800.0 is abnormally low
        }

    test_features_data = {
            "feature_A": [10.2, 11.1, 10.5, 10.8],  # Normal data (around 10-11)
            "feature_B": [20.5, 19.8, 21.0, 20.2],  # Normal data (around 20)
            "feature_C": [10.1, 10.5, 950.0, 10.3], # OUTLIER: 950.0 is abnormally high
            "feature_D": [15.2, 14.8, 15.1, 15.5],  # Normal data (around 15)
            "feature_E": [50.1, 49.8, -800.0, 50.5],
            "Contract": ["Month-to-month", "One year", np.nan, "Two year"],
            "InternetService": ["DSL", np.nan, "Fiber optic", "DSL"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", np.nan],
            "Churn": ["Yes", "No", "No", "Yes"] # OUTLIER: -800.0 is abnormally low
            }

    df = pd.DataFrame(train_features_data)
    test_df = pd.DataFrame(test_features_data)
    # #act
    # result1 = preprocessing_pipeline(df)

    # check.is_instance(result1, pd.DataFrame, "Return result should be in dataframe")
    # check.equal(df.shape[0], result1.shape[0], "Input dataset and return dataset should have same rows or instances.")
    # check.equal(df.shape[1], result1.shape[1], "Both dataset should have same features")

    # # check missing values
    # cat_cols = df.select_dtypes(exclude="number").columns
    # check.equal((result1[cat_cols] == "missing").any().any(), True, "In dataframe there should be not have any missing values")

    # # check test for outliers
    # Q1 = df["feature_C"].quantile(0.25)
    # Q3 = df["feature_C"].quantile(0.75)

    # IQR = Q3 - Q1
    # lower_bound_C = Q1 - 1.5 * IQR
    # upper_bound_C = Q3 + 1.5 * IQR
    # print(f"Result feature C: \n{result1["feature_C"]}")
    # print(f"Lower bound of result feature_C: {lower_bound_C}")
    # print(f"Upper bound of result feature C: {upper_bound_C}")

    # check.equal((result1["feature_C"] < lower_bound_C).any(), False, "In feature there shouldn't be a lower value from lower bound.")
    # check.equal((result1["feature_C"] > upper_bound_C).any(), False, "In feature there shouldn't be upper value from upper bound.")

    
    # result2, test_result2 = preprocessing_pipeline(df=df, test_dataset=test_df)

    # check.is_instance(result2, pd.DataFrame, "Return result should be a DataFrame.")
    # check.is_instance(test_result2, pd.DataFrame, "Return test result should be a DataFrame")

    # # check missing value in test result
    # cat_cols = test_df.select_dtypes(exclude="number").columns

    # check.equal((test_result2[cat_cols] == "missing").any().any(), True, "There shouldn't be a missing valueu in dataset.")

    # # check outlier in test result
    # Q1 = test_df["feature_C"].quantile(0.25)
    # Q3 = test_df["feature_C"].quantile(0.75)

    # IQR = Q3 - Q1
    # lower_bound_test = Q1 - 1.5 * IQR
    # upper_bound_test = Q3 + 1.5 * IQR

    # print(test_result2["feature_C"])
    # print(f"Lower bound of test feature_C: {lower_bound_test}")
    # print(f"Upper bound of test feature_C: {upper_bound_test}")

    # check.equal((test_result2["feature_C"] < lower_bound_test).any(), False, "In testing there should be a lower outliers from lower bound..")
    # check.equal((test_result2["feature_C"] > upper_bound_test).any(), True, "In testing there should be a upper value from upper_bound.")

    # result3, test_result3 = preprocessing_pipeline(df=df, test_dataset=test_df, drop_features=["feature_C"])

    # # check shape for both result and test result
    # check.greater(df.shape[1], result3.shape[1], "There should be less feature in results")
    # check.greater(test_df.shape[1], test_result3.shape[1], "There should be less feature in test results")
    # check.equal(result3.shape[1], test_result3.shape[1], "Both result and test result should have same features.")
    
    #act
    result_train, result_val, result_test, result_train_label, result_val_label, result_test_label = preprocessing_pipeline(df=df, target_feature="Churn")
    
    #assert

    check.is_instance(result_train, np.ndarray, "Return result should be in numpy array.")
    check.equal(result_train.shape[0], result_train_label.shape[0], "Training data and label should have same instances.")
    numeric_checker = False
    if result_train.dtype.kind in 'biufc':
        numeric_checker = True

    check.equal(numeric_checker, True, "Return numpy array should have numerical values.")

    numeric_checker_for_val = False
    if result_val.dtype.kind in "biufc":
        numeric_checker_for_val = True

    check.equal(numeric_checker_for_val, True, "Return Validation set should have numeric data.")

# Testing for save_preprocessed_data():
def test_save_preprocessed_data():
    #Arrange

    train_features_data = {
            "feature_A": [10.2, 11.1, 10.5, 10.8, 11.0, 10.5],  # Normal data (around 10-11)
            "feature_B": [20.5, 19.8, 21.0, 20.2, 19.9, 22.0],  # Normal data (around 20)
            "feature_C": [10.1, 10.5, 950.0, 10.3, 10.9, 11.1], # OUTLIER: 950.0 is abnormally high
            "feature_D": [15.2, 14.8, 15.1, 15.5, 14.9, 15.6],  # Normal data (around 15)
            "feature_E": [50.1, 49.8, -800.0, 50.5, 50.2, 50.4],
            "Contract": ["Month-to-month", "One year", np.nan, "Two year", "Month-to-month", np.nan],
            "InternetService": ["DSL", np.nan, "Fiber optic", "DSL", np.nan, "Fiber optic"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", np.nan, "Credit card", "Electronic check"],
            "Churn": ["Yes", "No", "No", "Yes", "No", "Yes"] # OUTLIER: -800.0 is abnormally low
            }
    test_features_data = {
                "feature_A": [10.2, 11.1, 10.5, 10.8],  # Normal data (around 10-11)
                "feature_B": [20.5, 19.8, 21.0, 20.2],  # Normal data (around 20)
                "feature_C": [10.1, 10.5, 950.0, 10.3], # OUTLIER: 950.0 is abnormally high
                "feature_D": [15.2, 14.8, 15.1, 15.5],  # Normal data (around 15)
                "feature_E": [50.1, 49.8, -800.0, 50.5],
                "Contract": ["Month-to-month", "One year", np.nan, "Two year"],
                "InternetService": ["DSL", np.nan, "Fiber optic", "DSL"],
                "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", np.nan],
                "Churn": ["Yes", "No", "No", "Yes"] # OUTLIER: -800.0 is abnormally low
                }

    train_df = pd.DataFrame(train_features_data)
    test_df = pd.DataFrame(test_features_data)

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    # Split train into Validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    #act
    save_preprocessed_data(config["paths"]["preprocessed_data_dir"], X_train, X_test, y_train, y_test, X_val=X_val, y_val=y_val)

    #assert
    target_file = Path(config["paths"]["X_train_preprocessed"])
    with check:
        assert target_file.is_file(), f"Expected file {config["paths"]["X_train_preprocessed"]} to exist"