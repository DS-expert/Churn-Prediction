import pytest
from src.data.preprocessing import handle_missing_values, drop_feature, Encoder, scaling, preprocesser
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pytest_check as check
import numpy as np

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

    #act
    result1, result1_test = Encoder(X_train, X_test, all_categorical=True)
    check.equal(result1.shape[1], result1_test.shape[1], "Encoded train and test data should have the same number of columns.")
    check.is_true(isinstance(result1, pd.DataFrame), "Encoded return value should be a DataFrame.")
    check.is_true(isinstance(result1_test, pd.DataFrame), "Encoded return value should be a DataFrame")
    check.greater(result1.shape[1], X_train.shape[1], "Encoded data should have more columns than the original data due to one-hot encoding.")
    cat_features = X_train.select_dtypes(exclude='number')
    result_cat = result1.select_dtypes(exclude='number')
    check.greater(cat_features.shape[1], result_cat.shape[1], "Encoded categorical features should have more columns than the original category features")

    with pytest.raises(ValueError):
        Encoder(X_train, X_test, all_categorical=False)

    result3, result3_test = Encoder(X_train, X_test, all_categorical=False, features=["Contract", "InternetService"])
    check.equal(result3.shape[1], result3_test.shape[1], "Encoded train and test data should have the same number of columns.")
    check.is_true(isinstance(result3, pd.DataFrame), "Encoded return value should be a DataFrame.")
    check.is_true(isinstance(result3_test, pd.DataFrame), "Encoded return value should be a DataFrame")
    check.is_in("PaymentMethod", result3.columns, f"Encoded data should contain PaymentMethod columns, but it is missing")
    print(X_train["PaymentMethod"].dtype)
    print(result3["PaymentMethod"].dtype)
    check.equal(X_train["PaymentMethod"].dtype, result3["PaymentMethod"].dtype, "Encoded data should have the same data types as the original data")
    forbidden_cols = [col for col in result3.columns if col.startswith("PaymentMethod_")]
    check.equal(len(forbidden_cols), 0, "Encoded data should not contain extra columns which isn't provided.")
    pd.testing.assert_series_equal(result3["PaymentMethod"], X_train["PaymentMethod"])

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

def test_preprocesser():
    #arrange
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

    #act
    X_train, X_test, y_train, y_test = preprocesser(df, target="Churn")

    #assert
    check.is_instance(X_train, np.ndarray, "Preprocessed return value should be a numpy")
    check.is_instance(y_train, pd.Series, "testing label should be pd.Series")
    check.greater(X_train.shape[1], df.shape[1], "Preprocessed data should have more features than original data")
    check.equal(pd.isnull(X_train).any(), False, "Preprocessed data should not have any missing values") # type: ignore
    check.equal(X_train.shape[1], X_test.shape[1], "Preprocessed train and test data should have the same number of features")