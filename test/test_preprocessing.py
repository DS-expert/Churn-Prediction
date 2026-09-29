import pytest
from src.data.preprocessing import handle_missing_values, drop_feature
import pandas as pd
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