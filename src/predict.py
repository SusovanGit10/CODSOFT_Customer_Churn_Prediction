import joblib
import pandas as pd

# Load saved objects
model = joblib.load("../models/model.pkl")
scaler = joblib.load("../models/scaler.pkl")
le_gender = joblib.load("../models/le_gender.pkl")
le_geo = joblib.load("../models/le_geo.pkl")

# IMPORTANT: keep same column order as training
FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]


def predict_churn(data_dict):
    """
    data_dict example:
    {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000
    }
    """

    # Convert to DataFrame
    df = pd.DataFrame([data_dict])

    # Encode categorical values
    df["Geography"] = le_geo.transform(df["Geography"])
    df["Gender"] = le_gender.transform(df["Gender"])

    # Ensure correct column order
    df = df[FEATURE_COLUMNS]

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "probability": round(probability, 3)
    }


# Example run
if __name__ == "__main__":
    sample = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 60000
    }

    result = predict_churn(sample)

    print("Prediction:", result["prediction"])
    print("Churn Probability:", result["probability"])