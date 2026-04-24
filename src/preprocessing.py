import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    return df


def encode_data(df):
    le_gender = LabelEncoder()
    le_geo = LabelEncoder()

    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Geography'] = le_geo.fit_transform(df['Geography'])

    return df, le_gender, le_geo


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler