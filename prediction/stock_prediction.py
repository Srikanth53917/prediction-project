import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def train_stock_model(data):
    data = data.copy()

    # Rename column if needed
    if '4. close' in data.columns:
        data = data.rename(columns={'4. close': 'Close'})

    data['Prediction'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    X = data[['Close']]
    y = data['Prediction']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/stock_model.pkl")

    return model   # ✅ FIXED


def predict_next_day(model, df):
    df = df.copy()

    # Rename column if needed
    if '4. close' in df.columns:
        df = df.rename(columns={'4. close': 'Close'})

    last_close = df['Close'].iloc[-1]

    input_df = pd.DataFrame([[last_close]], columns=['Close'])

    prediction = model.predict(input_df)[0]

    return float(prediction)   # ✅ always return number
