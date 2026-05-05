import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_stock_model(data):
    data = data.copy()
    data['Prediction'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    X = data[['Close']]        # 2D
    y = data['Prediction']     # 1D

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/stock_model.pkl")
    return model, X_test, y_test

def predict_next_day(model, last_close):
    input_df=pd.DataFrame([[last_close]],columns=['Close'])
    return model.predict(input_df)[0]
    