from flask import Flask, render_template, request
import requests
import pandas as pd
import traceback
import time
import os

# Use env variable if available, else fallback
API_KEY = os.getenv("API_KEY", "JTLZVD07EXJDLYWW")

from prediction.stock_prediction import train_stock_model, predict_next_day
from sentiment.sentiment_analysis import analyze_sentiment_from_dataset
from visualization.charts import stock_price_chart, sentiment_chart

app = Flask(__name__)

def get_stock_data(ticker):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={API_KEY}"
    
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" not in data:
        return None

    prices = []
    for date in data["Time Series (Daily)"]:
        prices.append(float(data["Time Series (Daily)"][date]["4. close"]))

    prices.reverse()
    return prices

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        try:
            ticker = request.form.get("ticker").upper()

            # API limit control
            time.sleep(2)

            prices = get_stock_data(ticker)

            if prices is None or len(prices) == 0:
                result = {"error": "Invalid ticker or API limit reached"}
                return render_template("index.html", result=result)

            df = pd.DataFrame(prices, columns=["Close"])

            last_price = df["Close"].iloc[-1]

            # Prediction
            try:
                model = train_stock_model(df)
                predicted_price = predict_next_day(model, df)
            except:
                predicted_price = last_price

            # Sentiment
            try:
                sentiment_score = analyze_sentiment_from_dataset(ticker)
            except:
                sentiment_score = 0

            if sentiment_score > 0:
                sentiment = "Positive"
                recommendation = "Buy"
            elif sentiment_score < 0:
                sentiment = "Negative"
                recommendation = "Sell"
            else:
                sentiment = "Neutral"
                recommendation = "Hold"

            # Charts
            try:
                stock_chart = stock_price_chart(df, ticker)
                sentiment_img = sentiment_chart(sentiment_score)
            except:
                stock_chart = None
                sentiment_img = None

            result = {
                "company": ticker,
                "last_price": round(last_price, 2),
                "predicted_price": round(predicted_price, 2),
                "sentiment": sentiment,
                "recommendation": recommendation,
                "stock_chart": stock_chart,
                "sentiment_chart": sentiment_img
            }

        except Exception as e:
            print("ERROR:", e)
            traceback.print_exc()
            result = {"error": "Something went wrong"}

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)