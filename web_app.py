from flask import Flask, render_template, request
import pandas as pd
import traceback
import time
import requests
import os

# Import your project modules
from prediction.stock_prediction import train_stock_model, predict_next_day
from sentiment.sentiment_analysis import analyze_sentiment_from_dataset
from visualization.charts import stock_price_chart, sentiment_chart

app = Flask(__name__)

# 🔑 Your Alpha Vantage API key
API_KEY = os.getenv("API_KEY", "LE09ZONH4OKJAS2Q")


# 📊 Function to fetch stock data
def get_stock_data(symbol):
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()

        # Check API limit / error
        if "Time Series (Daily)" not in data:
            print("API Error:", data)
            return None

        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        return df

    except Exception as e:
        print("Stock Fetch Error:", e)
        return None


# 🌐 Main route
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        try:
            print("🔵 Request received")

            ticker = request.form["ticker"].upper()
            start_year = int(request.form["start_year"])
            end_year = int(request.form["end_year"])

            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"

            # 📥 Fetch data
            df = get_stock_data(ticker)

            if df is None or df.empty:
                raise Exception("Stock data not available")

            # ⏳ Prevent API rate limit
            time.sleep(12)

            # Filter data
            df = df.loc[start_date:end_date]

            if df.empty:
                raise Exception("No data in selected range")

            # 📌 Last price
            last_price = round(df["4. close"].iloc[-1], 2)

            # 🤖 Train + Predict
            model = train_stock_model(df)

            predicted_price = predict_next_day(model, df)

            if predicted_price is None:
                predicted_price = last_price

            predicted_price = round(predicted_price, 2)

            # 🧠 Sentiment
            sentiment_score = analyze_sentiment_from_dataset(start_year,end_year)

            if sentiment_score > 0:
                sentiment = "Positive"
                sentiment_class = "positive"
                decision = "Buy"
            elif sentiment_score < 0:
                sentiment = "Negative"
                sentiment_class = "negative"
                decision = "Sell"
            else:
                sentiment = "Neutral"
                sentiment_class = "neutral"
                decision = "Hold"

            # 📈 Charts
            stock_price_chart(df)
            sentiment_chart(sentiment_score)

            # 📦 Result
            result = {
                "company": ticker,
                "last_price": last_price,
                "predicted_price": predicted_price,
                "sentiment": sentiment,
                "sentiment_class": sentiment_class,
                "decision": decision
            }

            print("✅ Success:", result)

        except Exception as e:
            print("❌ ERROR:", traceback.format_exc())

            result = {
                "company": "Error",
                "last_price": "-",
                "predicted_price": "-",
                "sentiment": "Error",
                "sentiment_class": "negative",
                "decision": "Check logs"
            }

    return render_template("index.html", result=result)


# 🚀 Run app (for local testing)
if __name__ == "__main__":
    app.run(debug=True)
