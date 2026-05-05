import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def stock_price_chart(data):
    plt.figure()

    # Fix column name issue
    if '4. close' in data.columns:
        data = data.rename(columns={'4. close': 'Close'})

    plt.plot(data['Close'])
    plt.title("Stock Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.savefig("static/stock_chart.png")
    plt.close()


def sentiment_chart(score):
    plt.figure()

    # Convert score → categories
    if score > 0:
        scores = {"positive": 1, "neutral": 0, "negative": 0}
    elif score < 0:
        scores = {"positive": 0, "neutral": 0, "negative": 1}
    else:
        scores = {"positive": 0, "neutral": 1, "negative": 0}

    labels = list(scores.keys())
    values = list(scores.values())

    plt.bar(labels, values)
    plt.title("Sentiment Analysis")

    plt.savefig("static/sentiment_chart.png")
    plt.close()
