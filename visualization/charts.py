import matplotlib
matplotlib.use('Agg')   # Fix server issue

import matplotlib.pyplot as plt


def stock_price_chart(data):
    plt.figure()
    plt.plot(data['Close'])
    plt.title("Stock Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.savefig("static/stock_chart.png")

    # OPTIONAL popup (comment if not needed)
    # plt.show()

    plt.close()


def sentiment_chart(scores):
    plt.figure()

    # ✅ Fix: handle empty or wrong data
    if not isinstance(scores, dict) or sum(scores.values()) == 0:
        print("⚠️ Using fallback sentiment data")
        scores = {"positive": 1, "negative": 1, "neutral": 1}

    labels = list(scores.keys())
    values = list(scores.values())

    plt.bar(labels, values)
    plt.title("Sentiment Analysis")

    plt.savefig("static/sentiment_chart.png")

    # OPTIONAL popup
    # plt.show()

    plt.close()