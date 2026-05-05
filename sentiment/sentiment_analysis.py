import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_from_dataset(start_year, end_year):
    try:
        df = pd.read_csv("data/news_data.csv")

        # 🔥 SIMPLE YEAR EXTRACTION (NO datetime issues)
        df['year'] = df['year'].astype(str).str[:4].astype(int)

        # ✅ FILTER USING YEAR COLUMN
        filtered_df = df[
            (df['year'] >= start_year) &
            (df['year'] <= end_year)
        ]

        print("Filtered rows:", len(filtered_df))

        scores = {"positive": 0, "negative": 0, "neutral": 0}

        for text in filtered_df['headline']:
            score = analyzer.polarity_scores(str(text))['compound']

            if score > 0.05:
                scores["positive"] += 1
            elif score < -0.05:
                scores["negative"] += 1
            else:
                scores["neutral"] += 1

        print("Scores:", scores)

        # ❗ Only fallback if NO data
        if len(filtered_df) == 0:
            print("⚠️ No data found")
            scores = {"positive": 1, "negative": 1, "neutral": 1}

        # FINAL SENTIMENT
        if scores["negative"] > scores["positive"]:
            sentiment = "NEGATIVE"
        elif scores["positive"] > scores["negative"]:
            sentiment = "POSITIVE"
        else:
            sentiment = "NEUTRAL"

        return sentiment, scores

    except Exception as e:
        print("ERROR:", e)
        return "NEUTRAL", {"positive": 1, "negative": 1, "neutral": 1}