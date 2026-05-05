import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment_from_dataset(start_year, end_year):
    try:
        df = pd.read_csv("prediction/data/news_data.csv")

        # Ensure year column exists
        df['year'] = df['year'].astype(str).str[:4].astype(int)

        filtered_df = df[
            (df['year'] >= start_year) &
            (df['year'] <= end_year)
        ]

        print("Filtered rows:", len(filtered_df))

        if len(filtered_df) == 0:
            return 0.0   # neutral fallback

        total_score = 0

        for text in filtered_df['headline']:
            score = analyzer.polarity_scores(str(text))['compound']
            total_score += score

        avg_score = total_score / len(filtered_df)

        print("Average Sentiment:", avg_score)

        return float(avg_score)   # ✅ RETURN NUMBER ONLY

    except Exception as e:
        print("ERROR:", e)
        return 0.0   # safe fallback
