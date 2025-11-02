import matplotlib.pyplot as plt
import pandas as pd
import io

def plot_sentiment_distribution(sentiments):
    """Create a bar chart showing sentiment distribution"""
    df = pd.DataFrame(sentiments, columns=['Sentiment'])
    counts = df['Sentiment'].value_counts()

    plt.figure(figsize=(6,4))
    plt.bar(counts.index, counts.values)
    plt.title("Sentiment Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")

    # Save to buffer for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf
