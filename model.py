from transformers import pipeline

def load_model():
    """Load pre-trained sentiment analysis model"""
    try:
        sentiment_model = pipeline("sentiment-analysis")
        return sentiment_model
    except Exception as e:
        raise Exception(f"Failed to load sentiment analysis model: {str(e)}")

def analyze_sentiment(model, text):
    """Analyze sentiment of given text"""
    try:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        result = model(text)
        if not result or len(result) == 0:
            raise ValueError("Model returned no results")
        
        label = result[0]['label']
        score = round(result[0]['score'], 3)
        return label, score
    except Exception as e:
        raise Exception(f"Error during sentiment analysis: {str(e)}")
