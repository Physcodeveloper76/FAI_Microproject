import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources (only if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

def clean_text(text):
    """Clean and preprocess text data"""
    if not text:
        return ""
    
    try:
        text = re.sub(r'http\S+|@\S+|#\S+', '', text)  # remove links, mentions, hashtags
        text = re.sub(r'[^A-Za-z\s]', '', text.lower())  # keep only letters
        words = text.split()
        
        if not words:
            return ""
        
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 0]
        return ' '.join(cleaned_words)
    except Exception as e:
        # Return original text if preprocessing fails
        return text
