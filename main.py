import streamlit as st
from preprocess import clean_text
from model import load_model, analyze_sentiment

st.set_page_config(page_title="Mental Health Sentiment Analyzer", page_icon="🧠")

st.title("🧠 Mental Health Sentiment Analyzer")
st.write("Analyze social media posts to detect stress, sadness, or happiness levels.")

# Load model with caching to avoid reloading on every interaction
@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = get_model()

if model is None:
    st.error("Failed to load sentiment analysis model. Please check your internet connection and try again.")
    st.stop()

user_input = st.text_area("✍️ Enter a post or message:")

if st.button("Analyze"):
    if user_input.strip() != "":
        try:
            cleaned = clean_text(user_input)
            
            # Check if cleaned text is not empty
            if not cleaned.strip():
                st.warning("⚠️ The text couldn't be processed. Please enter a message with meaningful words.")
            else:
                label, score = analyze_sentiment(model, cleaned)
                st.subheader(f"Detected Emotion: **{label}** ({score*100:.1f}% confidence)")

                # Simple visualization (example for one text)
                st.write("### Emotion Indicator")
                st.progress(int(score * 100))
        except Exception as e:
            st.error(f"Error analyzing sentiment: {str(e)}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

st.markdown("---")
st.caption("⚠ For educational use only. Do not use this tool for real medical decisions.")
