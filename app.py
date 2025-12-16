import streamlit as st
from transformers import pipeline
from datetime import datetime

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="MindEase – Mental Wellness",
    page_icon="🌱",
    layout="centered"
)

# ------------------ CUSTOM STYLE ------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.chat-box {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.bot {
    color: #2c7be5;
}
.user {
    color: #333333;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("🌱 MindEase")
st.subheader("Your AI companion for emotional support")
st.caption("Private • Supportive • Non-judgmental")

st.warning("⚠️ MindEase is not a replacement for professional mental health care.")

# ------------------ LOAD AI MODEL ------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_analyzer = load_model()

def detect_emotion(text):
    emotions = emotion_analyzer(text)[0]
    emotions.sort(key=lambda x: x['score'], reverse=True)
    return emotions[0]['label']

def get_supportive_response(emotion):
    responses = {
        "anxiety": "I’m really glad you shared this. Let’s slow things down together. Try breathing in for 4 seconds and out for 6.",
        "sadness": "I’m sorry you’re feeling this way. Your emotions are valid, and you don’t have to go through this alone.",
        "anger": "It sounds like something has been weighing on you. Taking a pause can sometimes bring clarity.",
        "fear": "Feeling afraid can be overwhelming. You’re safe here, and we can take this one step at a time."
    }
    return responses.get(emotion, "Thank you for opening up. I’m here with you.")

def log_mood(emotion):
    with open("mood_log.txt", "a") as file:
        file.write(f"{datetime.now().date()} | {emotion}\n")

# ------------------ USER GOAL ------------------
st.markdown("### 🎯 Today’s Wellness Goal")
goal = st.text_input("What is one thing you’d like to feel better about today?")

# ------------------ CHAT ------------------
st.markdown("### 💬 Talk to MindEase")
user_input = st.text_area("Share whatever is on your mind", height=120)

if st.button("Send"):
    if user_input.strip():
        emotion = detect_emotion(user_input)
        response = get_supportive_response(emotion)
        log_mood(emotion)

        st.markdown(f"<div class='chat-box user'><b>You:</b> {user_input}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-box bot'><b>MindEase:</b> {response}</div>", unsafe_allow_html=True)

        st.caption("How are you feeling compared to earlier today?")
