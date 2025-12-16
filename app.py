import streamlit as st
from transformers import pipeline
from datetime import datetime

# ---------- CHAT MEMORY ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


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
    if emotion == "anxiety":
        return (
            "Thank you for telling me this. Anxiety can make everything feel overwhelming. "
            "Let’s slow down together. Try taking one gentle breath with me right now."
        )
    elif emotion == "sadness":
        return (
            "I’m really glad you shared this with me. Feeling low can be heavy, "
            "and it’s okay to not have all the answers right now."
        )
    elif emotion == "anger":
        return (
            "It sounds like something has been building up inside you. "
            "You’re allowed to feel this way. Want to talk about what triggered it?"
        )
    else:
        return (
            "I’m here and listening. You don’t have to filter your thoughts here. "
            "Say as much or as little as you want."
        )

def log_mood(emotion):
    with open("mood_log.txt", "a") as file:
        file.write(f"{datetime.now().date()} | {emotion}\n")

# ------------------ USER GOAL ------------------
st.markdown("### 🎯 Today’s Wellness Goal")
goal = st.text_input("What is one thing you’d like to feel better about today?")

# ------------------ CHAT ------------------
st.markdown("### 💬 Talk to MindEase")
st.caption("This is a safe, non-judgmental space. Take your time.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input box (real chat style)
user_input = st.chat_input("Type your message here...")

if user_input:
    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Emotion detection
    emotion = detect_emotion(user_input)
    log_mood(emotion)

    # More human-like responses
    response = get_supportive_response(emotion)

    # Store bot reply
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.markdown(response)
        st.caption("How are you feeling after sharing this?")
