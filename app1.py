import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from textblob import TextBlob  # For basic sentiment analysis

# Load environment variables
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

# Streamlit page setup
st.set_page_config(page_title="MindMate - Student Wellness Chatbot")
st.title("🧠 MindMate")
st.subheader("A supportive companion for student mental wellness 💬")

# Initialize OpenAI client with NVIDIA API
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Initial system prompt (context for the bot)
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are MindMate, an empathetic mental wellness assistant for university students. "
        "Your goal is to support students dealing with stress, anxiety, or academic pressure. "
        "Offer encouraging, calm, and helpful responses. If the user's sentiment seems very negative, "
        "suggest seeking support from a trusted person or a counselor. Keep replies simple, respectful, "
        "and supportive without sounding clinical or robotic."
    )
}

# Sentiment interpretation function
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity < -0.4:
        return "negative"
    elif polarity > 0.3:
        return "positive"
    else:
        return "neutral"

# Chat completion function with sentiment-aware response
def get_response(conversation_history, sentiment):
    # Adjust prompt slightly based on sentiment
    sentiment_msg = {
        "role": "system",
        "content": f"The current sentiment of the user is '{sentiment}'. Respond with appropriate empathy."
    }

    messages = [SYSTEM_PROMPT, sentiment_msg] + conversation_history

    completion = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=messages,
        temperature=0.3,
        top_p=0.7,
        max_tokens=1024,
        stream=False
    )
    return completion.choices[0].message.content

# Session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# User input text box
user_input = st.text_input("You: ", key="input", placeholder="What's on your mind today?")

# Chat processing
if user_input:
    sentiment = analyze_sentiment(user_input)

    # Add user's message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Generate bot response
    bot_reply = get_response(st.session_state.chat_history, sentiment)

    # Add bot reply to history
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

# Display chat history
st.subheader("🗨 Chat History")
for msg in st.session_state.chat_history:
    speaker = "You" if msg["role"] == "user" else "MindMate"
    st.markdown(f"{speaker}:** {msg['content']}")
    st.markdown("---")