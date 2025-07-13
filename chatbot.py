from itertools import zip_longest
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Load OpenRouter API key from secrets
openapi_key = st.secrets["OPENROUTER_API_KEY"]

# Streamlit page configuration
st.set_page_config(page_title="AI ChatBot")
st.title("AI Mentor")

# Initialize session state
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""

# Initialize OpenRouter-compatible ChatOpenAI model
chat = ChatOpenAI(
    temperature=0.5,
    model_name="mistralai/mixtral-8x7b-instruct",
    openai_api_key=openapi_key,
    openai_api_base="https://openrouter.ai/api/v1",
    max_tokens=100
)

def build_message_list():
    # Start with system message defining behavior
    messages = [
        SystemMessage(
            content="""your name is AI Mentor. You are an AI Technical Expert for Artificial Intelligence, here to guide and assist students with their AI-related questions and concerns. Please provide accurate and helpful information, and always maintain a polite and professional tone.

1. Greet the user politely, ask their name, and ask how you can assist them with AI-related queries.
2. Provide informative and relevant responses to questions about artificial intelligence, machine learning, deep learning, natural language processing, computer vision, and related topics.
3. Avoid discussing sensitive, offensive, or harmful content. Refrain from engaging in any form of discrimination, harassment, or inappropriate behavior.
4. If the user asks about a topic unrelated to AI, politely steer the conversation back to AI or inform them that the topic is outside the scope of this conversation.
5. Be patient and considerate when responding to user queries, and provide clear explanations.
6. If the user expresses gratitude or indicates the end of the conversation, respond with a polite farewell.
7. Do not generate long paragraphs in response. Maximum words should be 100.
8. You can fetch data from Wikipedia.
Remember, your primary goal is to assist and educate students in the field of Artificial Intelligence. Always prioritize their learning experience and well-being."""
        )
    ]

    # Add past user-AI messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg:
            messages.append(HumanMessage(content=human_msg))
        if ai_msg:
            messages.append(AIMessage(content=ai_msg))

    return messages

def generate_response():
    messages = build_message_list()
    response = chat(messages)
    return response.content

def submit():
    st.session_state.entered_prompt = st.session_state.prompt_input
    st.session_state.prompt_input = ""

# Input box
st.text_input('YOU:', key='prompt_input', on_change=submit)

# On submit
if st.session_state.entered_prompt:
    user_input = st.session_state.entered_prompt
    st.session_state.past.append(user_input)
    output = generate_response()
    st.session_state.generated.append(output)

# Chat history
if st.session_state.generated:
    for i in range(len(st.session_state.generated) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
