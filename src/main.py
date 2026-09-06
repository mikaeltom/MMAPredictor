from AgentOrchestrator import AgentOrchestrator
import streamlit as st
import spacy
import time
from utils import retrieve_checked_schedule

spacy_model = spacy.load("en_core_web_sm")

st.title("MMA Predictor")
st.write("Welcome to MMA Predictor, a Multi-Agent System.")
st.markdown("**You can click on the left sidebar (>>) to open the ChatBot**")

if "agent" not in st.session_state:
    st.session_state.agent = AgentOrchestrator(spacy_model)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start predicting ! Give me two fighter names"}]

st.subheader("Upcoming Fights : ")

fights = retrieve_checked_schedule()

if not fights:
    st.info("No upcoming fights found.")
else:
    columns = st.columns(3)
    # code bellow is inspired by source : https://docs.streamlit.io/develop/api-reference/layout/st.container
    for fight in fights:
        f1, f2, date = fight["fighter1"], fight["fighter2"], fight["date"]
        with st.container(border=True):
            st.markdown(f"**{f1} vs {f2}**")
            if date:
                st.caption(date)
            if st.button("Predict", key=f"{f1}_vs_{f2}_{date}"):
                with st.spinner(f"Predicting {f1} vs {f2}..."):
                    result = st.session_state.agent.predict_winner(f1, f2)
                st.success(result)

with st.sidebar: # source : https://docs.streamlit.io/develop/api-reference/layout/st.sidebar
    st.subheader("MMA Predictor Bot")
    # code bellow is inspired by source : https://streamlit.io/playground?example=llm_chat
    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    if prompt := st.chat_input("What is up? Ask me to predict something."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container: # source : https://docs.streamlit.io/develop/api-reference/layout/st.container
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                with st.spinner("Processing fighters..."): # https://docs.streamlit.io/develop/api-reference/status/st.spinner
                    controller_response = st.session_state.agent.process_request(prompt, fights)

                if controller_response is None:
                    assistant_response = "Sorry, I encountered an error processing that request. Please try again."
                else:
                    assistant_response = controller_response

                full_response = ""
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})