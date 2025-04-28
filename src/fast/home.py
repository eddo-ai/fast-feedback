import streamlit as st

CLIENT_NAME = st.secrets.get("CLIENT_NAME", "Eddo")

st.title(f":sparkles::hedgehog: {CLIENT_NAME} AI Tools by Eddo Learning")
st.subheader("Feedback and Analysis for Student Thinking")

st.write(f"Welcome to the {CLIENT_NAME} AI Tools homepage!")
st.write("Select a tool from the sidebar to get started.")
