import streamlit as st

CLIENT_NAME = st.secrets.get("CLIENT_NAME", "Eddo")

st.title(f":sparkles::hedgehog: {CLIENT_NAME} AI Tools by Eddo Learning")
st.subheader("Feedback and Analysis for Student Thinking")

st.write(f"Welcome to the {CLIENT_NAME} AI Tools homepage!")

if st.experimental_user.get("is_logged_in"):
    st.write("Choose a tool from the sidebar to get started.")
else:
    st.write("Please sign in to use the tools.")
    if st.button("Sign In"):
        st.login()
