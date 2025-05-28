import streamlit as st

client_name = st.session_state.get("client_name", "Eddo")
st.title(f":sparkles::hedgehog: {client_name} AI Tools by Eddo Learning")
st.subheader("Feedback and Analysis for Student Thinking")

st.write(f"Welcome to the {client_name} AI Tools homepage!")

if st.experimental_user.get("is_logged_in"):
    st.write("Choose a tool from the sidebar to get started.")
else:
    st.write("Please sign in to use the tools.")
    if st.button("Sign In"):
        st.login()
