import streamlit as st

st.set_page_config(
    page_title="FAST: Feedback and Analysis for Student Thinking",
    page_icon="⚡",
    initial_sidebar_state="collapsed",
)

pages = [
    st.Page("src/fast/assessment_feedback.py", title="Assessment Feedback", icon="💬"),
    st.Page("src/fast/transcribe_images.py", title="Transcribe Images", icon="✍️"),
]

entry_page = st.navigation(pages, position="sidebar", expanded=True)
st.title("⚡ FAST")
st.subheader("Feedback and Analysis for Student Thinking")
st.write(
    """
Welcome to FAST!

This is a collection of tools to help analyze student work and provide feedback.
"""
)

entry_page.run()
