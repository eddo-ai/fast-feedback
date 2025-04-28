import streamlit as st

CLIENT_NAME = st.secrets.get("CLIENT_NAME", "Eddo")

st.set_page_config(
    page_title=f"🦔 {CLIENT_NAME} AI Tools by Eddo Learning",
    page_icon="🦔",
    layout="wide",
)

pages = [
    st.Page("src/fast/home.py", title="Home", icon="🏠"),
    st.Page("src/fast/green_anole.py", title="Green Anole Assessment", icon="🦎"),
    st.Page("src/fast/assessment_feedback.py", title="Assessment Feedback", icon="💬"),
    st.Page("src/fast/transcribe_images.py", title="Transcribe Images", icon="✍️"),
]

entry_page = st.navigation(pages, position="sidebar", expanded=True)
entry_page.run()
