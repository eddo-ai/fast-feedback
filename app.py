import logging
import streamlit as st

logger = logging.getLogger("streamlit")

CLIENT_NAME = st.secrets.get("CLIENT_NAME", "Eddo")

st.set_page_config(
    page_title=f"🦔 {CLIENT_NAME} AI Tools by Eddo Learning",
    page_icon="🦔",
    layout="wide",
)

pages = [
    st.Page("src/fast/home.py", title="Home", icon="🏠"),
]

authenticated_pages = [
    st.Page("src/fast/green_anole.py", title="Green Anole Assessment", icon="🦎"),
    st.Page("src/fast/assessment_feedback.py", title="Assessment Feedback", icon="💬"),
    st.Page("src/fast/transcribe_images.py", title="Transcribe Images", icon="✍️"),
]


if st.experimental_user.get("is_logged_in"):
    user = st.experimental_user  # pages_list.append(profile_page)
    pages.extend(authenticated_pages)
    with st.sidebar:
        cols = st.columns([1, 3])
        with cols[0]:
            if user.get("picture"):
                st.image(str(user.get("picture")))
        with cols[1]:
            st.write(str(user.get("name", "")))
            email_display = (
                f"{user.get('email')} ✓"
                if user.get("email_verified")
                else "Email not verified."
            )
            st.write(email_display)
        if st.button("Logout"):
            st.logout()
        if logger.getEffectiveLevel() <= logging.DEBUG:
            st.write(user)



entry_page = st.navigation(pages, position="sidebar", expanded=True)
entry_page.run()
