import logging
import streamlit as st

logger = logging.getLogger("streamlit")

client_config = st.secrets.get("client", {})
st.session_state.client_name = client_config.get("name", "Eddo")

st.set_page_config(
    page_title=f"🦔 {st.session_state.get('client_name', 'Eddo')} AI Tools by Eddo Learning",
    page_icon="🦔",
    layout="wide",
)

pages = [
    st.Page("src/fast/home.py", title="Home", icon="🏠"),
    st.Page("src/fast/green_anole_public.py", title="Green Anole Assessment", icon="🦎"),
    # st.Page("src/fast/transcribe_images.py", title="Transcribe Images", icon="✍️"),
    st.Page("src/fast/assessment_feedback.py", title="Assessment Feedback", icon="💬"),
]

authenticated_pages = [
    # st.Page("src/fast/green_anole.py", title="Green Anole Assessment", icon="🦎"),
    # st.Page("src/fast/transcribe_images.py", title="Transcribe Images", icon="✍️"),
]

admin_pages = [
    # st.Page("src/fast/db_viewer.py", title="DB Viewer", icon="📊"),
]

if st.experimental_user.get("is_logged_in"):
    user = st.experimental_user  # pages_list.append(profile_page)
    pages.extend(authenticated_pages)
    allowed_domains = st.secrets.get("allowed_domains", "").split(",")
    if any(domain in user.get("email", "") for domain in allowed_domains):
        pages.extend(admin_pages)
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
