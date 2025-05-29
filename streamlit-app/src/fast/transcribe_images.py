from langchain import hub
from langchain_openai import ChatOpenAI
import streamlit as st
import asyncio
import base64
import pandas as pd
from datetime import datetime
import os


st.title("Transcribe Images")

st.write("""
This tool transcribes images of hand-written notes into text.
""")



def convert_image_to_base64(uploaded_file):
    # Read the BytesIO buffer
    bytes_data = uploaded_file.getvalue()

    # Get the mime type from the uploaded file
    file_type = uploaded_file.type

    # Encode to base64
    base64_str = base64.b64encode(bytes_data).decode()

    # Create the base64 string with mime type
    return f"data:{file_type};base64,{base64_str}"

def clean_content(text):
    """Clean text content by removing newlines and extra whitespace"""
    if not text:
        return ""
    # Replace newlines with spaces and remove extra whitespace
    return " ".join(text.replace("\n", " ").split())

def save_and_display_results(results):
    """Save results to CSV and display them in a table"""
    if not results:
        return

    # Create data directory if it doesn't exist
    data_dir = "data/transcriptions"
    os.makedirs(data_dir, exist_ok=True)
    csv_path = f"{data_dir}/transcriptions.csv"

    # Process all results
    rows = []
    for filename, result in results:
        if not result.get("is_orientation_upright"):
            st.error(f"Image {filename} is not upright. Please rotate it and try again.")
            continue

        for response in result.get("responses", []):
            row = {
                "timestamp": datetime.now(),
                "filename": filename,
                "prompt": clean_content(response.get("prompt")),
                "content": clean_content(response.get("content"))
            }
            rows.append(row)

    if rows:
        # Save to CSV
        df = pd.DataFrame(rows)
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

        # Display results
        st.subheader("Transcription Results")
        st.dataframe(
            df.drop('timestamp', axis=1),  # Don't show timestamp in display
            column_config={
                "filename": st.column_config.TextColumn("Filename", width="medium"),
                "prompt": st.column_config.TextColumn("Prompt", width="medium"),
                "content": st.column_config.TextColumn("Content", width="large"),
            },
            hide_index=True,
        )
    return results



async def transcribe_images(uploaded_files):
    if len(uploaded_files) == 0:
        st.error("No images uploaded. Please upload some images first.")
        return []
        
    MODEL_NAME = st.secrets.get("OPENAI_MODEL", "gpt-4o")
    prompt = hub.pull("transcribe_student_work")
    model = ChatOpenAI(model=MODEL_NAME)
    chain = prompt | model

    # Create list of (filename, base64_image) tuples
    image_data = [(file.name, convert_image_to_base64(file)) for file in uploaded_files]

    # Invoke in async batch with base64 images
    results = await chain.abatch([img for _, img in image_data])

    # Combine results with filenames and process
    results_enriched = list(zip([name for name, _ in image_data], results))
    return save_and_display_results(results_enriched)



# Transcribe the images
with st.form(key="upload_images", clear_on_submit=True):
    uploaded_files = st.file_uploader(
        "Upload images", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )
    if st.form_submit_button("Transcribe"):
        with st.status("Transcribing images...") as status:
            try:
                results = asyncio.run(transcribe_images(uploaded_files))
                if results:
                    status.update(label="Transcription complete!", state="complete")
                else:
                    status.update(label="No images were processed", state="error")
            except Exception as e:
                st.error(f"An error occurred during transcription: {str(e)}")
                status.update(label="Transcription failed", state="error")
            