# FAST: Feedback and Analysis for Student Thinking

By Eddo Learning in collaboration with Wauwatosa Public Schools

This project provides a collection of AI-powered tools designed to assist educators with analyzing student work and generating feedback based on defined rubrics and instructions.

## Features

- **Assessment Feedback:** Generates individual feedback for batches of student responses based on provided instructions and a rubric. It also summarizes common themes across the submissions.
- **Transcribe Images:** Transcribes text from images of student work, including handwriting, using OpenAI's multimodal API.

## Setup

This project uses `uv` for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd fast
    ```
2.  **Install dependencies:**
    ```bash
    uv sync
    ```
3.  **Set up environment variables:**
    Create a `.env` file in the project root directory by copying the `.env.example` file:
    ```bash
    cp .env.example .env
    ```
    Fill in the required API keys and endpoints in the `.env` file. See `.env.example` for the required variables.

## Running the Application

To run the Streamlit application locally:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

## Development

To install development dependencies (e.g., for live reloading):

```bash
uv sync --dev
```

## Deployment

A `dockerfile` is included for building a container image of the application, facilitating deployment.
