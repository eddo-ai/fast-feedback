FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Create a temporary requirements file without the local package reference
RUN grep -v "file:///Users/aw/Developer/wauwatosa" requirements.txt > requirements_clean.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_clean.txt

# Copy the rest of the application
COPY . .

# Install the local package
RUN pip install -e .

EXPOSE 8502

CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
