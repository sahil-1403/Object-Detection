#!/bin/bash
# Easy startup script for the Streamlit Object Detection App

# Activate virtual environment
source venv/bin/activate

# Run Streamlit app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Note: After the app starts, open your browser to:
# http://localhost:8501
