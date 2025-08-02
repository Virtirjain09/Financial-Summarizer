# streamlit_app.py
import streamlit as st
import requests
import json
import os
import re

# --- Configuration ---
# The URL of the Flask backend API.
API_URL = "http://127.0.0.1:5000/summarize"
MODEL_DIR = "./financial_summarizer_model"

# --- Helper Function to Load Metrics ---
@st.cache_data
def load_metrics():
    """
    Finds the trainer_state.json file and loads the metrics from the
    log history. This version is more robust.
    """
    try:
        # Find the trainer_state.json file by searching the model directory
        metrics_file = None
        for root, dirs, files in os.walk(MODEL_DIR):
            if "trainer_state.json" in files:
                metrics_file = os.path.join(root, "trainer_state.json")
                break # Stop searching once found
        
        if not metrics_file:
            print("DEBUG: trainer_state.json not found anywhere in the model directory.")
            return None

        print(f"DEBUG: Found metrics file at: {metrics_file}")
        
        with open(metrics_file, "r") as f:
            data = json.load(f)
        
        log_history = data.get("log_history")
        if not log_history:
            print("DEBUG: log_history is empty or not found in the JSON file.")
            return None
            
        # Search for the evaluation metrics entry in the entire log history
        for entry in reversed(log_history):
            # Check for the specific keys that appear in your console log
            if "eval_rouge1" in entry and "eval_rouge2" in entry and "eval_rougeL" in entry:
                print("DEBUG: Found eval metrics in a log entry.")
                return {
                    "ROUGE-1": entry.get("eval_rouge1", 0),
                    "ROUGE-2": entry.get("eval_rouge2", 0),
                    "ROUGE-L": entry.get("eval_rougeL", 0),
                }
        
        # If the loop finishes without finding the metrics, print debug info
        print("DEBUG: Could not find an entry with 'eval_rouge1', 'eval_rouge2', and 'eval_rougeL' in the log_history.")
        if log_history:
            print("DEBUG: Keys in the last log_history entry were:", log_history[-1].keys())
        return None

    except Exception as e:
        print(f"An error occurred while loading metrics: {e}")
        return None

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Financial Document Summarizer",
    page_icon="�",
    layout="wide"
)

# --- Sidebar for Parameters ---
with st.sidebar:
    st.header(" Model Parameters")
    st.markdown("Adjust the generation parameters to control the output.")
    
    max_len = st.slider("Maximum Summary Length", min_value=50, max_value=300, value=150, step=10)
    num_beams = st.slider("Beam Search Width", min_value=1, max_value=10, value=4, step=1)
    st.info("A higher 'Beam Search Width' can lead to better quality summaries but takes longer to generate.")
    
    st.markdown("---")
    st.header(" Model Performance")
    metrics = load_metrics()
    if metrics:
        st.markdown("ROUGE scores from the last evaluation run:")
        st.metric(label="ROUGE-1 (Recall)", value=f"{metrics['ROUGE-1']:.2f}")
        st.metric(label="ROUGE-2 (Fluency)", value=f"{metrics['ROUGE-2']:.2f}")
        st.metric(label="ROUGE-L (Structure)", value=f"{metrics['ROUGE-L']:.2f}")
    else:
        st.warning("Could not load model metrics. Check terminal for DEBUG messages.")


# --- Main Page UI ---
st.title(" LLM-Powered Financial Document Summarizer")
st.markdown("""
Welcome to the Financial Document Summarization Tool! This application uses a fine-tuned BART model to generate concise summaries of long financial or legal texts. 
Paste your document into the text area below and use the sliders in the sidebar to control the summary output.
""")

st.markdown("---")

# Text area for user input
st.header("Enter Document Text")
input_text = st.text_area(
    "Paste the full text of the document here.",
    height=300,
    placeholder="""Example: Shields a business entity from civil liability relating to any injury or death occurring at a facility of that entity..."""
)

# Button to trigger summarization
if st.button("Summarize", type="primary"):
    if input_text and input_text.strip():
        with st.spinner("Generating summary... This may take a moment."):
            try:
                # Prepare the data for the POST request, including the new parameters
                payload = {
                    "text": input_text,
                    "max_length": max_len,
                    "num_beams": num_beams
                }
                headers = {"Content-Type": "application/json"}
                
                response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
                
                if response.status_code == 200:
                    summary = response.json().get("summary", "No summary returned.")
                    st.subheader("Generated Summary")
                    st.success(summary)
                else:
                    error_message = response.json().get("error", "An unknown error occurred.")
                    st.error(f"Error from server: {error_message} (Status code: {response.status_code})")

            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Could not connect to the backend API. Is the Flask app (app.py) running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter some text to summarize.")

st.markdown("---")
st.markdown("Developed by fine-tuning a `facebook/bart-large-cnn` model.")
