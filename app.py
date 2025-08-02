# app.py
import os
from flask import Flask, request, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# --- Configuration ---
# Path to the fine-tuned model directory
MODEL_DIR = "./financial_summarizer_model"
# Check if a GPU is available and set the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Load Model and Tokenizer ---
# This is done once when the app starts to avoid reloading on every request.
print("Loading model and tokenizer...")
try:
    tokenizer = BartTokenizer.from_pretrained(MODEL_DIR)
    model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.to(DEVICE) # Move the model to the appropriate device (GPU or CPU)
    model.eval() # Set the model to evaluation mode
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None

# --- Helper Function for Summarization ---
# *** MODIFIED: Function now accepts generation parameters ***
def summarize_text(text, max_len=150, num_beams=4):
    """
    Takes a string of text and returns a generated summary.
    
    Args:
        text (str): The input document text.
        max_len (int): The maximum length of the generated summary.
        num_beams (int): The number of beams to use for beam search.
    """
    if not model or not tokenizer:
        return "Model is not available."

    # Tokenize the input text
    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt" # Return PyTorch tensors
    ).to(DEVICE) # Move tensors to the same device as the model

    # Generate the summary
    with torch.no_grad(): # Disable gradient calculation for inference
        summary_ids = model.generate(
            inputs['input_ids'],
            # *** Use the parameters passed from the request ***
            num_beams=num_beams,
            max_length=max_len,
            early_stopping=True
        )

    # Decode the summary and clean up the text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# --- API Endpoint ---
@app.route('/summarize', methods=['POST'])
def summarize_endpoint():
    """
    Flask endpoint to handle summarization requests.
    Expects a JSON payload with a "text" key and optional "max_length" and "num_beams".
    e.g., {"text": "...", "max_length": 200, "num_beams": 5}
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if 'text' not in data or not data['text']:
        return jsonify({"error": "Missing 'text' field in request"}), 400

    input_text = data['text']
    # *** MODIFIED: Get parameters from request, with defaults ***
    max_len = data.get('max_length', 150)
    num_beams = data.get('num_beams', 4)
    
    print(f"Received request with text length: {len(input_text)}, max_length: {max_len}, num_beams: {num_beams}")

    # Generate the summary
    summary = summarize_text(input_text, max_len=max_len, num_beams=num_beams)
    
    print(f"Generated summary: {summary}")

    return jsonify({"summary": summary})

# --- Main Execution ---
if __name__ == '__main__':
    # Note: Using app.run() is for development only.
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    app.run(host='0.0.0.0', port=5000, debug=True)
