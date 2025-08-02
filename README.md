# LLM-Powered Financial Document Summarization Tool
This project is a complete, end-to-end application that uses a fine-tuned BART Large language model to summarize long financial and legal documents. It features a scalable backend API and an interactive web-based user interface built with Streamlit.

# Features
* Fine-Tuned Summarization Model: Utilizes a pre-trained BART model fine-tuned on the billsum dataset to generate high-quality, abstractive summaries.

* Scalable Backend: A Flask-based REST API serves the model, decoupling the machine learning logic from the user interface.

* Interactive Frontend: A user-friendly Streamlit application allows users to paste document text and receive a summary in real-time.

* Tunable Parameters: The UI includes controls for adjusting key model parameters like maximum summary length and beam search width, allowing users to customize the output.

* Modular Pipeline: The project is broken into clear, logical scripts for data preparation, model training, backend logic, and frontend UI.

#  Project Structure
*  financial-summarizer/
* ├── financial_summarizer_model/   <-- Saved fine-tuned model files
* ├── app.py                        <-- Flask backend API
* ├── streamlit_app.py              <-- Streamlit frontend UI
* ├── fine_tune.py                  <-- Script for model fine-tuning
* ├── data_prep.py                  <-- Script for data downloading & processing
* ├── requirements.txt              <-- Python dependencies
* └── README.md                     <-- This file

# How to Run This Project
* 1. Setup the Environment
First, create a Python environment and install the required dependencies.
*  Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
* Install all required packages
pip install -r requirements.txt

*  2. Run the Data Preparation (Optional)
  This step was used to inspect the data. You do not need to run it again as the fine-tuning script handles data loading automatically.
python data_prep.py

*  3. Run the Model Fine-Tuning (Optional)
The fine-tuned model is already included in the financial_summarizer_model directory. You only need to run this script if you want to retrain the model from scratch.
Warning: This is computationally expensive and will take many hours on a CPU. It is recommended to run this on a GPU.
python fine_tune.py

*  4. Start the Backend API
This server loads the fine-tuned model and waits for requests from the frontend.

 Run this command in a terminal
python app.py

 Leave this terminal running.

* 5. Start the Frontend UI
This launches the interactive Streamlit web application.

Open a NEW terminal and run this command
streamlit run streamlit_app.py


