LLM-Powered Financial Document Summarization Tool
This project is a complete, end-to-end application that uses a fine-tuned BART large language model to summarize long financial and legal documents. It features a scalable backend API and an interactive web-based interface built with Streamlit.

Features
Fine-Tuned Summarization Model
Utilizes a pre-trained BART model fine-tuned on the BillSum dataset to generate high-quality, abstractive summaries.

Scalable Backend
A Flask-based REST API serves the model, decoupling the machine learning logic from the user interface.

Interactive Frontend
A user-friendly Streamlit application allows users to paste document text and receive a summary in real time.

Tunable Parameters
The UI includes controls for adjusting key model parameters like maximum summary length and beam search width.

Modular Pipeline
Clear separation of concerns across data preparation, model training, backend API, and frontend UI.

Project Structure
bash
Copy
Edit
financial-summarizer/
├── financial_summarizer_model/   # Saved fine-tuned model files
├── app.py                        # Flask backend API
├── streamlit_app.py              # Streamlit frontend UI
├── fine_tune.py                  # Script for model fine-tuning
├── data_prep.py                  # Script for data downloading & processing
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
How to Run This Project
1. Set Up the Environment
Create a Python virtual environment and install the required dependencies:

bash
Copy
Edit
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
2. Run Data Preparation (Optional)
This step was used to inspect the data. The fine-tuning script automatically handles data loading, so this is not necessary unless modifying the data pipeline.

bash
Copy
Edit
python data_prep.py
3. Run Model Fine-Tuning (Optional)
The fine-tuned model is already included in the financial_summarizer_model/ directory. Run this script only if you want to retrain the model from scratch.

Note: This is computationally expensive and should be done on a GPU.

bash
Copy
Edit
python fine_tune.py
4. Start the Backend API
Launch the Flask server that loads the fine-tuned model and serves prediction requests.

bash
Copy
Edit
python app.py
Keep this terminal running.

5. Start the Frontend UI
In a new terminal window, launch the Streamlit app:

bash
Copy
Edit
streamlit run streamlit_app.py
Model Details
Base model: facebook/bart-large

Fine-tuning dataset: BillSum

Frameworks used: Hugging Face Transformers, PyTorch, Flask, Streamlit

License
This project is licensed under the MIT License.
