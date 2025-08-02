# LLM-Powered Financial Document Summarization Tool

This project is a complete, end-to-end application that uses a fine-tuned BART large language model to summarize long financial and legal documents. It features a scalable backend API and an interactive web-based interface built with Streamlit.

## Features

- **Fine-Tuned Summarization Model**  
  Utilizes a pre-trained BART model fine-tuned on the [BillSum](https://huggingface.co/datasets/billsum) dataset to generate high-quality, abstractive summaries.

- **Scalable Backend**  
  A Flask-based REST API serves the model, decoupling the machine learning logic from the user interface.

- **Interactive Frontend**  
  A user-friendly Streamlit application allows users to paste document text and receive a summary in real time.

- **Tunable Parameters**  
  The UI includes controls for adjusting key model parameters like maximum summary length and beam search width.

- **Modular Pipeline**  
  Clear separation of concerns across data preparation, model training, backend API, and frontend UI.

## Project Structure

