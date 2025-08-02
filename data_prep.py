# data_prep.py
import re
from datasets import load_dataset
from transformers import BartTokenizer
import nltk

# --- Configuration ---
# We'll use the same model for tokenization as the one we plan to fine-tune.
# This ensures the token IDs match the model's vocabulary.
MODEL_CHECKPOINT = "facebook/bart-large-cnn"
# The maximum number of tokens the BART model can handle in a single input.
MAX_INPUT_LENGTH = 1024
# The 'billsum' dataset is a standard benchmark for summarization.
# It contains US congressional bills, which serve as a good proxy for
# complex legal and financial documents.
DATASET_NAME = "billsum"


def setup_nltk():
    """
    Downloads the NLTK sentence tokenizer model (punkt), which is needed for
    splitting text into sentences. This is useful for more intelligent chunking.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' model not found. Downloading...")
        nltk.download('punkt')
        print("Download complete.")

def clean_text(text):
    """
    Performs basic text cleaning.
    - Removes extra whitespace and newlines.
    - Can be expanded to handle HTML tags or other artifacts if using raw data.
    """
    if not isinstance(text, str):
        return ""
    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_chunks(text, tokenizer, max_len):
    """
    Splits a long text into smaller chunks that fit within the model's max token limit.
    This is a crucial step for handling long financial documents.

    Args:
        text (str): The input text to be chunked.
        tokenizer: The tokenizer instance.
        max_len (int): The maximum token length for each chunk.

    Returns:
        list[str]: A list of text chunks.
    """
    # First, tokenize the entire text to see the total length
    tokens = tokenizer.encode(text)

    # If the text is already short enough, no need to chunk
    if len(tokens) <= max_len:
        return [text]

    # Use NLTK to split text into sentences for more coherent chunking
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk_tokens = []
    
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        
        # If adding the next sentence exceeds the max length, finalize the current chunk
        if len(current_chunk_tokens) + len(sentence_tokens) > max_len - 2: # -2 for special tokens
            if current_chunk_tokens: # Ensure the chunk is not empty
                chunk_text = tokenizer.decode(current_chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
            current_chunk_tokens = sentence_tokens # Start a new chunk with the current sentence
        else:
            current_chunk_tokens.extend(sentence_tokens)

    # Add the last remaining chunk
    if current_chunk_tokens:
        chunk_text = tokenizer.decode(current_chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
    return chunks

def process_and_chunk_dataset(dataset, tokenizer, max_input_length):
    """
    Applies cleaning and chunking to the entire dataset.
    """
    processed_examples = {'document_chunks': [], 'summary': []}
    
    for example in dataset:
        document = clean_text(example['text'])
        summary = clean_text(example['summary'])
        
        if not document or not summary:
            continue
            
        chunks = create_chunks(document, tokenizer, max_input_length)
        
        # For each chunk, we keep the original summary.
        # During training, the model learns to summarize a part of the document.
        for chunk in chunks:
            processed_examples['document_chunks'].append(chunk)
            processed_examples['summary'].append(summary)
            
    return processed_examples


def main():
    """
    Main function to execute the data preparation pipeline.
    """
    print("--- Phase 1: Data Preparation ---")
    
    # 1. Setup NLTK
    setup_nltk()
    
    # 2. Load Tokenizer
    print(f"Loading tokenizer for '{MODEL_CHECKPOINT}'...")
    tokenizer = BartTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # 3. Load Dataset from Hugging Face
    print(f"Loading '{DATASET_NAME}' dataset...")
    # We'll use a small portion of the training set for demonstration
    raw_dataset = load_dataset(DATASET_NAME, split='train[:1%]') 
    print(f"Loaded {len(raw_dataset)} examples.")
    
    # 4. Process and Chunk a single example for demonstration
    print("\n--- Demonstrating Chunking on a Single Document ---")
    example_document = raw_dataset[0]['text']
    example_summary = raw_dataset[0]['summary']
    
    print(f"Original document length (chars): {len(example_document)}")
    
    # Clean the text first
    cleaned_document = clean_text(example_document)
    
    # Create chunks
    chunks = create_chunks(cleaned_document, tokenizer, MAX_INPUT_LENGTH)
    
    print(f"Document was split into {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        chunk_token_count = len(tokenizer.encode(chunk))
        print(f"  - Chunk {i+1}: {len(chunk)} chars, {chunk_token_count} tokens")
        
    print("\nOriginal Summary:")
    print(clean_text(example_summary))

    # In a full pipeline, you would process the entire dataset like this:
    # print("\nProcessing the full dataset (this might take a while)...")
    # processed_data = process_and_chunk_dataset(raw_dataset, tokenizer, MAX_INPUT_LENGTH)
    # print(f"Created {len(processed_data['document_chunks'])} training examples from {len(raw_dataset)} documents.")
    # From here, you would save `processed_data` to a file or use it directly for training.
    

if __name__ == "__main__":
    main()

