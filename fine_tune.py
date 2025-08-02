# fine_tune.py
import os
import nltk
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from rouge_score import rouge_scorer

# --- Configuration ---
# Model checkpoint for the pre-trained model
MODEL_CHECKPOINT = "facebook/bart-large-cnn"
# Dataset from Hugging Face
DATASET_NAME = "billsum"
# Directory to save the fine-tuned model
MODEL_OUTPUT_DIR = "./financial_summarizer_model"

# --- Hyperparameters for Training ---
# These are set for a quick test run. For a full training, you'd increase these.
NUM_TRAIN_EPOCHS = 1
TRAIN_BATCH_SIZE = 4  # Per device
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 2e-5
# Percentage of the dataset to use for training and testing. 1.0 for full dataset.
# We use a small fraction to make the test run fast.
DATASET_FRACTION_TO_USE = 0.01 # Use 1% of the data

# --- Helper Functions from Phase 1 (included for a self-contained script) ---
def setup_nltk():
    """Downloads the NLTK sentence tokenizer models if not present."""
    for resource in ['punkt', 'punkt_tab']:
        try:
            # The find method raises a LookupError if the resource is not found.
            nltk.data.find(f'tokenizers/{resource}')
        # *** FIXED: Changed to catch the correct exception, LookupError. ***
        except LookupError:
            print(f"Downloading NLTK '{resource}' resource...")
            nltk.download(resource)

def preprocess_function(examples, tokenizer):
    """Tokenizes the source text and target summaries."""
    # The 'text' column contains the document content.
    inputs = tokenizer(examples["text"], max_length=1024, truncation=True, padding="max_length")
    
    # The 'summary' column is the target.
    # The tokenizer needs to be aware that it's tokenizing the target labels.
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")

    inputs["labels"] = labels["input_ids"]
    return inputs

def compute_metrics(eval_pred, tokenizer):
    """
    Computes ROUGE scores for evaluation.
    This function is called by the Trainer during the evaluation loop.
    """
    predictions, labels = eval_pred
    # Decode generated summaries and reference summaries
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects newline-separated sentences
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate scores
    result = {
        key: [] for key in ['rouge1', 'rouge2', 'rougeL']
    }
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(label, pred)
        result['rouge1'].append(scores['rouge1'].fmeasure)
        result['rouge2'].append(scores['rouge2'].fmeasure)
        result['rougeL'].append(scores['rougeL'].fmeasure)

    # Get the average
    result = {key: np.mean(val) * 100 for key, val in result.items()}
    
    # Add a metric for the length of the generated summaries
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


def main():
    """Main function to execute the fine-tuning pipeline."""
    print("--- Phase 2: Model Fine-Tuning ---")
    
    # 1. Setup NLTK
    setup_nltk()

    # 2. Load Tokenizer and Model
    print(f"Loading tokenizer and model for '{MODEL_CHECKPOINT}'...")
    tokenizer = BartTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = BartForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)

    # 3. Load and Prepare Dataset
    print(f"Loading '{DATASET_NAME}' dataset...")
    raw_datasets = load_dataset(DATASET_NAME)
    
    # For a quick run, let's subsample the dataset
    print(f"Subsampling dataset to {DATASET_FRACTION_TO_USE*100}% of its size.")
    train_size = int(len(raw_datasets["train"]) * DATASET_FRACTION_TO_USE)
    test_size = int(len(raw_datasets["test"]) * DATASET_FRACTION_TO_USE)

    subsampled_datasets = DatasetDict({
        "train": raw_datasets["train"].select(range(train_size)),
        "test": raw_datasets["test"].select(range(test_size))
    })
    
    print("Tokenizing the dataset...")
    tokenized_datasets = subsampled_datasets.map(
        lambda examples: preprocess_function(examples, tokenizer), 
        batched=True
    )

    # 4. Set up Training Arguments
    print("Configuring training arguments...")
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        
    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        save_total_limit=3,
        predict_with_generate=True,
        eval_strategy="epoch",
        logging_strategy="epoch",
        fp16=False,  # Set to True if you have a compatible GPU and CUDA setup
    )

    # 5. Define Data Collator
    # This pads inputs and labels dynamically per batch, which is more efficient.
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 6. Initialize Trainer
    print("Initializing Trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer)
    )

    # 7. Start Fine-Tuning
    print("\nStarting fine-tuning process...")
    print(f"This will run for {NUM_TRAIN_EPOCHS} epoch(s) on {train_size} training examples.")
    trainer.train()
    print("Fine-tuning complete.")

    # 8. Save the Final Model
    print(f"Saving the fine-tuned model to '{MODEL_OUTPUT_DIR}'...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print("Model saved successfully!")
    print(f"\n--- Phase 2 Complete ---")
    print(f"Your fine-tuned model is now available at: {os.path.abspath(MODEL_OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
