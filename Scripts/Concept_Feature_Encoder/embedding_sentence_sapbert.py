# Import required libraries
import json  # For working with JSON files
import pandas as pd  # For handling tabular data
import numpy as np  # For numerical operations
import torch  # For PyTorch deep learning framework
import os  # For interacting with the operating system
from transformers import AutoModel, AutoTokenizer  # Hugging Face models and tokenizers
from sentence_transformers import SentenceTransformer  # Sentence-level embeddings

# **Configuration Section**

# Define the task name for ontology matching
task = "body"

# Define local directory paths
base_dir = "../biogitom"  # Replace with your local base directory
dir = os.path.join(base_dir, "BioGITOM-VLDB", task)

# Dataset directory for source and target ontologies
dataset_dir = os.path.join(base_dir, "Datasets", task)

# Directory to store embeddings, adjacency matrices, etc.
data_dir = os.path.join(dir, "Data")

# Paths to source and target ontology JSON files and embedding files
src_class = os.path.join(data_dir, "snomed.body_classes.json")  # Source ontology classes
src_Emb = os.path.join(data_dir, "snomed.body_BERT_Hybrid_emb.csv")  # File to save source embeddings

tgt_class = os.path.join(data_dir, "fma.body_classes.json")  # Target ontology classes
tgt_Emb = os.path.join(data_dir, "fma.body_BERT_Hybrid_emb.csv")  # File to save target embeddings

# **Model Loading**

# Load SapBERT model and tokenizer (pre-trained transformer for biomedical text)
sapbert_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
sapbert_tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

# Load a SentenceTransformer model for generating sentence embeddings
sentence_model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")

# **Function Definition**

def gen_embeddings(sentences, batch_size=8, max_length=128, use_gpu=True):
    """
    Generate sentence embeddings using a SentenceTransformer model with SapBERT weights.

    Args:
        sentences (list of str): A list of sentences to encode.
        batch_size (int): Number of sentences to process in a single batch.
        max_length (int): Maximum sequence length for tokenization.
        use_gpu (bool): Whether to use GPU for computation. Defaults to True.

    Returns:
        np.ndarray: The generated embeddings for the input sentences.
    """
    # Determine the device (GPU or CPU) to use
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Load the SentenceTransformer model and transfer to the selected device
    sentence_model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens").to(device)
    # Access the underlying transformer model in SentenceTransformer
    sentence_transformer_model = sentence_model._first_module().auto_model
    # Load SapBERT weights into the model
    sentence_transformer_model.load_state_dict(sapbert_model.state_dict(), strict=False)

    # Tokenizer for the transformer model (reuse SapBERT tokenizer)
    tokenizer = sapbert_tokenizer

    # List to store embeddings for all sentences
    all_embeddings = []

    # Process sentences in batches
    for i in range(0, len(sentences), batch_size):
        # Extract the current batch of sentences
        batch_sentences = sentences[i:i + batch_size]

        # Tokenize the sentences with truncation to the specified maximum length
        encoded_input = tokenizer(batch_sentences, padding=True, truncation=True,
                                  max_length=max_length, return_tensors='pt').to(device)

        # Generate embeddings (no gradients required for inference)
        with torch.no_grad():
            model_output = sentence_transformer_model(**encoded_input)
            # Compute mean of the last hidden state for sentence embeddings
            batch_embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()

        # Append batch embeddings to the list
        all_embeddings.append(batch_embeddings)

        # Clear GPU memory to avoid overflow
        torch.cuda.empty_cache()

    # Concatenate all batch embeddings into a single array
    return np.vstack(all_embeddings)

# **Embedding Generation for Source Ontology**

# Load source ontology JSON file
with open(src_class, "r") as f:
    class_dict = json.loads(f.read())

# Prepare sentences for embedding generation by joining all synonyms/labels with commas
concat_arr = [", ".join(list(x)) for x in class_dict.values()]

# Generate embeddings for the source ontology
emb = gen_embeddings(concat_arr)

# Save the embeddings to a CSV file
df = pd.DataFrame(emb)  # Convert embeddings to a DataFrame
df.to_csv(src_Emb, index=False)  # Save DataFrame to CSV

# **Embedding Generation for Target Ontology**

# Load target ontology JSON file
with open(tgt_class, "r") as f:
    class_dict = json.loads(f.read())

# Prepare sentences for embedding generation by joining all synonyms/labels with commas
concat_arr = [", ".join(list(x)) for x in class_dict.values()]

# Generate embeddings for the target ontology
embtgt = gen_embeddings(concat_arr)

# Save the embeddings to a CSV file
df = pd.DataFrame(embtgt)  # Convert embeddings to a DataFrame
df.to_csv(tgt_Emb, index=False)  # Save DataFrame to CSV
