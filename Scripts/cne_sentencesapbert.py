# Required Libraries
import os
import json
import numpy as np
import pandas as pd
from rdflib import Graph, Namespace
from transformers import AutoTokenizer, AutoModel
import torch


# Define utility functions
def get_exact_matches(graph, class_iri, label, namespace, annotation):
    """
    Retrieves exact matches for a given class in an ontology graph based on a specified annotation property.
    """
    exact_matches = [label]
    for iri, _, exact_match in graph.triples((None, namespace[annotation], None)):
        if str(iri) == class_iri:
            exact_matches.append(str(exact_match))
    return list(np.unique(exact_matches))


def extract_nodes(file_path):
    """
    Extracts nodes (classes) from an OWL file, retrieving their labels and exact matches.
    """
    class_dict = {}
    graph = Graph()
    graph.parse(file_path, format="xml")
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

    # Namespace for SKOS-like annotations
    SKOS, annotation = get_file_synonym_properties(file_path)

    for subj in graph.subjects(predicate=None):
        if str(subj).startswith("http") and graph.value(subject=subj, predicate=RDFS.label):
            label = str(graph.value(subject=subj, predicate=RDFS.label))
            class_dict[str(subj)] = get_exact_matches(graph, str(subj), label, Namespace(SKOS), annotation)
    return class_dict


def get_file_synonym_properties(file_path):
    """
    Loads synonym properties for a specific ontology from a dictionary file.
    """
    with open("dictionary.json", "r") as f:
        synonym_dict = json.load(f)
    file_name = os.path.basename(file_path)
    return synonym_dict.get(file_name, [])


def extract_links(file_path, class_dict):
    """
    Extracts binary relationships (e.g., subClassOf) from an OWL file for specified classes.
    """
    graph = Graph()
    graph.parse(file_path, format="xml")
    links = []
    for subj, pred, obj in graph:
        if (
            str(subj).startswith("http")
            and str(obj).startswith("http")
            and str(pred).endswith("subClassOf")
            and str(subj) in class_dict
            and str(obj) in class_dict
        ):
            links.append([str(subj), str(obj)])
    return np.array(links)

# Load SapBERT model and tokenizer
sapbert_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
sapbert_tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

# Load SentenceTransformer-compatible BERT model
sentence_model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")

def gen_embeddings(sentences, batch_size=8, max_length=128, use_gpu=True):
    """
    Generate sentence embeddings using a SentenceTransformer model with SapBERT weights.

    Args:
        sentences (list of str): A list of sentences to encode.
        batch_size (int): Number of sentences to process at a time (for batching).
        max_length (int): Maximum sequence length for tokenization.
        use_gpu (bool): Whether to use GPU for computation. Defaults to True.

    Returns:
        np.ndarray: The embeddings for the input sentences.
    """
    # Determine the device to use: GPU (if available and requested) or CPU
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Load the SentenceTransformer model with SapBERT weights
    sentence_model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens").to(device)
    sentence_transformer_model = sentence_model._first_module().auto_model  # Access underlying BERT model
    sentence_transformer_model.load_state_dict(sapbert_model.state_dict(), strict=False)

    # Tokenizer for the SentenceTransformer model
    tokenizer = sapbert_tokenizer  # Reuse SapBERT tokenizer

    # Store all embeddings here
    all_embeddings = []

    # Process the sentences in batches
    for i in range(0, len(sentences), batch_size):
        # Get the current batch
        batch_sentences = sentences[i:i + batch_size]

        # Tokenize the batch with truncation to limit sequence length
        encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)

        # Generate embeddings without computing gradients (for efficiency)
        with torch.no_grad():
            model_output = sentence_transformer_model(**encoded_input)
            batch_embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()

        # Append batch embeddings to the list
        all_embeddings.append(batch_embeddings)

        # Clear the GPU cache to free memory
        torch.cuda.empty_cache()

    # Concatenate all batch embeddings into a single array
    all_embeddings = np.vstack(all_embeddings)

    return all_embeddings

def save_results(class_dict, embeddings, adjacence, file_path):
    """
    Saves processed data (class dictionary, embeddings, and adjacency matrix) to files.
    """
    base_path = f"{os.path.splitext(file_path)[0]}"
    with open(f"{base_path}_classes.json", "w") as f:
        json.dump(class_dict, f, indent=4)
    pd.DataFrame(embeddings).to_csv(f"{base_path}_emb.csv", index=False)
    pd.DataFrame(adjacence).to_csv(f"{base_path}_adjacence.csv", index=False)


# Main script execution
if __name__ == "__main__":
    """
    Main entry point for extracting nodes, relationships, embeddings, and adjacency matrices from an OWL ontology.
    """
    # Define task-specific parameters
    src_ent = "snomed.pharm"  # Source ontology name
    tgt_ent = "ncit.pharm"    # Target ontology name
    task = "pharm"            # Task identifier
    # Paths and parameters
    # # Define base directory structure
    base_dir = "../../../biogitom"  # Base directory for datasets and tasks
    dataset_dir = os.path.join(base_dir, "Datasets", task)  # Dataset directory
    # Define paths for source and target OWL files
    src_file_path = os.path.join(dataset_dir, f"{src_ent}.owl")  # Source ontology OWL file
    tgt_file_path = os.path.join(dataset_dir, f"{tgt_ent}.owl")
   
    # Extract nodes and relationships
    print("Extracting concepts from source ontology...")
    class_dict = extract_nodes(src_file_path)

    print("Extracting relationships from source ontology...")
    links = extract_links(src_file_path, class_dict)

    print("Generating embeddings...")
    concat_labels = [", ".join(labels) for labels in class_dict.values()]
    embeddings = gen_embeddings(concat_labels)

    print("Saving results...")
    save_results(class_dict, embeddings, links, src_file_path)

    print("Processing completed successfully.")
