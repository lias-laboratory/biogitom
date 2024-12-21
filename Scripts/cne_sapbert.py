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


def gen_embeddings(sentences, batch_size=8, max_length=128, use_gpu=True):
    """
    Generates sentence embeddings using SapBERT.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model_name = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            embeddings = model(**encoded_input).last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)


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
