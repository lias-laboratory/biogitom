# ================================
# 📦 Required Libraries
# ================================

import os
import json
import numpy as np
import pandas as pd
import torch
import shutil
from rdflib import Graph, Namespace
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# ================================
# 🧠 Load Models
# ================================

print("🔄 Loading SapBERT model and tokenizer...")
# Load the SapBERT model for biomedical concept embedding
sapbert_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
sapbert_tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

print("🔄 Loading SentenceTransformer model...")
# Load SentenceTransformer for mean pooling (used internally)
sentence_model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")


# ================================
# 📂 Ontology File Upload Utility (optional use)
# ================================

def upload_ontology(prompt, dataset_dir):
    """
    Ask the user to upload an ontology file (in OWL format).
    The file will be copied into the dataset directory.
    """
    print(f"\n{prompt}")
    for attempt in range(3):
        path = input("📄 Provide full path to ontology file (.owl): ").strip()
        if os.path.exists(path) and path.endswith(".owl"):
            filename = os.path.basename(path)
            shutil.copy(path, os.path.join(dataset_dir, filename))
            print("✅ Ontology copied.")
            return filename
        else:
            print("❌ File not found or invalid format.")
            if attempt < 2 and input("Try again? (y/n): ").strip().lower() != "y":
                exit(1)
    print("🚘 Too many failed attempts.")
    exit(1)


# ================================
# 🔧 Utility Functions
# ================================

def get_file_synonym_properties(file_path):
    """
    Get the appropriate namespace and synonym annotation property from dictionary.json,
    based on the file name (e.g., ncit.owl → [namespace, annotation]).
    """
    with open("dictionary.json", "r") as f:
        synonym_dict = json.load(f)
    file_name = os.path.basename(file_path)
    return synonym_dict.get(file_name, [])

def get_exact_matches(graph, class_iri, label, namespace, annotation):
    """
    Collect all synonym values for a given class IRI using the provided annotation property.
    """
    matches = [label]
    for iri, _, val in graph.triples((None, namespace[annotation], None)):
        if str(iri) == class_iri:
            matches.append(str(val))
    return list(np.unique(matches))

def extract_nodes(file_path):
    """
    Parse the ontology to extract a dictionary mapping each class IRI
    to its label and synonyms using RDFS.label and the provided synonym property.
    """
    class_dict = {}
    graph = Graph()
    graph.parse(file_path, format="xml")
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    SKOS_ns, annotation = get_file_synonym_properties(file_path)
    for subj in graph.subjects():
        if str(subj).startswith("http") and graph.value(subj, RDFS.label):
            label = str(graph.value(subj, RDFS.label))
            class_dict[str(subj)] = get_exact_matches(graph, str(subj), label, Namespace(SKOS_ns), annotation)
    return class_dict

def extract_links(file_path, class_dict):
    """
    Extract subclass relations from the ontology as directed edges (subject → object),
    only if both subject and object are among the parsed classes.
    """
    graph = Graph()
    graph.parse(file_path, format="xml")
    links = []
    for subj, pred, obj in graph:
        if all([str(x).startswith("http") for x in (subj, obj)]) \
            and str(pred).endswith("subClassOf") \
            and str(subj) in class_dict \
            and str(obj) in class_dict:
            links.append([str(subj), str(obj)])
    return np.array(links)

def oht(class_dict, offset=0):
    """
    One-hot index encoder: assign an integer index to each URI.
    """
    return dict(zip(list(class_dict.keys()), [i + offset for i in range(len(class_dict))]))

def get_adjacency_matrix(class_dict, links):
    """
    Generate a binary adjacency matrix based on subclass relations between class URIs.
    """
    id_map = oht(class_dict)
    n = len(class_dict)
    adj_matrix = np.zeros((n, n), dtype=int)
    for subj, obj in links:
        if subj in id_map and obj in id_map:
            i = id_map[subj]
            j = id_map[obj]
            adj_matrix[i, j] = 1
    return adj_matrix

def gen_embeddings(sentences, batch_size=8, max_length=128, use_gpu=True):
    """
    Generate sentence embeddings using SapBERT on top of SentenceTransformer pooling.
    Uses mean pooling over token embeddings from the encoder output.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    sentence_model.to(device)

    # Replace internal base model weights with SapBERT (transfer learning)
    base_model = sentence_model._first_module().auto_model
    base_model.load_state_dict(sapbert_model.state_dict(), strict=False)

    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = sapbert_tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            output = base_model(**inputs)
            batch_embeds = output.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(batch_embeds)
        torch.cuda.empty_cache()
    return np.vstack(all_embeddings)

def save_results(class_dict, embeddings, adjacence, owl_path, task="task_name"):
    """
    Save ontology results:
    - classes JSON with synonyms,
    - embeddings as CSV,
    - adjacency matrix as CSV.
    """
    base_name = os.path.splitext(os.path.basename(owl_path))[0]
    data_dir = os.path.join("Tasks", task, "Data")
    os.makedirs(data_dir, exist_ok=True)

    class_path = os.path.join(data_dir, f"{base_name}_classes.json")
    emb_path = os.path.join(data_dir, f"{base_name}_emb.csv")
    adj_path = os.path.join(data_dir, f"{base_name}_adjacence.csv")

    with open(class_path, "w", encoding="utf-8") as f:
        json.dump(class_dict, f, indent=2)

    pd.DataFrame(embeddings).to_csv(emb_path, index=False)
    pd.DataFrame(adjacence).to_csv(adj_path, index=False)

    print(f"✅ Saved files for {base_name} to {data_dir}")


# ================================
# 🚀 Main Execution
# ================================

if __name__ == "__main__":
    print("📁 Starting preprocessing...")

    task = "task_name"         # Automatically replaced by create_cfe_script
    src_call = "src_name"      # Automatically replaced by create_cfe_script
    tgt_call = "tgt_name"      # Automatically replaced by create_cfe_script

    base_dir = "../biogitom"
    dataset_dir = os.path.join(base_dir, "Datasets", task)
    os.makedirs(dataset_dir, exist_ok=True)

    # Paths to source and target OWL ontology files
    src_path = os.path.join(dataset_dir, f"{src_call}.owl")
    tgt_path = os.path.join(dataset_dir, f"{tgt_call}.owl")

    for owl_path, name in [(src_path, src_call), (tgt_path, tgt_call)]:
        print(f"\n📄 Processing ontology '{name}.owl' — generating embeddings and adjacency matrix...")
        class_dict = extract_nodes(owl_path)                              # Extract class labels and synonyms
        links = extract_links(owl_path, class_dict)                      # Extract subclass relations
        labels = [", ".join(labs) for labs in class_dict.values()]      # Join all labels/synonyms
        embeddings = gen_embeddings(labels)                              # Encode into vectors
        adjacency = get_adjacency_matrix(class_dict, links)             # Generate graph structure

        # Save results to the task's data directory
        save_results(
            class_dict,
            embeddings,
            adjacency,
            owl_path=os.path.join("Tasks", task, "Data", f"{name}.owl"),
            task=task
        )

    print("\n✅ Ontology embeddings and graphs generated for both source and target.")

