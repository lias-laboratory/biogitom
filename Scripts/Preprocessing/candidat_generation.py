# **Importing Required Libraries**
import json  # For working with JSON data
from owlready2 import *  # For manipulating OWL ontologies
import pandas as pd  # For handling tabular data
import numpy as np  # For numerical computations
import torch  # For PyTorch deep learning models
import os  # For interacting with the operating system
from transformers import AutoTokenizer  # For Hugging Face tokenizer
from rdflib import Graph, Namespace, URIRef  # For RDF graph handling
from collections import defaultdict  # For default dictionary initialization
from typing import List  # For type hints in Python
import math  # For mathematical operations
import csv  # For handling CSV file reading and writing
from bs4 import BeautifulSoup  # For parsing HTML/XML data
from lxml import etree  # For advanced XML/HTML processing
import pickle  # For serializing and deserializing objects
import networkx as nx  # For working with graph-based data structures

# **Class Definitions**

# Class to handle ontology labels and synonyms (text processing)
class OntoText:
    def __init__(self, data):
        """
        Initialize the OntoText class with ontology data.
        :param data: A dictionary where keys are concept IDs and values are lists of labels or synonyms.
        """
        self.data = data  # Raw input data
        self.texts = defaultdict(list)  # Store processed labels or synonyms
        self.class2idx = {}  # Map concept IDs to unique numerical indices
        self.idx2class = {}  # Map indices back to concept IDs
        self.extract_texts()  # Extract and preprocess the labels
        self.create_class_idx_mappings()  # Generate mapping dictionaries

    def extract_texts(self):
        """
        Extracts and normalizes text data by converting to lowercase.
        """
        for concept_id, labels in self.data.items():
            for label in labels:
                self.texts[concept_id].append(label.lower())

    def create_class_idx_mappings(self):
        """
        Create unique mappings between concept IDs and numerical indices.
        """
        for idx, concept_id in enumerate(self.texts.keys()):
            self.class2idx[concept_id] = idx
            self.idx2class[idx] = concept_id


# Class to build an inverted index for token-based concept mapping
class OntoInvertedIndex:
    def __init__(self, ontotext: OntoText, tokenizer_path: str, cut: int = 0):
        """
        Initialize the OntoInvertedIndex class.
        :param ontotext: An OntoText object containing processed ontology data.
        :param tokenizer_path: Path to a pre-trained tokenizer (e.g., BioBERT).
        :param cut: Minimum token length to consider in the index.
        """
        self.ontotext = ontotext  # Processed ontology data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # Load the tokenizer
        self.cut = cut  # Length threshold for tokens to include
        self.index = self.construct_index()  # Build the inverted index

    def tokenize(self, texts: List[str]) -> List[str]:
        """
        Tokenize a list of texts using the pre-trained tokenizer.
        :param texts: List of strings to tokenize.
        :return: List of tokens.
        """
        return [token for text in texts for token in self.tokenizer.tokenize(text)]

    def construct_index(self):
        """
        Build the inverted index mapping tokens to concept indices.
        :return: A dictionary with tokens as keys and lists of concept indices as values.
        """
        index = defaultdict(list)  # Initialize an empty index
        for concept_id, labels in self.ontotext.texts.items():
            tokens = self.tokenize(labels)  # Tokenize the labels
            for token in tokens:
                if len(token) > self.cut:  # Include tokens exceeding the length threshold
                    index[token].append(self.ontotext.class2idx[concept_id])
        return index


# Class to manage source and target ontologies and their candidate mappings
class OntoBox:
    def __init__(self, src_data, tgt_data, tokenizer_path="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR", cut=0):
        """
        Initialize the OntoBox class with source and target ontology data.
        :param src_data: Dictionary representing the source ontology.
        :param tgt_data: Dictionary representing the target ontology.
        :param tokenizer_path: Path to a pre-trained tokenizer (e.g., BioBERT or SapBERT).
        :param cut: Minimum token length for the inverted index.
        """
        # Process source and target ontology data
        self.src_ontotext = OntoText(src_data)
        self.tgt_ontotext = OntoText(tgt_data)

        # Create inverted indices for source and target data
        self.src_onto_index = OntoInvertedIndex(self.src_ontotext, tokenizer_path, cut=cut)
        self.tgt_onto_index = OntoInvertedIndex(self.tgt_ontotext, tokenizer_path, cut=cut)

    def select_candidates(self, concept_texts: List[str], candidate_limit: int = 10):
        """
        Select candidate concepts from the target ontology based on token overlap and IDF scoring.
        :param concept_texts: Labels or synonyms for a source concept.
        :param candidate_limit: Maximum number of candidates to return.
        :return: A list of candidate IDs from the target ontology.
        """
        candidate_pool = defaultdict(lambda: 0)  # Dictionary to store candidate scores
        tokens = self.tgt_onto_index.tokenize(concept_texts)  # Tokenize the source concept labels
        D = len(self.tgt_ontotext.class2idx)  # Total number of target concepts

        for token in tokens:
            potential_candidates = self.tgt_onto_index.index.get(token, [])  # Retrieve concepts with this token
            if not potential_candidates:
                continue
            idf = math.log10(D / len(potential_candidates))  # Calculate Inverse Document Frequency (IDF)
            for class_id in potential_candidates:
                candidate_pool[class_id] += idf  # Accumulate IDF scores for each candidate

        # Sort candidates by their scores and limit the result
        sorted_candidates = sorted(candidate_pool.items(), key=lambda x: x[1], reverse=True)[:candidate_limit]
        return [self.tgt_ontotext.idx2class[c[0]] for c in sorted_candidates]

    def generate_candidates(self, candidate_limit: int = 10):
        """
        Generate candidate mappings between source and target ontologies.
        :param candidate_limit: Maximum number of candidates per source concept.
        :return: List of (source concept, target concept) pairs.
        """
        candidate_pairs = []  # Initialize an empty list of mappings
        for src_id, text_dict in self.src_ontotext.texts.items():
            src_texts = text_dict  # Retrieve all synonyms for the source concept
            candidates = self.select_candidates(src_texts, candidate_limit)  # Find target candidates
            for tgt_id in candidates:
                candidate_pairs.append((src_id, tgt_id))  # Add source-target pair
        return candidate_pairs


# **Functions for Cleaning Ontology JSON Files**

def clean_json_by_alignment(file_path, json_path, output_path):
    """
    Clean a JSON file by filtering concepts based on alignment annotations in an OWL file.
    :param file_path: Path to the OWL file.
    :param json_path: Path to the JSON file to clean.
    :param output_path: Path to save the cleaned JSON file.
    """
    g = Graph()  # Initialize an RDF graph
    g.parse(file_path, format='xml')  # Load the OWL ontology into the graph

    # Define the alignment property
    USE_IN_ALIGNMENT = URIRef("http://oaei.ontologymatching.org/bio-ml/ann/use_in_alignment")

    with open(json_path, "r") as f:
        data = json.load(f)  # Load the JSON data

    # Identify valid IRIs
    valid_iris = {str(subj) for subj in g.subjects(predicate=None)
                  if g.value(subject=subj, predicate=USE_IN_ALIGNMENT) in (None, "true")}

    # Filter JSON data based on valid IRIs
    cleaned_data = {iri: labels for iri, labels in data.items() if iri in valid_iris}

    # Save the cleaned JSON data
    with open(output_path, "w") as f:
        json.dump(cleaned_data, f, indent=4)
    print(f"Cleaned JSON saved to {output_path}")


def clean_json_using_tsv(json_file_path, tsv_file_path, output_json_path):
    """
    Clean a JSON file by filtering entries based on a reference TSV file.
    :param json_file_path: Path to the JSON file.
    :param tsv_file_path: Path to the TSV file containing valid entries.
    :param output_json_path: Path to save the cleaned JSON file.
    """
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)  # Load JSON data

    # Load TSV data and extract valid entities
    tsv_data = pd.read_csv(tsv_file_path, sep="\t")
    valid_entities = set(tsv_data['SrcEntity'])

    # Filter the JSON data
    cleaned_data = {key: value for key, value in json_data.items() if key in valid_entities}

    # Save the cleaned JSON data
    with open(output_json_path, 'w') as output_file:
        json.dump(cleaned_data, output_file, indent=4)
    print(f"Cleaned JSON saved to {output_json_path}")

# **Function for Building Indexed Dictionary**

def build_indexed_dict(file_path):
    """
    Create an indexed dictionary where each URI is assigned a unique numerical index.
    :param file_path: Path to the JSON file containing ontology data.
    :return: A dictionary mapping URIs to unique indices.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)  # Load JSON data from the file
    return {key: index for index, key in enumerate(data.keys())}  # Map URIs to indices


# **Function to Encode URIs**

def encode_uris(row, source_dict, target_dict):
    """
    Encode URIs into numerical representations using indexed dictionaries.
    :param row: A row from a pandas DataFrame containing `SrcEntity` and `TgtEntity`.
    :param source_dict: Dictionary mapping source URIs to indices.
    :param target_dict: Dictionary mapping target URIs to indices.
    :return: A pandas Series with encoded source and target URIs.
    """
    uri_1, uri_2 = row['SrcEntity'], row['TgtEntity']  # Extract source and target URIs
    return pd.Series([source_dict.get(uri_1, -1), target_dict.get(uri_2, -1)])  # Encode URIs or assign -1 if missing


# **Main Script Execution**

if __name__ == "__main__":
    """
    Main entry point for ontology processing tasks such as cleaning JSON files,
    building inverted indices, and generating candidates.
    """

    # **Task-Specific Configuration**

    # Define the source and target ontology names and the task identifier
    src_ent = "snomed.pharm"  # Source ontology
    tgt_ent = "ncit.pharm"    # Target ontology
    task = "pharm"            # Task identifier

    # **Directory Configuration**

    # Base directory containing datasets and task data
    base_dir = "../../../biogitom"
    dataset_dir = os.path.join(base_dir, "Datasets", task)  # Path to ontology datasets
    data_dir = os.path.join(base_dir, "Tasks", task, "Data")  # Path for processed data

    # **File Paths for Ontology Data**

    # Paths for source and target OWL files
    src_owl_path = os.path.join(dataset_dir, f"{src_ent}.owl")
    tgt_owl_path = os.path.join(dataset_dir, f"{tgt_ent}.owl")

    # Paths for source and target JSON files (before and after cleaning)
    src_json_path = os.path.join(data_dir, f"{src_ent}_classes2.json")
    tgt_json_path = os.path.join(data_dir, f"{tgt_ent}_classes.json")
    src_cleaned_json_path = os.path.join(data_dir, f"{src_ent}_cleaned_classes2.json")
    tgt_cleaned_json_path = os.path.join(data_dir, f"{tgt_ent}_cleaned_classes2.json")

    # Ensure that the necessary directories exist
    os.makedirs(data_dir, exist_ok=True)

    # **Cleaning JSON Files**

    # Clean the source ontology JSON file based on alignment annotations in the OWL file
    print("Cleaning source ontology JSON...")
    clean_json_by_alignment(src_owl_path, src_json_path, src_cleaned_json_path)
    print(f"Source ontology cleaned JSON saved to: {src_cleaned_json_path}")

    # Clean the target ontology JSON file based on alignment annotations in the OWL file
    print("Cleaning target ontology JSON...")
    clean_json_by_alignment(tgt_owl_path, tgt_json_path, tgt_cleaned_json_path)
    print(f"Target ontology cleaned JSON saved to: {tgt_cleaned_json_path}")

    # **Candidate Generation**

    # Load cleaned source and target ontology JSON data
    with open(src_cleaned_json_path, 'r') as f:
        src_data = json.load(f)  # Load cleaned source ontology data

    with open(tgt_cleaned_json_path, 'r') as f:
        tgt_data = json.load(f)  # Load cleaned target ontology data

    # Instantiate the OntoBox class to manage ontologies and generate candidates
    print("Generating candidate mappings...")
    ontobox = OntoBox(src_data, tgt_data)
    candidates = ontobox.generate_candidates()  # Generate candidate mappings

    # Convert the candidate pairs into a pandas DataFrame
    candidate_df = pd.DataFrame(candidates, columns=["SrcEntity", "TgtEntity"])

    # Add an ID column to uniquely identify each candidate pair
    candidate_df.insert(0, "ID", range(len(candidate_df)))

    # Save the generated candidates to a CSV file
    candidate_path = os.path.join(data_dir, f"{task}_candidates.csv")
    candidate_df.to_csv(candidate_path, index=False)
    print(f"Candidate mappings saved successfully to: {candidate_path}")

    # **Encoding URIs**

    # Build indexed dictionaries for source and target ontologies
    print("Building indexed dictionaries for encoding URIs...")
    indexed_dict_source = build_indexed_dict(src_cleaned_json_path)
    indexed_dict_target = build_indexed_dict(tgt_cleaned_json_path)

    # Encode the URIs in the candidate mappings using the indexed dictionaries
    print("Encoding candidate URIs...")
    candidate_df[["SrcEntity", "TgtEntity"]] = candidate_df.apply(
        encode_uris, axis=1, source_dict=indexed_dict_source, target_dict=indexed_dict_target
    )

    # Add the ID column again after encoding
    candidate_df["ID"] = candidate_df.index + 1

    # Save the encoded candidates to a CSV file
    encoded_candidate_path = os.path.join(data_dir, f"{task}_candidates_encoded.csv")
    candidate_df[["ID", "SrcEntity", "TgtEntity"]].to_csv(
        encoded_candidate_path, index=False, quoting=csv.QUOTE_NONNUMERIC
    )
    print(f"Encoded candidate mappings saved to: {encoded_candidate_path}")

