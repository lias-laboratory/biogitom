# ============================================
# 🔧 Environment Setup: Import Required Packages
# ============================================

print("🔍 Setting up the environment and importing required libraries...")

import torch

import os

# Import pandas for data manipulation and analysis, such as loading, processing, and saving tabular data.
import pandas as pd

# Import pickle for saving and loading serialized objects (e.g., trained models or preprocessed data).
import pickle

# Import function to convert a directed graph to an undirected one, useful for certain graph algorithms.
from torch_geometric.utils import to_undirected

# Import optimizer module from PyTorch for training models using gradient-based optimization techniques.
import torch.optim as optim

# Import PyTorch's modules for defining neural network architectures and operations:
from torch.nn import (
    Linear,       # For linear transformations (dense layers).
    Sequential,   # For stacking layers sequentially.
    BatchNorm1d,  # For normalizing input within mini-batches.
    PReLU,        # Parametric ReLU activation function.
    Dropout       # For regularization by randomly dropping connections during training.
)

# Import functional API from PyTorch for operations like activations and loss functions.
import torch.nn.functional as F

# Import Matplotlib for visualizations, such as plotting training loss curves.
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt  

# Import PyTorch Geometric's graph convolutional layers:
from torch_geometric.nn import GCNConv, GINConv

# Import pooling operations for aggregating node embeddings to graph-level representations:
from torch_geometric.nn import global_mean_pool, global_add_pool

# Import NumPy for numerical operations, such as working with arrays and matrices.
import numpy as np

# Import time module for measuring execution time of code blocks.
import time

# Import typing module for specifying types in function arguments and return values.
from typing import Optional, Tuple, Union, Callable

# Import PyTorch's DataLoader and TensorDataset for handling data batching and loading during training.
from torch.utils.data import DataLoader, TensorDataset

# Import PyTorch's Parameter class for defining learnable parameters in custom models.
from torch.nn import Parameter

# Import math module for performing mathematical computations.
import math

# Import Tensor type from PyTorch for defining and manipulating tensors.
from torch import Tensor

# Import PyTorch's nn module for defining and building neural network architectures.
import torch.nn as nn

# Import initialization utilities from PyTorch Geometric for resetting weights and biases in layers.
from torch_geometric.nn.inits import reset

# Import the base class for defining message-passing layers in graph neural networks (GNNs).
from torch_geometric.nn.conv import MessagePassing

# Import linear transformation utilities for creating dense representations in graph models.
from torch_geometric.nn.dense.linear import Linear as PyGLinear

# Import typing utilities for defining adjacency matrices and tensor types specific to PyTorch Geometric.
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor

# Import softmax function for normalizing attention scores in GNNs.
from torch_geometric.utils import softmax

# Import initialization utilities for weight initialization (e.g., Glorot initialization).
from torch_geometric.nn.inits import glorot, zeros

# Import F1 score metric from scikit-learn for evaluating model performance in binary/multi-class tasks.
from sklearn.metrics import f1_score

# Import JSON module for reading and writing JSON files, useful for storing configuration or ontology data.
import json

# Import Ontology class from DeepOnto for representing and manipulating ontologies in the pipeline.
from deeponto.onto import Ontology

# Import tools from DeepOnto for handling Ontology Alignment Evaluation Initiative (OAEI) tasks.
from deeponto.align.oaei import *

# Import evaluation tools from DeepOnto for assessing alignment results using metrics like precision, recall, and F1.
from deeponto.align.evaluation import AlignmentEvaluator

# Import mapping utilities from DeepOnto for working with reference mappings and entity pairs.
from deeponto.align.mapping import ReferenceMapping, EntityMapping

# Import utility function for reading tables (e.g., TSV, CSV) from DeepOnto.
from deeponto.utils import read_table

# Importing the train_test_split function from sklearn's model_selection module.
from sklearn.model_selection import train_test_split

import random

# Set the seed for PyTorch's random number generator to ensure reproducibility
torch.manual_seed(42)

# Set the seed for NumPy's random number generator to ensure reproducibility
np.random.seed(42)

# Set the seed for Python's built-in random module to ensure reproducibility
random.seed(42)

"""**Paths Definition**"""

# === Ontology Matching Configuration ===

# Define the source ontology name
src_ent = "snomed.neoplas"

# Define the target ontology name
tgt_ent = "ncit.neoplas"

# Define the task name for this ontology matching process
task = "neoplas"          # Used to name intermediate and output files

# Set the value of top-k candidates to consider 
k = 2
# Score margin used for relaxed top-1 selection
score_margin = 0.007           # Minimum score difference required between top-1 and runner-up

print(f"Matching {src_ent}.owl and {tgt_ent}.owl:")

# Sets the relative path to the root directory of the BioGITOM project.
# This path points three levels up from the current working directory
# and navigates to the 'biogitom' folder. Adjust this path if the 
# directory structure changes or the script is executed from a different location.
dir = "../GNN-Match"

# === Define directories ===

# Directory containing the source and target ontologies
dataset_dir = f"{dir}/Datasets/{task}"

# Directory containing intermediate data (embeddings, adjacency matrices, etc.)
data_dir = f"{dir}/Tasks/{task}/Data"

# Directory to store prediction results and evaluation outputs
results_dir = f"{dir}/Tasks/{task}/Results"

# === Load Ontologies ===

# Load the source ontology (.owl file)
src_onto = Ontology(f"{dataset_dir}/{src_ent}.owl")

# Load the target ontology (.owl file)
tgt_onto = Ontology(f"{dataset_dir}/{tgt_ent}.owl")

# === Embeddings ===

# Initial semantic embeddings generated by SapBERT
src_Emb = f"{data_dir}/{src_ent}_Sentence_SapBERT_emb.csv"
tgt_Emb = f"{data_dir}/{tgt_ent}_Sentence_SapBERT_emb.csv"

# Final processed embeddings (e.g., after GNN/Gated network)
Emb_final_src = f"{data_dir}/{src_ent}_final_embeddings.csv"
Emb_final_tgt = f"{data_dir}/{tgt_ent}_final_embeddings.csv"

# Cleaned embeddings (with ignored concepts removed)
Emb_final_src_cl = f"{data_dir}/{src_ent}_final_embeddings_cleaned.csv"
Emb_final_tgt_cl = f"{data_dir}/{tgt_ent}_final_embeddings_cleaned.csv"

# Embeddings enriched and filtered for ranking-based evaluation
src_rank_emb = f"{data_dir}/{src_ent}_cands_with_embeddings.tsv"
tgt_rank_emb = f"{data_dir}/{tgt_ent}_cands_with_embeddings.tsv"

# === Graph Structures ===

# Adjacency matrix representing subclass relationships in the source ontology
src_Adjacence = f"{data_dir}/{src_ent}_adjacence.csv"

# Adjacency matrix representing subclass relationships in the target ontology
tgt_Adjacence = f"{data_dir}/{tgt_ent}_adjacence.csv"

# === Labels / Class Files ===

# JSON file mapping source entity URIs to labels/synonyms
src_class = f"{data_dir}/{src_ent}_classes.json"

# JSON file mapping target entity URIs to labels/synonyms
tgt_class = f"{data_dir}/{tgt_ent}_classes.json"

# === Training and Test Data ===

# Training file (e.g., positive/negative alignment pairs used to train the model)
train_file = f"{data_dir}/{task}_train.csv"
train_file_origin = f"{dataset_dir}/refs_equiv/train.tsv"

# Test set with gold-standard reference mappings (used for evaluation)
test_file = f"{dataset_dir}/refs_equiv/test.tsv"

# Candidate pairs for each test source entity (used for ranking metrics)
test_cands = f"{dataset_dir}/refs_equiv/test.cands.tsv"

# Reformatted candidate file derived from test.cands.tsv
cands_path = f"{data_dir}/{task}_cands.csv"

# === Prediction Output Files ===

# Final top-k predictions (e.g., from FAISS or ranking model)
all_predictions_path = f"{results_dir}/{task}_top_{k}_mappings.tsv"

# Top-1 predictions (most likely mapping per source entity)
top_1_predictions = f"{results_dir}/{task}_top_1_mappings.tsv"

# Raw predictions (all mappings after scoring/ranking)
prediction_path = f"{results_dir}/{task}_matching_results.tsv"

# Ranked predictions formatted for MRR and Hits@k evaluation
formatted_predictions_path = f"{results_dir}/{task}_formatted_predictions.tsv"

# File used specifically for MRR and Hits@k evaluation (top-200 candidates)
mappings_mrr = f"{results_dir}/{task}_top_200_mappings_mrr_hit.tsv"


# **GIN Architecture**

class GIN_mod(nn.Module):
    """
    Multi-layer GIN with optional linear pre-processing.
    Output dimension = input dimension (compatible with GatedCombination).
    """
    def __init__(self, dim_h, num_layers=1, num_linear_layers=1, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.num_linear_layers = num_linear_layers
        self.dropout = dropout

        # 🔹 Linear preprocessing (same idée que pour ton ancien GCN_mod / RGIT_mod)
        self.linears = nn.ModuleList()
        for _ in range(num_linear_layers):
            self.linears.append(Linear(dim_h, dim_h))
            self.linears.append(PReLU(num_parameters=dim_h))

        # 🔹 GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = Sequential(
                Linear(dim_h, dim_h),
                PReLU(),
                Linear(dim_h, dim_h),
            )
            conv = GINConv(mlp)
            self.convs.append(conv)
            self.bns.append(BatchNorm1d(dim_h))

    def forward(self, x, edge_index):
        # 1️⃣ Linear preprocessing
        for layer in self.linears:
            x = layer(x)

        # 2️⃣ Apply GIN layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


# **Gated Network Architecture**

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedCombination(nn.Module):
    def __init__(self, input_dim):
        """
        Initialize the GatedCombination module.

        Args:
            input_dim (int): Dimension of the input embeddings.
        """
        super(GatedCombination, self).__init__()
        # Gating networks for source and target embeddings
        self.gate_A_fc = nn.Linear(input_dim, input_dim)
        self.gate_B_fc = nn.Linear(input_dim, input_dim)

        # Final classification layer (binary output)
        self.fc = nn.Linear(1, 1)

    def euclidean_distance(self, a, b):
        """
        Compute the Euclidean (L2) distance between two tensors.

        Args:
            a (Tensor): Tensor of shape [batch_size, dim]
            b (Tensor): Tensor of shape [batch_size, dim]

        Returns:
            Tensor: Euclidean distance of shape [batch_size]
        """
        return torch.norm(a - b, p=2, dim=1)

    def forward(self, x1, x2, x3, x4, return_embeddings=False):
        """
        Forward pass of the model.

        Args:
            x1 (Tensor): Structural embedding of source concept
            x2 (Tensor): Semantic embedding of source concept
            x3 (Tensor): Structural embedding of target concept
            x4 (Tensor): Semantic embedding of target concept
            return_embeddings (bool): If True, return the fused embeddings (a, b)

        Returns:
            Tensor: Sigmoid score if return_embeddings is False,
                    otherwise tuple (a, b) of combined embeddings.
        """
        # Gated combination for source concept
        gate_values1 = torch.sigmoid(self.gate_A_fc(x1))
        a = x1 * gate_values1 + x2 * (1 - gate_values1)

        # Gated combination for target concept
        gate_values2 = torch.sigmoid(self.gate_B_fc(x3))
        b = x3 * gate_values2 + x4 * (1 - gate_values2)

        # Return the fused embeddings directly (used in inference/embedding export)
        if return_embeddings:
            return a, b

        # Compute Euclidean distance between source and target combined representations
        distance = self.euclidean_distance(a, b)

        # Pass the distance through a final sigmoid-activated classification layer
        out = torch.sigmoid(self.fc(distance.unsqueeze(1)))
        return out

# **Utility functions**

def adjacency_matrix_to_undirected_edge_index(adjacency_matrix):
    """
    Converts an adjacency matrix into an undirected edge index for use in graph-based neural networks.

    Args:
        adjacency_matrix: A 2D list or array representing the adjacency matrix of a graph.

    Returns:
        edge_index_undirected: A PyTorch tensor representing the undirected edges.
    """
    # Convert each element in the adjacency matrix to an integer (from boolean or float)
    adjacency_matrix = [[int(element) for element in sublist] for sublist in adjacency_matrix]

    # Convert the adjacency matrix into a PyTorch LongTensor (used for indexing)
    edge_index = torch.tensor(adjacency_matrix, dtype=torch.long)

    # Transpose the edge_index tensor so that rows represent edges in the form [source, target]
    edge_index = edge_index.t().contiguous()

    # Convert the directed edge_index into an undirected edge_index, meaning both directions are added (i.e., (i, j) and (j, i))
    edge_index_undirected = to_undirected(edge_index)

    return edge_index_undirected  # Return the undirected edge index

def build_indexed_dict(file_path):
    """
    Builds a dictionary with numeric indexes for each key from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        indexed_dict (dict): A new dictionary where each key from the JSON file is assigned a numeric index.
    """
    # Load the JSON file into a Python dictionary
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Create a new dictionary with numeric indexes as keys and the original JSON keys as values
    indexed_dict = {index: key for index, key in enumerate(data.keys())}

    return indexed_dict  # Return the newly created dictionary

def select_rows_by_index(embedding_vector, index_vector):
    """
    Select rows from an embedding vector using an index vector.

    Args:
        embedding_vector (torch.Tensor): 2D tensor representing the embedding vector with shape [num_rows, embedding_size].
        index_vector (torch.Tensor): 1D tensor representing the index vector.

    Returns:
        torch.Tensor: New tensor with selected rows from the embedding vector.
    """
    # Use torch.index_select to select the desired rows
    new_tensor = torch.index_select(embedding_vector, 0, index_vector)

    return new_tensor

def contrastive_loss(source_embeddings, target_embeddings, labels, margin=1.0):
    """
    Computes the contrastive loss, a type of loss function used to train models in tasks like matching or similarity learning.

    Args:
        source_embeddings (torch.Tensor): Embeddings of the source graphs, shape [batch_size, embedding_size].
        target_embeddings (torch.Tensor): Embeddings of the target graphs, shape [batch_size, embedding_size].
        labels (torch.Tensor): Binary labels indicating if the pairs are matched (1) or not (0), shape [batch_size].
        margin (float): Margin value for the contrastive loss. Defaults to 1.0.

    Returns:
        torch.Tensor: The contrastive loss value.
    """
    # Calculate the pairwise Euclidean distance between source and target embeddings
    distances = F.pairwise_distance(source_embeddings, target_embeddings)

    # Compute the contrastive loss
    loss = torch.mean(
        labels * 0.4 * distances.pow(2) +
        (1 - labels) * 0.4 * torch.max(torch.zeros_like(distances), margin - distances).pow(2)
    )

    return loss  # Return the computed contrastive loss

def save_gated_embeddings(gated_model, embeddings_src, x_src, embeddings_tgt, x_tgt,
                          indexed_dict_src, indexed_dict_tgt,
                          Emb_final_src, Emb_final_tgt):
    """
    Compute and save the final entity embeddings generated by the GatedCombination model
    for both source and target ontologies. Outputs include entity URIs and their final vectors.
    Measures and prints the execution time of the entire operation.
    """
    import pandas as pd
    import torch
    import time

    start_time = time.time()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gated_model = gated_model.to(device)
    gated_model.eval()

    # Move inputs to the same device
    embeddings_src = embeddings_src.to(device)
    x_src = x_src.to(device)
    embeddings_tgt = embeddings_tgt.to(device)
    x_tgt = x_tgt.to(device)

    with torch.no_grad():
        # === Source ontology ===
        gate_src = torch.sigmoid(gated_model.gate_A_fc(embeddings_src))
        final_src = embeddings_src * gate_src + x_src * (1 - gate_src)
        final_src = final_src.cpu().numpy()

        # === Target ontology ===
        gate_tgt = torch.sigmoid(gated_model.gate_B_fc(embeddings_tgt))
        final_tgt = embeddings_tgt * gate_tgt + x_tgt * (1 - gate_tgt)
        final_tgt = final_tgt.cpu().numpy()

    # Create DataFrames with Concept URI and embedding values
    df_src = pd.DataFrame(final_src)
    df_src.insert(0, "Concept", [indexed_dict_src[i] for i in range(len(df_src))])

    df_tgt = pd.DataFrame(final_tgt)
    df_tgt.insert(0, "Concept", [indexed_dict_tgt[i] for i in range(len(df_tgt))])

    # Save embeddings to file
    df_src.to_csv(Emb_final_src, sep='\t', index=False)
    df_tgt.to_csv(Emb_final_tgt, sep='\t', index=False)

    elapsed_time = time.time() - start_time
    print(f"✅ Gated embeddings saved:\n- Source: {Emb_final_src}\n- Target: {Emb_final_tgt}")
    print(f"⏱️ Execution time: {elapsed_time:.2f} seconds")

import pandas as pd


def filter_ignored_class(src_emb_path, tgt_emb_path, src_onto, tgt_onto, Emb_final_src_cl, Emb_final_tgt_cl):
    """
    Filters the source and target embedding files by removing concepts considered "ignored classes"
    (e.g., owl:Thing, deprecated entities, etc.) based on both source and target ontologies.
    """
    df_src = pd.read_csv(src_emb_path, sep='\t', dtype=str)
    df_tgt = pd.read_csv(tgt_emb_path, sep='\t', dtype=str)
    
    ignored_class_index = get_ignored_class_index(src_onto)
    ignored_class_index.update(get_ignored_class_index(tgt_onto))
    ignored_uris = set(str(uri).strip() for uri in ignored_class_index)

    df_src_cleaned = df_src[~df_src['Concept'].isin(ignored_uris)].reset_index(drop=True)
    df_tgt_cleaned = df_tgt[~df_tgt['Concept'].isin(ignored_uris)].reset_index(drop=True)

    df_src_cleaned.to_csv(Emb_final_src_cl, sep='\t', index=False)
    df_tgt_cleaned.to_csv(Emb_final_tgt_cl, sep='\t', index=False)

    return Emb_final_src_cl, Emb_final_tgt_cl


import pandas as pd

def format_ranked_predictions_for_mrr(reference_file, predicted_file, output_file):
    """
    Format predicted scores into ranked candidate lists per source entity,
    in a structure compatible with MRR and Hits@k evaluation.
    """
    reference_data = pd.read_csv(reference_file, sep='\t').values.tolist()

    predicted_data = pd.read_csv(predicted_file, sep="\t")
    predicted_data["Score"] = predicted_data["Score"].apply(
        lambda x: float(x.strip("[]")) if isinstance(x, str) else float(x)
    )

    score_lookup = {
        (row["SrcEntity"], row["TgtEntity"]): row["Score"]
        for _, row in predicted_data.iterrows()
    }

    ranking_results = []

    for src_entity, tgt_gold, tgt_cands in reference_data:
        try:
            raw = eval(tgt_cands)
            candidates = list(raw) if isinstance(raw, (list, tuple)) else []
        except:
            candidates = []

        scored_cands = [
            (cand, score_lookup.get((src_entity, cand), -1e9))
            for cand in candidates
        ]

        ranked = sorted(scored_cands, key=lambda x: x[1], reverse=True)

        ranking_results.append((src_entity, tgt_gold, ranked))

    pd.DataFrame(ranking_results, columns=["SrcEntity", "TgtEntity", "TgtCandidates"]).to_csv(
        output_file, sep="\t", index=False
    )

    print(f"✅ Ranked predictions saved for evaluation: {output_file}")
    return output_file

# **FAISS Similarity**

import pandas as pd
import numpy as np
import faiss
import time

def load_embeddings(src_emb_path, tgt_emb_path):
    df_src = pd.read_csv(src_emb_path, sep='\t')
    df_tgt = pd.read_csv(tgt_emb_path, sep='\t')
    uris_src = df_src["Concept"].values
    uris_tgt = df_tgt["Concept"].values
    src_vecs = df_src.drop(columns=["Concept"]).values.astype('float32')
    tgt_vecs = df_tgt.drop(columns=["Concept"]).values.astype('float32')
    return uris_src, uris_tgt, src_vecs, tgt_vecs

def save_results(uris_src, uris_tgt, indices, scores, output_file, top_k):
    rows = []
    for i, (ind_row, score_row) in enumerate(zip(indices, scores)):
        src_uri = uris_src[i]
        for j, tgt_idx in enumerate(ind_row):
            tgt_uri = uris_tgt[tgt_idx]
            score = score_row[j]
            rows.append((src_uri, tgt_uri, score))
    df_result = pd.DataFrame(rows, columns=["SrcEntity", "TgtEntity", "Score"])
    df_result.to_csv(output_file, sep='\t', index=False)
    print(f"Top-{top_k} FAISS similarity results saved to: {output_file}")

def topk_faiss_l2(src_emb_path, tgt_emb_path, top_k=15, output_file="topk_l2.tsv"):
    print("🔹 Using L2 (Euclidean) distance with FAISS")
    start = time.time()

    uris_src, uris_tgt, src_vecs, tgt_vecs = load_embeddings(src_emb_path, tgt_emb_path)

    dim = src_vecs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(tgt_vecs)

    distances, indices = index.search(src_vecs, top_k)

    similarity_scores = 1 / (1 + distances)

    save_results(uris_src, uris_tgt, indices, similarity_scores, output_file, top_k)

    print(f"⏱️ Execution time: {time.time() - start:.2f} seconds")

def select_best_candidates_per_src_with_margin(df, score_margin=0.01):
    selected_rows = []

    for src, group in df.groupby("SrcEntity"):
        group_sorted = group.sort_values(by="Score", ascending=False)
        best_score = group_sorted.iloc[0]["Score"]
        threshold = best_score * (1 - score_margin)

        close_matches = group_sorted[group_sorted["Score"] >= threshold]
        selected_rows.append(close_matches)

    result_df = pd.concat(selected_rows).reset_index(drop=True)
    print(f"🏆 Selected candidates within {(1 - score_margin) * 100:.1f}% of best score per SrcEntity: {len(result_df)} rows")
    return result_df

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def evaluate_predictions(
    pred_file, train_file, test_file,
    threshold=0.0, margin_ratio=0.997
):
    """
    Evaluate predicted mappings by applying filtering, thresholding, top-1 selection with margin,
    and computing precision, recall, and F1-score against the test set.
    """

    df = pd.read_csv(pred_file, sep='\t', encoding='utf-8')
    df.columns = df.columns.str.strip()
    
    train_df = pd.read_csv(train_file, sep="\t", encoding='utf-8')
    train_df.columns = train_df.columns.str.strip()
     
    test_df = pd.read_csv(test_file, sep="\t", encoding='utf-8')
    test_df.columns = test_df.columns.str.strip()
  
    train_uris = set(train_df['SrcEntity']) | set(train_df['TgtEntity'])
    test_uris = set(test_df['SrcEntity']) | set(test_df['TgtEntity'])
    uris_to_exclude = train_uris - test_uris
    df = df[~df['SrcEntity'].isin(uris_to_exclude) & ~df['TgtEntity'].isin(uris_to_exclude)]
    
    test_src_entities = set(test_df['SrcEntity'])
    df = df[df['SrcEntity'].isin(test_src_entities)]
    
    df.to_csv(all_predictions_path, sep='\t', index=False)
    
    df_topk = select_best_candidates_per_src_with_margin(df, score_margin=score_margin)

    df_topk.to_csv(prediction_path, sep='\t', index=False)
   
    print(f"   ➤ Mappings file:   {prediction_path}")

    preds = EntityMapping.read_table_mappings(prediction_path)
    refs = ReferenceMapping.read_table_mappings(test_file)

    preds_set = {p.to_tuple() for p in preds}
    refs_set = {r.to_tuple() for r in refs}
    correct = len(preds_set & refs_set)

    results = AlignmentEvaluator.f1(preds, refs)

    print("\n🎯 Evaluation Summary:")
    print(f"   - Correct mappings:     {correct}")
    print(f"   - Total predictions:    {len(preds)}")
    print(f"   - Total references:     {len(refs)}")
    print(f"📊 Precision:              {results['P']:.3f}")
    print(f"📊 Recall:                 {results['R']:.3f}")
    print(f"📊 F1-score:               {results['F1']:.3f}\n")

    return prediction_path, results, correct


import pandas as pd
from collections import defaultdict

def evaluate_topk(topk_file, train_file, test_file, k=1, threshold=0.0):
    """
    Evaluate Top-K predictions using Precision, Recall, and F1-score.
    """
    df = pd.read_csv(topk_file, sep='\t', encoding='utf-8')
    df.columns = df.columns.str.strip()
    
    train_df = pd.read_csv(train_file, sep="\t", encoding='utf-8')
    train_df.columns = train_df.columns.str.strip()
     
    test_df = pd.read_csv(test_file, sep="\t", encoding='utf-8')
    test_df.columns = test_df.columns.str.strip()

    train_uris = set(train_df['SrcEntity']) | set(train_df['TgtEntity'])
    test_uris = set(test_df['SrcEntity']) | set(test_df['TgtEntity'])
    uris_to_exclude = train_uris - test_uris
    df = df[~(df['SrcEntity'].isin(uris_to_exclude) | df['TgtEntity'].isin(uris_to_exclude))].reset_index(drop=True)

    src_entities_test = set(test_df['SrcEntity'])
    df = df[df['SrcEntity'].isin(src_entities_test)].reset_index(drop=True)

    df['Score'] = df['Score'].apply(lambda x: float(x.strip("[]")) if isinstance(x, str) else float(x))
    df_sorted = df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    matched_sources = set()
    matched_targets = set()
    result = []

    for _, row in df_sorted.iterrows():
        src, tgt, score = row['SrcEntity'], row['TgtEntity'], row['Score']
        if src not in matched_sources and tgt not in matched_targets and score >= threshold:
            result.append((src, tgt, score))
            matched_sources.add(src)
            matched_targets.add(tgt)

    matching_results_df = pd.DataFrame(result, columns=['SrcEntity', 'TgtEntity', 'Score'])
    matching_results_df.to_csv(prediction_path, sep='\t', index=False)

    ref_dict = defaultdict(set)
    for _, row in test_df.iterrows():
        ref_dict[row['SrcEntity']].add(row['TgtEntity'])

    matching_results_df['Score'] = matching_results_df['Score'].astype(float)
    topk_df = matching_results_df.sort_values(by='Score', ascending=False).groupby('SrcEntity').head(k)

    total_tp = total_pred = total_ref = 0

    for src, group in topk_df.groupby('SrcEntity'):
        predicted = set(group['TgtEntity'])
        true = ref_dict.get(src, set())
        tp = len(predicted & true)
        total_tp += tp
        total_pred += len(predicted)
        total_ref += len(true)

    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_ref if total_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if precision + recall > 0 else 0.0

    print(f"📊 Precision@{k}:            {precision:.3f}")
    print(f"📊 Recall@{k}:               {recall:.3f}")
    print(f"📊 F1@{k}:                   {f1:.3f}\n")

    return {
        f'Precision@{k}': round(precision, 3),
        f'Recall@{k}': round(recall, 3),
        f'F1@{k}': round(f1, 3)
    }


# Main Code

print("Reading semantic concepts embeddings provided by the Concept Features Encoder ...")

df_embbedings_src = pd.read_csv(src_Emb, index_col=0)
numpy_array = df_embbedings_src.to_numpy()
x_src = torch.FloatTensor(numpy_array)

df_embbedings_tgt = pd.read_csv(tgt_Emb, index_col=0)
numpy_array = df_embbedings_tgt.to_numpy()
x_tgt = torch.FloatTensor(numpy_array)


# Reading adjacency Matrix

df_ma1 = pd.read_csv(src_Adjacence, index_col=0)
ma1 = df_ma1.values.tolist()

df_ma2 = pd.read_csv(tgt_Adjacence, index_col=0)
ma2 = df_ma2.values.tolist()

edge_src = adjacency_matrix_to_undirected_edge_index(ma1)
edge_tgt = adjacency_matrix_to_undirected_edge_index(ma2)

print("GIN Training...")

def train_model_gnn(model, x_src, edge_src, x_tgt, edge_tgt,
                    tensor_term1, tensor_term2, tensor_score,
                    learning_rate, weight_decay_value, num_epochs, print_interval=10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    x_tgt = x_tgt.to(device)
    edge_tgt = edge_tgt.to(device)
    x_src = x_src.to(device)
    edge_src = edge_src.to(device)
    tensor_term1 = tensor_term1.to(device)
    tensor_term2 = tensor_term2.to(device)
    tensor_score = tensor_score.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)

    train_losses = []

    start_time = time.time()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        out1 = model(x_src, edge_src)
        out2 = model(x_tgt, edge_tgt)

        src_embeddings = select_rows_by_index(out1, tensor_term1)
        tgt_embeddings = select_rows_by_index(out2, tensor_term2)

        loss = contrastive_loss(src_embeddings, tgt_embeddings, tensor_score)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (epoch + 1) % print_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item()}")

    end_time = time.time()

    plt.semilogy(range(1, num_epochs + 1), train_losses, label="Training Loss", marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    training_time = end_time - start_time
    print(f"Training complete! Total training time: {training_time:.2f} seconds")

    return model

GIN_model = GIN_mod(x_tgt.shape[1], num_layers=1, num_linear_layers=1, dropout=0.0)

df_embbedings = pd.read_csv(train_file, index_col=0)

tensor_term1 = df_embbedings['SrcEntity'].values.astype(int)
tensor_term2 = df_embbedings['TgtEntity'].values.astype(int)
tensor_score = df_embbedings['Score'].values.astype(float)

tensor_term1_o = torch.from_numpy(tensor_term1).type(torch.LongTensor)
tensor_term2_o = torch.from_numpy(tensor_term2).type(torch.LongTensor)
tensor_score_o = torch.from_numpy(tensor_score).type(torch.FloatTensor)

trained_model_gnn = train_model_gnn(
    model=GIN_model,
    x_src=x_src,
    edge_src=edge_src,
    x_tgt=x_tgt,
    edge_tgt=edge_tgt,
    tensor_term1=tensor_term1_o,
    tensor_term2=tensor_term2_o,
    tensor_score=tensor_score_o,
    learning_rate=0.0001,
    weight_decay_value=1e-4,
    num_epochs=1000,
    print_interval=10
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GIN_model.to(device)

x_tgt = x_tgt.to(device)
edge_tgt = edge_tgt.to(device)
x_src = x_src.to(device)
edge_src = edge_src.to(device)

GIN_model.eval()

with torch.no_grad():
    embeddings_tgt = GIN_model(x_tgt, edge_tgt)
    embeddings_src = GIN_model(x_src, edge_src)

embeddings_tgt = embeddings_tgt.detach().cpu()
embeddings_src = embeddings_src.detach().cpu()

df_embeddings = pd.read_csv(train_file, index_col=0)

tensor_term1 = df_embeddings['SrcEntity'].values.astype(int)
tensor_term2 = df_embeddings['TgtEntity'].values.astype(int)
tensor_score = df_embeddings['Score'].values.astype(float)

tensor_term1_train, tensor_term1_val, tensor_term2_train, tensor_term2_val, tensor_score_train, tensor_score_val = train_test_split(
    tensor_term1, tensor_term2, tensor_score, test_size=0.3, random_state=42
)

tensor_term1_train = torch.from_numpy(tensor_term1_train).type(torch.LongTensor)
tensor_term2_train = torch.from_numpy(tensor_term2_train).type(torch.LongTensor)
tensor_score_train = torch.from_numpy(tensor_score_train).type(torch.FloatTensor)

tensor_term1_val = torch.from_numpy(tensor_term1_val).type(torch.LongTensor)
tensor_term2_val = torch.from_numpy(tensor_term2_val).type(torch.LongTensor)
tensor_score_val = torch.from_numpy(tensor_score_val).type(torch.FloatTensor)

x_tgt = x_tgt.cpu()
x_src = x_src.cpu()

X1_train = select_rows_by_index(embeddings_src, tensor_term1_train)
X2_train = select_rows_by_index(x_src, tensor_term1_train)
X3_train = select_rows_by_index(embeddings_tgt, tensor_term2_train)
X4_train = select_rows_by_index(x_tgt, tensor_term2_train)

X1_val = select_rows_by_index(embeddings_src, tensor_term1_val)
X2_val = select_rows_by_index(x_src, tensor_term1_val)
X3_val = select_rows_by_index(embeddings_tgt, tensor_term2_val)
X4_val = select_rows_by_index(x_tgt, tensor_term2_val)

print("Gated Network Training...")

def train_gated_combination_model(X1_t, X2_t, X3_t, X4_t, tensor_score_o,
                                  X1_val, X2_val, X3_val, X4_val, tensor_score_val,
                                  epochs=120, batch_size=32, learning_rate=0.001, weight_decay=1e-5):

    train_dataset = TensorDataset(X1_t, X2_t, X3_t, X4_t, tensor_score_o)
    val_dataset = TensorDataset(X1_val, X2_val, X3_val, X4_val, tensor_score_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedCombination(X1_t.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    train_losses, val_losses = [], []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        y_true_train, y_pred_train = [], []

        for batch_X1, batch_X2, batch_X3, batch_X4, batch_y in train_loader:
            batch_X1, batch_X2, batch_X3, batch_X4, batch_y = (
                batch_X1.to(device),
                batch_X2.to(device),
                batch_X3.to(device),
                batch_X4.to(device),
                batch_y.to(device),
            )
            optimizer.zero_grad()

            outputs = model(batch_X1, batch_X2, batch_X3, batch_X4)

            loss = F.binary_cross_entropy(outputs, batch_y.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            y_true_train.extend(batch_y.cpu().numpy())
            y_pred_train.extend((outputs > 0.5).float().cpu().numpy())

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        train_f1 = f1_score(y_true_train, y_pred_train)

        model.eval()
        total_val_loss = 0.0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for batch_X1, batch_X2, batch_X3, batch_X4, batch_y in val_loader:
                batch_X1, batch_X2, batch_X3, batch_X4, batch_y = (
                    batch_X1.to(device),
                    batch_X2.to(device),
                    batch_X3.to(device),
                    batch_X4.to(device),
                    batch_y.to(device),
                )
                outputs = model(batch_X1, batch_X2, batch_X3, batch_X4)

                val_loss = F.binary_cross_entropy(outputs, batch_y.unsqueeze(1).float())
                total_val_loss += val_loss.item()

                y_true_val.extend(batch_y.cpu().numpy())
                y_pred_val.extend((outputs > 0.5).float().cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_f1 = f1_score(y_true_val, y_pred_val)

        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] Training Loss: {train_loss:.4f}, F1 Score: {train_f1:.4f} | "
              f"Validation Loss: {avg_val_loss:.4f}, F1 Score: {val_f1:.4f}")

    end_time = time.time()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print(f"Training complete! Total time: {end_time - start_time:.2f} seconds")
    return model

trained_model = train_gated_combination_model(
    X1_train,
    X2_train,
    X3_train,
    X4_train,
    tensor_score_train,
    X1_val,
    X2_val,
    X3_val,
    X4_val,
    tensor_score_val,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    weight_decay=1e-4
)

print("Generate Embeddings...")

indexed_dict_src = build_indexed_dict(src_class)
indexed_dict_tgt = build_indexed_dict(tgt_class)

save_gated_embeddings(
    gated_model=trained_model,
    embeddings_src=embeddings_src,
    x_src=x_src,
    embeddings_tgt=embeddings_tgt,
    x_tgt=x_tgt,
    indexed_dict_src=indexed_dict_src,
    indexed_dict_tgt=indexed_dict_tgt,
    Emb_final_src=Emb_final_src,
    Emb_final_tgt=Emb_final_tgt
)

src_file, tgt_file = filter_ignored_class(
    src_emb_path=Emb_final_src,
    tgt_emb_path=Emb_final_tgt,
    src_onto=src_onto,
    tgt_onto=tgt_onto,
    Emb_final_src_cl=Emb_final_src_cl,
    Emb_final_tgt_cl=Emb_final_tgt_cl
)

print("Generate mappings...")
topk_faiss_l2(
    src_emb_path=Emb_final_src_cl,
    tgt_emb_path=Emb_final_tgt_cl,
    top_k=k,
    output_file=all_predictions_path
)

print("Calculating global evaluation metrics (precision, recall and F1-score) ....")

output_file, metrics, correct = evaluate_predictions(
    pred_file=all_predictions_path,
    train_file=train_file_origin,
    test_file=test_file,
)

print("Calculating ranking metrics (precision@k, recall@k and F1-score@k) ....")

topk_faiss_l2(
    src_emb_path=Emb_final_src_cl,
    tgt_emb_path=Emb_final_tgt_cl,
    top_k=1,
    output_file=top_1_predictions
)

results_topk = evaluate_topk(
    topk_file=top_1_predictions,
    train_file=train_file_origin,
    test_file=test_file
)

print("Calculating local ranking metrics (MRR and Hit@k) ....")

import pandas as pd

src_emb_path = Emb_final_src_cl
tgt_emb_path = Emb_final_tgt_cl

df_cands = pd.read_csv(cands_path)
src_emb_df = pd.read_csv(src_emb_path, sep="\t")
tgt_emb_df = pd.read_csv(tgt_emb_path, sep="\t")

unique_src_df = pd.DataFrame(df_cands["SrcEntity"].unique(), columns=["Concept"])
unique_tgt_df = pd.DataFrame(df_cands["TgtEntity"].unique(), columns=["Concept"])

merged_src_df = pd.merge(unique_src_df, src_emb_df, on="Concept", how="left")
merged_tgt_df = pd.merge(unique_tgt_df, tgt_emb_df, on="Concept", how="left")

merged_src_df.to_csv(src_rank_emb, sep="\t", index=False)
merged_tgt_df.to_csv(tgt_rank_emb, sep="\t", index=False)

topk_faiss_l2(
    src_emb_path=src_rank_emb,
    tgt_emb_path=tgt_rank_emb,
    top_k=200,
    output_file=mappings_mrr
)

format_ranked_predictions_for_mrr(
    reference_file=test_cands,
    predicted_file=mappings_mrr,
    output_file=formatted_predictions_path
)

results = ranking_eval(formatted_predictions_path, Ks=[1, 5, 10])

print("Ranking Evaluation Results at K=1, 5, and 10:")
print(results)
