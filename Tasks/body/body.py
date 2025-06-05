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
from torch_geometric.nn.dense.linear import Linear

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
src_ent = "snomed.body"

# Define the target ontology name
tgt_ent = "fma.body"

# Define the task name for this ontology matching process
task = "body"               # Used to name intermediate and output files

# Set the value of top-k candidates to consider 
k = 2
# Score margin used for relaxed top-1 selection
score_margin = 0.005           # Minimum score difference required between top-1 and runner-up

print(f"Matching {src_ent}.owl and {tgt_ent}.owl:")

# Sets the relative path to the root directory of the BioGITOM project.
# This path points three levels up from the current working directory
# and navigates to the 'biogitom' folder. Adjust this path if the 
# directory structure changes or the script is executed from a different location.
dir = "../biogitom"

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

# Final processed embeddings (e.g., after GIT/Gated network)
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
# It contains the same mappings (SrcEntity, TgtEntity, CandidateTgtEntities),
# but in a structure optimized for scoring (e.g., using FAISS or embedding-based similarity).
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


# **GIT Architecture**


# RGIT class definition which inherits from PyTorch Geometric's MessagePassing class
class RGIT(MessagePassing):

    _alpha: OptTensor  # Define _alpha as an optional tensor for storing attention weights

    def __init__(
        self,
        nn: Callable,  # Neural network to be used in the final layer of the GNN
        in_channels: Union[int, Tuple[int, int]],  # Input dimension, can be a single or pair of integers
        out_channels: int,  # Output dimension of the GNN
        eps: float = 0.,  # GIN parameter: epsilon for GIN aggregation
        train_eps: bool = False,  # GIN parameter: whether epsilon should be learnable
        heads: int = 1,  # Transformer parameter: number of attention heads
        dropout: float = 0.,  # Dropout rate for attention weights
        edge_dim: Optional[int] = None,  # Dimension for edge attributes (optional)
        bias: bool = True,  # Whether to use bias in linear layers
        root_weight: bool = True,  # GIN parameter: whether to apply root weight in aggregation
        **kwargs,  # Additional arguments passed to the parent class
    ):
        # Set the aggregation type to 'add' and initialize the parent class with node_dim=0
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        # Initialize input/output dimensions, neural network, and GIN/transformer parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn  # Neural network used by the GNN
        self.initial_eps = eps  # Initial value of epsilon for GIN

        # Set epsilon to be learnable or fixed
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))  # Learnable epsilon
        else:
            self.register_buffer('eps', torch.empty(1))  # Non-learnable epsilon (fixed)

        # Initialize transformer-related parameters
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None  # Placeholder for attention weights

        # Handle case where in_channels is a single integer or a tuple
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # Define the linear layers for key, query, and value for the transformer mechanism
        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)

        # Define linear transformation for edge embeddings if provided
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        # Reset all parameters to their initial values
        self.reset_parameters()

    # Function to reset model parameters
    def reset_parameters(self):
        super().reset_parameters()  # Call parent class reset method
        self.lin_key.reset_parameters()  # Reset key linear layer
        self.lin_query.reset_parameters()  # Reset query linear layer
        self.lin_value.reset_parameters()  # Reset value linear layer
        if self.edge_dim:
            self.lin_edge.reset_parameters()  # Reset edge linear layer if used
        reset(self.nn)  # Reset the neural network provided
        self.eps.data.fill_(self.initial_eps)  # Initialize epsilon with the starting value

    # Forward function defining how the input data flows through the model
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        # Unpack number of heads and output channels
        H, C = self.heads, self.out_channels

        # If x is a tensor, treat it as a pair of tensors (source and target embeddings)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # Extract source node embeddings
        x_t = x[0]

        # Apply linear transformations and reshape query, key, and value for multi-head attention
        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # Propagate messages through the graph using the propagate function
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        # Retrieve attention weights and reset them
        alpha = self._alpha
        self._alpha = None  # Reset _alpha after use
        out = out.mean(dim=1)  # Take the mean over all attention heads

        # Apply GIN aggregation by adding epsilon-scaled original node embeddings
        out = out + (1 + self.eps) * x_t
        return self.nn(out)  # Pass through the neural network

    # Message passing function which calculates attention and combines messages
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # If edge attributes are used, apply linear transformation and add them to the key
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr

        # Calculate attention (alpha) using the dot product between query and key
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)  # Apply softmax to normalize attention
        self._alpha = alpha  # Store attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # Apply dropout

        # Calculate the output message by applying attention to the value
        out = value_j
        if edge_attr is not None:
            out = out + edge_attr  # Add edge embeddings to the output if present
        out = out * alpha.view(-1, self.heads, 1)  # Scale by attention weights
        return out

    # String representation function for debugging or printing
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

# Define the RGIT_mod class, a multi-layer GNN that uses both RGIT and linear layers
class RGIT_mod(torch.nn.Module):
    """Multi-layer RGIT with optional linear layers"""

    # Initialize the model with hidden dimension, number of RGIT layers, and number of linear layers
    def __init__(self, dim_h, num_layers, num_linear_layers=1):
        super(RGIT_mod, self).__init__()
        self.num_layers = num_layers  # Number of RGIT layers
        self.num_linear_layers = num_linear_layers  # Number of linear layers
        self.linears = torch.nn.ModuleList()  # List to store linear layers
        self.rgit_layers = torch.nn.ModuleList()  # List to store RGIT layers

        # Create a list of Linear and PReLU layers (for encoding entity names)
        for _ in range(num_linear_layers):
            self.linears.append(Linear(dim_h, dim_h))  # Linear transformation layer
            self.linears.append(PReLU(num_parameters=dim_h))  # Parametric ReLU activation function

        # Create a list of RGIT layers
        for _ in range(num_layers):
            self.rgit_layers.append(RGIT(  # Each RGIT layer contains a small MLP with Linear and PReLU
                Sequential(Linear(dim_h, dim_h), PReLU(num_parameters=dim_h),
                           Linear(dim_h, dim_h), PReLU(num_parameters=dim_h)), dim_h, dim_h))

    # Forward pass through the model
    def forward(self, x, edge_index):
        # Apply the linear layers first to the input
        for layer in self.linears:
            x = layer(x)

        # Then apply the RGIT layers for message passing
        for layer in self.rgit_layers:
            x = layer(x, edge_index)

        return x  # Return the final node embeddings after all layers

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

    # Compute the contrastive loss:
    # - For matched pairs (label == 1), the loss is the squared distance between embeddings.
    # - For non-matched pairs (label == 0), the loss is based on how far apart the embeddings are,
    #   but penalizes them only if the distance is less than the margin.
    loss = torch.mean(
        labels * 0.4 * distances.pow(2) +  # For positive pairs, minimize the distance (squared)
        (1 - labels) * 0.4 * torch.max(torch.zeros_like(distances), margin - distances).pow(2)  # For negative pairs, maximize the distance (up to the margin)
    )

    return loss  # Return the computed contrastive loss

def save_gated_embeddings(gated_model, embeddings_src, x_src, embeddings_tgt, x_tgt,
                          indexed_dict_src, indexed_dict_tgt,
                          Emb_final_src, Emb_final_tgt):
    """
    Compute and save the final entity embeddings generated by the GatedCombination model
    for both source and target ontologies. Outputs include entity URIs and their final vectors.
    Measures and prints the execution time of the entire operation.

    Args:
        gated_model (nn.Module): The trained GatedCombination model.
        embeddings_src (Tensor): Structural embeddings for the source ontology.
        x_src (Tensor): Semantic embeddings for the source ontology.
        embeddings_tgt (Tensor): Structural embeddings for the target ontology.
        x_tgt (Tensor): Semantic embeddings for the target ontology.
        indexed_dict_src (dict): Index-to-URI mapping for the source ontology.
        indexed_dict_tgt (dict): Index-to-URI mapping for the target ontology.
        output_file_src (str): Path to save source embeddings (TSV).
        output_file_tgt (str): Path to save target embeddings (TSV).
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

    Args:
        src_emb_path (str): Path to the TSV file containing source embeddings with 'Concept' column.
        tgt_emb_path (str): Path to the TSV file containing target embeddings with 'Concept' column.
        src_onto (Ontology): Source ontology object loaded with DeepOnto.
        tgt_onto (Ontology): Target ontology object loaded with DeepOnto.

    Returns:
        (str, str): Paths to the cleaned source and target embedding files.
    """

    # === Load the embedding files ===
    df_src = pd.read_csv(src_emb_path, sep='\t', dtype=str)
  
    df_tgt = pd.read_csv(tgt_emb_path, sep='\t', dtype=str)

    # === Step 1: Retrieve ignored classes from both ontologies ===
    ignored_class_index = get_ignored_class_index(src_onto)  # e.g., owl:Thing, non-usable classes
    ignored_class_index.update(get_ignored_class_index(tgt_onto))  # Merge with target ontology's ignored classes
    ignored_uris = set(str(uri).strip() for uri in ignored_class_index)

    # === Step 2: Remove rows where the 'Concept' column matches ignored URIs ===
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

    Args:
        reference_file (str): Path to the reference test candidate file (e.g., 'test.cands.tsv'),
                              with columns: SrcEntity, TgtEntity (gold), CandidateTgtEntities (list)
        predicted_file (str): Path to the flat prediction file with columns: SrcEntity, TgtEntity, Score
        output_file (str): Path to save the formatted ranked candidates (for evaluation)

    Returns:
        str: Path to the formatted output file (TSV with columns: SrcEntity, TgtEntity, TgtCandidates)
    """

    # Load reference candidates (test.cands.tsv format)
    reference_data = pd.read_csv(reference_file, sep='\t').values.tolist()

    # Load predictions and ensure scores are floats
    predicted_data = pd.read_csv(predicted_file, sep="\t")
    predicted_data["Score"] = predicted_data["Score"].apply(
        lambda x: float(x.strip("[]")) if isinstance(x, str) else float(x)
    )

    # Build a dictionary for quick score lookup
    score_lookup = {
        (row["SrcEntity"], row["TgtEntity"]): row["Score"]
        for _, row in predicted_data.iterrows()
    }

    ranking_results = []

    # For each source entity, rank its candidate targets by predicted score
    for src_entity, tgt_gold, tgt_cands in reference_data:
        try:
            raw = eval(tgt_cands)
            candidates = list(raw) if isinstance(raw, (list, tuple)) else []
        except:
            candidates = []

        # Score each candidate (default to very low score if missing)
        scored_cands = [
            (cand, score_lookup.get((src_entity, cand), -1e9))
            for cand in candidates
        ]

        # Sort by score descending
        ranked = sorted(scored_cands, key=lambda x: x[1], reverse=True)

        # Append the ranking result
        ranking_results.append((src_entity, tgt_gold, ranked))

    # Save to TSV file (used later for MRR / Hits@k computation)
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
    """
    Load the embeddings for the source and target ontologies from TSV files.

    Args:
        src_emb_path (str): Path to the source embeddings file.
        tgt_emb_path (str): Path to the target embeddings file.

    Returns:
        uris_src (np.ndarray): URIs of source entities.
        uris_tgt (np.ndarray): URIs of target entities.
        src_vecs (np.ndarray): Embedding vectors for source entities.
        tgt_vecs (np.ndarray): Embedding vectors for target entities.
    """
    df_src = pd.read_csv(src_emb_path, sep='\t')  # Read source embeddings
    df_tgt = pd.read_csv(tgt_emb_path, sep='\t')  # Read target embeddings
    uris_src = df_src["Concept"].values           # Extract source URIs
    uris_tgt = df_tgt["Concept"].values           # Extract target URIs
    src_vecs = df_src.drop(columns=["Concept"]).values.astype('float32')  # Extract and convert source vectors
    tgt_vecs = df_tgt.drop(columns=["Concept"]).values.astype('float32')  # Extract and convert target vectors
    return uris_src, uris_tgt, src_vecs, tgt_vecs

def save_results(uris_src, uris_tgt, indices, scores, output_file, top_k):
    """
    Save the top-k mapping results to a TSV file.

    Args:
        uris_src (np.ndarray): URIs of source entities.
        uris_tgt (np.ndarray): URIs of target entities.
        indices (np.ndarray): Indices of top-k matched target entities.
        scores (np.ndarray): Corresponding similarity scores.
        output_file (str): Output TSV file path.
        top_k (int): Number of top matches per source entity.
    """
    rows = []
    for i, (ind_row, score_row) in enumerate(zip(indices, scores)):
        src_uri = uris_src[i]
        for j, tgt_idx in enumerate(ind_row):
            tgt_uri = uris_tgt[tgt_idx]
            score = score_row[j]
            rows.append((src_uri, tgt_uri, score))  # Store each top-k match
    df_result = pd.DataFrame(rows, columns=["SrcEntity", "TgtEntity", "Score"])
    df_result.to_csv(output_file, sep='\t', index=False)  # Save to file
    print(f"Top-{top_k} FAISS similarity results saved to: {output_file}")

def topk_faiss_l2(src_emb_path, tgt_emb_path, top_k=15, output_file="topk_l2.tsv"):
    """
    Compute the top-k most similar target entities for each source entity using FAISS with L2 distance.

    Args:
        src_emb_path (str): Path to the source embeddings file.
        tgt_emb_path (str): Path to the target embeddings file.
        top_k (int): Number of top matches to retrieve.
        output_file (str): Path to save the top-k results.
    """
    print("🔹 Using L2 (Euclidean) distance with FAISS")
    start = time.time()  # Start timing

    # Load embeddings
    uris_src, uris_tgt, src_vecs, tgt_vecs = load_embeddings(src_emb_path, tgt_emb_path)

    # Build FAISS index using L2 distance
    dim = src_vecs.shape[1]
    index = faiss.IndexFlatL2(dim)  # Create FAISS index for L2 distance
    index.add(tgt_vecs)             # Add target vectors to index

    # Perform nearest neighbor search
    distances, indices = index.search(src_vecs, top_k)

    # Convert distances to similarity scores (optional: inverse of distance)
    similarity_scores = 1 / (1 + distances)

    # Save the results
    save_results(uris_src, uris_tgt, indices, similarity_scores, output_file, top_k)

    # Display execution time
    print(f"⏱️ Execution time: {time.time() - start:.2f} seconds")

"""# **Mappings Evaluation Functions**

# **Precision, Recall, F1**

### Evaluation Strategy and Filtering Justification

### Filtering Justification

To ensure that evaluation metrics such as Precision, Recall, and F1-score accurately reflect the model's true performance, we apply two carefully designed filtering steps in the evaluate_predictions function. These filters are specifically crafted to focus the evaluation on verifiable predictions without unnecessarily penalizing the model or distorting the top-k candidate structure.

#### 1. Filtering Out Training-Only Entities

We remove all predicted mappings that involve entities (either source or target) that are present exclusively in the training set and do not appear in the test set.

This is a crucial step because:

- In datasets like Bio-ML, entities often appear in both training and test sets but are aligned with different targets. Filtering based solely on mappings would eliminate valuable generalization examples.

- Predictions involving entities that are not part of the test set cannot be evaluated and could unfairly skew precision and recall.

Importantly, we do not remove all mappings seen during training. Instead, we only discard those that involve non-testable entities. This distinction ensures that we retain valuable mappings between shared entities that can still play a meaningful role during prediction and ranking.

#### 2. Filtering on `SrcEntity` present in the test set

The second step keeps only the predictions where the `SrcEntity` is included in the test reference set.

- This eliminates **non-evaluable false positives**, i.e., predicted mappings for source entities that do not appear in the test set and therefore have no ground-truth correspondences. Including such predictions **unfairly penalizes precision and F1-score**, even though they are technically not verifiable errors.

- It focuses the evaluation on entities with defined ground-truth mappings, which is critical for computing metrics such as :

$P_{\text{test}} = \frac{|\mathcal{M}_{\text{out}} \cap \mathcal{M}_{\text{test}}|}{|\mathcal{M}_{\text{out}} \setminus (\mathcal{M}_{\text{ref}} \setminus \mathcal{M}_{\text{test}})|}$.

---

### 📌 Why This Works

Let’s illustrate the rationale using a simplified **Bio-ML** scenario:

| Dataset | Mappings                                      | Entities       |
|---------|-----------------------------------------------|----------------|
| Train   | (A:Cancer, B:Melanoma), (C:Radiation, D:Therapy) | A, B, C, D     |
| Test    | (A:Cancer, E:Carcinoma), (F:Skin, B:Melanoma)    | A, B, E, F     |

After applying our filtering strategy:

- **Removed**: (C:Radiation, D:Therapy) → both `C` and `D` are exclusive to train
- **Kept**: (A:Cancer, B:Melanoma) → `A` and `B` also appear in test

This means we preserve mappings that involve entities **shared between train and test**, even if the specific mapping was seen during training and is not part of the test reference set.

---

### ✅ Key Advantages

- **Preserves semantic context**  
  Keeping *(A, B)* helps the model calibrate similarity scores for other test mappings involving `A` or `B`, such as *(A, E)* or *(F, B)*.

- **Maintains fair competition in ranking**  
  Removing all training mappings would delete useful distractors (e.g., *(A, B)*), which could **artificially promote** weaker candidates (e.g., *(A, E)*) to the top rank, simply due to lack of strong alternatives.

---

This strategy strikes a balance between **evaluation fairness** and **preservation of top-k integrity**, ensuring that the ranking dynamics remain realistic and reflective of the model’s true generalization ability.
"""

def select_best_candidates_per_src_with_margin(df, score_margin=0.01):
    """
    For each SrcEntity, retain all candidate mappings whose similarity score is
    within 99% of the best score (default margin = 0.01).

    Args:
        df (pd.DataFrame): DataFrame containing columns ['SrcEntity', 'TgtEntity', 'Score'].
        score_margin (float): Score margin. 0.01 means keep scores ≥ 99% of best score.

    Returns:
        pd.DataFrame: Filtered DataFrame with multiple high-quality candidates per SrcEntity.
    """
    selected_rows = []

    for src, group in df.groupby("SrcEntity"):
        group_sorted = group.sort_values(by="Score", ascending=False)
        best_score = group_sorted.iloc[0]["Score"]
        threshold = best_score * (1 - score_margin)

        # Keep all target entities with score >= threshold
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

    # === Step 1: Load input files ===
    df = pd.read_csv(pred_file, sep='\t', encoding='utf-8')
    df.columns = df.columns.str.strip()
    
    train_df = pd.read_csv(train_file, sep="\t", encoding='utf-8')
    train_df.columns = train_df.columns.str.strip()
     
    test_df = pd.read_csv(test_file, sep="\t", encoding='utf-8')
    test_df.columns = test_df.columns.str.strip()

    # ✅ Step 2: Remove entities that appear only in the training set
    train_uris = set(train_df['SrcEntity']) | set(train_df['TgtEntity'])
    test_uris = set(test_df['SrcEntity']) | set(test_df['TgtEntity'])
    uris_to_exclude = train_uris - test_uris
    df = df[~df['SrcEntity'].isin(uris_to_exclude) & ~df['TgtEntity'].isin(uris_to_exclude)]
    
    # Step 3: Keep only predictions where SrcEntity is part of the test set
    test_src_entities = set(test_df['SrcEntity'])
    df = df[df['SrcEntity'].isin(test_src_entities)]
    
    # Step 5: Save filtered predictions to file
    df.to_csv(all_predictions_path, sep='\t', index=False)
    
    # Step 6: Select best predictions per SrcEntity using a relaxed top-1 margin
    df_topk = select_best_candidates_per_src_with_margin(df, score_margin=score_margin)

    # Step 7: Save the top-1 filtered predictions
    df_topk.to_csv(prediction_path, sep='\t', index=False)
   
    print(f"   ➤ Mappings file:   {prediction_path}")

    # === Step 8: Evaluate against reference mappings
    preds = EntityMapping.read_table_mappings(prediction_path)
    refs = ReferenceMapping.read_table_mappings(test_file)

    preds_set = {p.to_tuple() for p in preds}
    refs_set = {r.to_tuple() for r in refs}
    correct = len(preds_set & refs_set)

    results = AlignmentEvaluator.f1(preds, refs)

    # === Step 9: Print evaluation metrics
    print("\n🎯 Evaluation Summary:")
    print(f"   - Correct mappings:     {correct}")
    print(f"   - Total predictions:    {len(preds)}")
    print(f"   - Total references:     {len(refs)}")
    print(f"📊 Precision:              {results['P']:.3f}")
    print(f"📊 Recall:                 {results['R']:.3f}")
    print(f"📊 F1-score:               {results['F1']:.3f}\n")

    return prediction_path, results, correct


"""# **Precision@k, Recall@k, F1@k**"""

import pandas as pd
from collections import defaultdict

def evaluate_topk(topk_file, train_file, test_file, k=1, threshold=0.0):
    """
    Evaluate Top-K predictions using Precision, Recall, and F1-score,
    after filtering out training-only URIs, keeping only test sources, and applying 1-1 constraint.

    Args:
        topk_file (str): Path to the top-k prediction file (TSV with SrcEntity, TgtEntity, Score)
        train_file (str): Path to the training mappings file (TSV)
        test_file (str): Path to the test mappings file (TSV)
        k (int): Value of K for top-k evaluation
        threshold (float): Minimum score to consider a prediction valid

    Returns:
        dict: Dictionary containing Precision@K, Recall@K, and F1@K
    """

        # === Step 1: Load input files ===
    df = pd.read_csv(topk_file, sep='\t', encoding='utf-8')
    df.columns = df.columns.str.strip()
    
    train_df = pd.read_csv(train_file, sep="\t", encoding='utf-8')
    train_df.columns = train_df.columns.str.strip()
     
    test_df = pd.read_csv(test_file, sep="\t", encoding='utf-8')
    test_df.columns = test_df.columns.str.strip()

    # === Step 2: Remove URIs only present in the training set ===
    train_uris = set(train_df['SrcEntity']) | set(train_df['TgtEntity'])
    test_uris = set(test_df['SrcEntity']) | set(test_df['TgtEntity'])
    uris_to_exclude = train_uris - test_uris
    df = df[~(df['SrcEntity'].isin(uris_to_exclude) | df['TgtEntity'].isin(uris_to_exclude))].reset_index(drop=True)

    # === Step 3: Keep only source entities from the test set ===
    src_entities_test = set(test_df['SrcEntity'])
    df = df[df['SrcEntity'].isin(src_entities_test)].reset_index(drop=True)

    # === Step 4: Convert score column to float and sort ===
    df['Score'] = df['Score'].apply(lambda x: float(x.strip("[]")) if isinstance(x, str) else float(x))
    df_sorted = df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    # === Step 5: Apply 1-to-1 constraint (greedy strategy with optional threshold)
    matched_sources = set()
    matched_targets = set()
    result = []

    for _, row in df_sorted.iterrows():
        src, tgt, score = row['SrcEntity'], row['TgtEntity'], row['Score']
        if src not in matched_sources and tgt not in matched_targets and score >= threshold:
            result.append((src, tgt, score))
            matched_sources.add(src)
            matched_targets.add(tgt)

    # === Step 6: Create and save Top-K prediction dataframe
    matching_results_df = pd.DataFrame(result, columns=['SrcEntity', 'TgtEntity', 'Score'])
    output_file = topk_file.replace(".tsv", "_predictions.tsv")
    matching_results_df.to_csv(output_file, sep='\t', index=False)

    # === Step 7: Build reference dictionary from test set
    ref_dict = defaultdict(set)
    for _, row in test_df.iterrows():
        ref_dict[row['SrcEntity']].add(row['TgtEntity'])

    # === Step 8: Select Top-K predictions for each source entity
    matching_results_df['Score'] = matching_results_df['Score'].astype(float)
    topk_df = matching_results_df.sort_values(by='Score', ascending=False).groupby('SrcEntity').head(k)

    # === Step 9: Compute Precision@K, Recall@K, F1@K
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

    # === Step 10: Print metrics

    print(f"📊 Precision@{k}:            {precision:.3f}")
    print(f"📊 Recall@{k}:               {recall:.3f}")
    print(f"📊 F1@{k}:                   {f1:.3f}\n")

    return {
        f'Precision@{k}': round(precision, 3),
        f'Recall@{k}': round(recall, 3),
        f'F1@{k}': round(f1, 3)
    }



# Main Code

# Reading semantic node embeddings provided by the Concept Features Encoder (CFE)

print("Reading semantic concepts embeddings provided by the Concept Features Encoder ...")

# Read the source embeddings from a CSV file into a pandas DataFrame
df_embbedings_src = pd.read_csv(src_Emb, index_col=0)

# Convert the DataFrame to a NumPy array, which will remove the index and store the data as a raw matrix
numpy_array = df_embbedings_src.to_numpy()

# Convert the NumPy array into a PyTorch FloatTensor, which is the format required for PyTorch operations
x_src = torch.FloatTensor(numpy_array)

# Read the target embeddings from a CSV file into a pandas DataFrame
df_embbedings_tgt = pd.read_csv(tgt_Emb, index_col=0)

# Convert the DataFrame to a NumPy array, which removes the index and converts the data to a raw matrix
numpy_array = df_embbedings_tgt.to_numpy()

# Convert the NumPy array into a PyTorch FloatTensor, which is required for PyTorch operations
x_tgt = torch.FloatTensor(numpy_array)


# Reading adjacency Matrix

# Read the source adjacency matrix from a CSV file into a pandas DataFrame
df_ma1 = pd.read_csv(src_Adjacence, index_col=0)

# Convert the DataFrame to a list of lists (Python native list format)
ma1 = df_ma1.values.tolist()

# Read the target adjacency matrix from a CSV file into a pandas DataFrame
df_ma2 = pd.read_csv(tgt_Adjacence, index_col=0)

# Convert the DataFrame to a list of lists (Python native list format)
ma2 = df_ma2.values.tolist()

# Convert Adjacency matrix (in list format) to an undirected edge index

# Convert the source adjacency matrix (in list format) to an undirected edge index for PyTorch Geometric
edge_src = adjacency_matrix_to_undirected_edge_index(ma1)

# Convert the target adjacency matrix (in list format) to an undirected edge index for PyTorch Geometric
edge_tgt = adjacency_matrix_to_undirected_edge_index(ma2)

# GIT Training

print("GIT Training...")

def train_model_gnn(model, x_src, edge_src, x_tgt, edge_tgt,
                    tensor_term1, tensor_term2, tensor_score,
                    learning_rate, weight_decay_value, num_epochs, print_interval=10):
    """
    Trains a graph neural network (GNN) model using source and target embeddings and contrastive loss.

    Args:
        model: The GNN model to be trained.
        x_src (torch.Tensor): Source node embeddings.
        edge_src (torch.Tensor): Source graph edges.
        x_tgt (torch.Tensor): Target node embeddings.
        edge_tgt (torch.Tensor): Target graph edges.
        tensor_term1 (torch.Tensor): Indices of the source nodes to be compared.
        tensor_term2 (torch.Tensor): Indices of the target nodes to be compared.
        tensor_score (torch.Tensor): Labels indicating if the pairs are matched (1) or not (0).
        learning_rate (float): Learning rate for the optimizer.
        weight_decay_value (float): Weight decay (L2 regularization) value for the optimizer.
        num_epochs (int): Number of epochs for training.
        print_interval (int): Interval at which training progress is printed (every `print_interval` epochs).

    Returns:
        model: The trained GNN model.
    """

    # Step 1: Set device (GPU or CPU) for computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 2: Move the model and all inputs to the selected device
    model.to(device)
    x_tgt = x_tgt.to(device)               # Target node embeddings
    edge_tgt = edge_tgt.to(device)         # Target graph edges
    x_src = x_src.to(device)               # Source node embeddings
    edge_src = edge_src.to(device)         # Source graph edges
    tensor_term1 = tensor_term1.to(device) # Indices for source nodes
    tensor_term2 = tensor_term2.to(device) # Indices for target nodes
    tensor_score = tensor_score.to(device) # Ground truth labels

    # Step 3: Define optimizer with learning rate and regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)

    # Step 4: Initialize list to store training losses
    train_losses = []

    # Record the start time of training
    start_time = time.time()

    # Step 5: Training loop
    for epoch in range(num_epochs):
        # Zero out gradients from the previous iteration
        optimizer.zero_grad()

        # Forward pass: Compute embeddings for source and target graphs
        out1 = model(x_src, edge_src)  # Updated source embeddings
        out2 = model(x_tgt, edge_tgt)  # Updated target embeddings

        # Extract specific rows of embeddings for terms being compared
        src_embeddings = select_rows_by_index(out1, tensor_term1)
        tgt_embeddings = select_rows_by_index(out2, tensor_term2)

        # Compute contrastive loss based on the embeddings and ground truth labels
        loss = contrastive_loss(src_embeddings, tgt_embeddings, tensor_score)

        # Backward pass: Compute gradients
        loss.backward()

        # Update the model's parameters
        optimizer.step()

        # Append the loss for this iteration to the list
        train_losses.append(loss.item())

        # Print loss every `print_interval` epochs
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item()}")

    # Step 6: Record end time of training
    end_time = time.time()

    # Step 7: Plot the training loss over time
    plt.semilogy(range(1, num_epochs + 1), train_losses, label="Training Loss", marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Print the total training time
    training_time = end_time - start_time
    print(f"Training complete! Total training time: {training_time:.2f} seconds")

    # Step 8: Return the trained model
    return model

# Initialize the GIT_mod model with the dimensionality of the target embeddings
# The first argument is the dimensionality of the target node embeddings (x_tgt.shape[1])
# The second argument (1) represents the number of RGIT layers in the model
GIT_model = RGIT_mod(x_tgt.shape[1], 1)

# Reading the training pairs from a CSV file into a pandas DataFrame
df_embbedings = pd.read_csv(train_file, index_col=0)

# Extract the 'SrcEntity' and 'TgtEntity' columns as NumPy arrays and convert them to integers
tensor_term1 = df_embbedings['SrcEntity'].values.astype(int)  # Source entity indices
tensor_term2 = df_embbedings['TgtEntity'].values.astype(int)  # Target entity indices

# Extract the 'Score' column as a NumPy array and convert it to floats
tensor_score = df_embbedings['Score'].values.astype(float)  # Scores (labels) indicating if pairs match (1) or not (0)

# Convert the NumPy arrays to PyTorch LongTensors (for indices) and FloatTensors (for scores)
tensor_term1_o = torch.from_numpy(tensor_term1).type(torch.LongTensor)  # Source entity tensor
tensor_term2_o = torch.from_numpy(tensor_term2).type(torch.LongTensor)  # Target entity tensor
tensor_score_o = torch.from_numpy(tensor_score).type(torch.FloatTensor)  # Score tensor

# Train the GNN model using the provided source and target graph embeddings, edges, and training data
trained_model = train_model_gnn(
    model=GIT_model,                # The GNN model to be trained (initialized earlier)
    x_src=x_src,                    # Source node embeddings (tensor for source graph)
    edge_src=edge_src,              # Source graph edges (undirected edge index for source graph)
    x_tgt=x_tgt,                    # Target node embeddings (tensor for target graph)
    edge_tgt=edge_tgt,              # Target graph edges (undirected edge index for target graph)
    tensor_term1=tensor_term1_o,    # Indices of source entities for training
    tensor_term2=tensor_term2_o,    # Indices of target entities for training
    tensor_score=tensor_score_o,    # Scores (labels) indicating if pairs match (1) or not (0)
    learning_rate=0.0001,            # Learning rate for the Adam optimizer
    weight_decay_value=1e-4,        # Weight decay for L2 regularization to prevent overfitting
    num_epochs=1000,                # Number of training epochs
    print_interval=10               # Interval at which to print training progress (every 10 epochs)
)

# GIT Application

# Determine if a GPU is available and move the computations to it; otherwise, use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming the model has been trained and hyperparameters (x_src, edge_src, x_tgt, edge_tgt) are set
# Move the trained GIT_model to the device (GPU or CPU)
GIT_model.to(device)

# Move the data tensors to the same device (GPU or CPU)
x_tgt = x_tgt.to(device)         # Target node embeddings
edge_tgt = edge_tgt.to(device)   # Target graph edges
x_src = x_src.to(device)         # Source node embeddings
edge_src = edge_src.to(device)   # Source graph edges

# Set the model to evaluation mode; this disables dropout and batch normalization
GIT_model.eval()

# Pass the source and target embeddings through the trained GNN model to update the embeddings
with torch.no_grad():  # Disable gradient computation (inference mode)
    embeddings_tgt = GIT_model(x_tgt, edge_tgt)  # Get updated embeddings for the target graph
    embeddings_src = GIT_model(x_src, edge_src)  # Get updated embeddings for the source graph

# Detach the embeddings from the computation graph and move them back to the CPU
# This step is useful if you need to use the embeddings for tasks outside PyTorch (e.g., saving to disk)
embeddings_tgt = embeddings_tgt.detach().cpu()  # Target graph embeddings
embeddings_src = embeddings_src.detach().cpu()  # Source graph embeddings

# At this point, embeddings_tgt and embeddings_src contain the updated embeddings, ready for downstream tasks
# Selecting embedding pairs to train the Gated Network
# Read the training pairs from a CSV file into a pandas DataFrame
df_embeddings = pd.read_csv(train_file, index_col=0)

# Extract columns and convert to NumPy arrays
tensor_term1 = df_embeddings['SrcEntity'].values.astype(int)  # Source entity indices
tensor_term2 = df_embeddings['TgtEntity'].values.astype(int)  # Target entity indices
tensor_score = df_embeddings['Score'].values.astype(float)  # Matching scores

# Split data into training and validation sets
tensor_term1_train, tensor_term1_val, tensor_term2_train, tensor_term2_val, tensor_score_train, tensor_score_val = train_test_split(
    tensor_term1, tensor_term2, tensor_score, test_size=0.3, random_state=42
)

# Convert split data to PyTorch tensors
tensor_term1_train = torch.from_numpy(tensor_term1_train).type(torch.LongTensor)
tensor_term2_train = torch.from_numpy(tensor_term2_train).type(torch.LongTensor)
tensor_score_train = torch.from_numpy(tensor_score_train).type(torch.FloatTensor)

tensor_term1_val = torch.from_numpy(tensor_term1_val).type(torch.LongTensor)
tensor_term2_val = torch.from_numpy(tensor_term2_val).type(torch.LongTensor)
tensor_score_val = torch.from_numpy(tensor_score_val).type(torch.FloatTensor)

# Move the embeddings back to the CPU if not already there
x_tgt = x_tgt.cpu()  # Target node embeddings
x_src = x_src.cpu()  # Source node embeddings

# Select embeddings for the training set
X1_train = select_rows_by_index(embeddings_src, tensor_term1_train)
X2_train = select_rows_by_index(x_src, tensor_term1_train)
X3_train = select_rows_by_index(embeddings_tgt, tensor_term2_train)
X4_train = select_rows_by_index(x_tgt, tensor_term2_train)

# Select embeddings for the validation set
X1_val = select_rows_by_index(embeddings_src, tensor_term1_val)
X2_val = select_rows_by_index(x_src, tensor_term1_val)
X3_val = select_rows_by_index(embeddings_tgt, tensor_term2_val)
X4_val = select_rows_by_index(x_tgt, tensor_term2_val)

# Now we have:
# - Training tensors: X1_train, X2_train, X3_train, X4_train, tensor_score_train
# - Validation tensors: X1_val, X2_val, X3_val, X4_val, tensor_score_val

# Gated Network Training
print("Gated Network Training...")

def train_gated_combination_model(X1_t, X2_t, X3_t, X4_t, tensor_score_o,
                                  X1_val, X2_val, X3_val, X4_val, tensor_score_val,
                                  epochs=120, batch_size=32, learning_rate=0.001, weight_decay=1e-5):
    """
    Trains the GatedCombination model with training and validation data, using ReduceLROnPlateau scheduler.
    Also calculates and displays F1-score during training and validation.
    """

    # Create datasets and DataLoaders
    train_dataset = TensorDataset(X1_t, X2_t, X3_t, X4_t, tensor_score_o)
    val_dataset = TensorDataset(X1_val, X2_val, X3_val, X4_val, tensor_score_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedCombination(X1_t.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Use ReduceLROnPlateau scheduler
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

            # Forward pass
            outputs = model(batch_X1, batch_X2, batch_X3, batch_X4)

            # Compute loss
            loss = F.binary_cross_entropy(outputs, batch_y.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # Store true labels and predictions for F1-score
            y_true_train.extend(batch_y.cpu().numpy())
            y_pred_train.extend((outputs > 0.5).float().cpu().numpy())

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Calculate F1-score for training
        train_f1 = f1_score(y_true_train, y_pred_train)

        # Validation phase
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

                # Compute loss
                val_loss = F.binary_cross_entropy(outputs, batch_y.unsqueeze(1).float())
                total_val_loss += val_loss.item()

                # Store true labels and predictions for F1-score
                y_true_val.extend(batch_y.cpu().numpy())
                y_pred_val.extend((outputs > 0.5).float().cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate F1-score for validation
        val_f1 = f1_score(y_true_val, y_pred_val)

        # Step the scheduler with validation loss
        scheduler.step(avg_val_loss)

        # Print training and validation metrics
        print(f"Epoch [{epoch + 1}/{epochs}] Training Loss: {train_loss:.4f}, F1 Score: {train_f1:.4f} | "
              f"Validation Loss: {avg_val_loss:.4f}, F1 Score: {val_f1:.4f}")

    end_time = time.time()

    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print(f"Training complete! Total time: {end_time - start_time:.2f} seconds")
    return model

# Train the GatedCombination model using training and validation data
trained_model = train_gated_combination_model(
    X1_train,          # Updated source embeddings (after applying the GNN model)
    X2_train,          # Original source embeddings (before applying the GNN model)
    X3_train,          # Updated target embeddings (after applying the GNN model)
    X4_train,          # Original target embeddings (before applying the GNN model)
    tensor_score_train, # Ground truth labels for the training set (1 for matched pairs, 0 for unmatched pairs)

    X1_val,            # Updated source embeddings for the validation set
    X2_val,            # Original source embeddings for the validation set
    X3_val,            # Updated target embeddings for the validation set
    X4_val,            # Original target embeddings for the validation set
    tensor_score_val,  # Ground truth labels for the validation set (1 for matched pairs, 0 for unmatched pairs)

    epochs=100,        # Number of epochs (iterations over the entire training dataset)
    batch_size=32,     # Number of training samples processed in one forward/backward pass
    learning_rate=0.001, # Learning rate for the optimizer (controls step size during optimization)
    weight_decay=1e-4  # Weight decay (L2 regularization) to prevent overfitting
)

# # **Second Round Modifications**

# Mappings Selector

print("Generate Embeddings...")

# # **Generate Embeddings**

# Build an indexed dictionary for the source ontology classes
# src_class is the file path to the JSON file containing the source ontology classes
indexed_dict_src = build_indexed_dict(src_class)

# Build an indexed dictionary for the target ontology classes
# tgt_class is the file path to the JSON file containing the target ontology classes
indexed_dict_tgt = build_indexed_dict(tgt_class)

# Save the final gated embeddings for all concepts in source and target ontologies
save_gated_embeddings(
    gated_model=trained_model,          # The trained GatedCombination model
    embeddings_src=embeddings_src,      # GNN-transformed embeddings for source entities
    x_src=x_src,                        # Initial semantic embeddings for source entities
    embeddings_tgt=embeddings_tgt,      # GNN-transformed embeddings for target entities
    x_tgt=x_tgt,                        # Initial semantic embeddings for target entities
    indexed_dict_src=indexed_dict_src,  # Index-to-URI mapping for source ontology
    indexed_dict_tgt=indexed_dict_tgt,  # Index-to-URI mapping for target ontology
    Emb_final_src=Emb_final_src,    # Destination file path for source embeddings
    Emb_final_tgt=Emb_final_tgt     # Destination file path for target embeddings
)


# # **Filter No Used Concepts**

# Call the function to filter out ignored concepts (e.g., owl:Thing, deprecated, etc.)
# from the source and target ontology embeddings.

# Input:
# - src_emb_path: Path to the TSV file containing embeddings for the source ontology
# - tgt_emb_path: Path to the TSV file containing embeddings for the target ontology
# - src_onto / tgt_onto: DeepOnto ontology objects used to identify ignored concepts

# Output:
# - src_file: Path to the cleaned source embeddings (with ignored concepts removed)
# - tgt_file: Path to the cleaned target embeddings (with ignored concepts removed)

src_file, tgt_file = filter_ignored_class(
    src_emb_path=Emb_final_src,
    tgt_emb_path=Emb_final_tgt,
    src_onto=src_onto,
    tgt_onto=tgt_onto,
    Emb_final_src_cl=Emb_final_src_cl,
    Emb_final_tgt_cl=Emb_final_tgt_cl
)


# # **Mappings Generation**

# # **Using faiss l2**

# Compute the top-10 most similar mappings using l2 distance
# between ResMLP-encoded embeddings of the source and target ontologies.
# The input embeddings were previously encoded using the ResMLPEncoder,
# and the similarity score is computed as the inverse of the l2 distance.
# Results are saved in a TSV file with columns: SrcEntity, TgtEntity, Score.

print("Generate mappings...")
topk_faiss_l2(
    src_emb_path=Emb_final_src_cl,
    tgt_emb_path=Emb_final_tgt_cl,
    top_k=k,
    output_file=all_predictions_path
)


# # **Evaluation**

# # **Global Metrics: Precision, Recall and F1 score**

# Run the evaluation on the predicted mappings using evaluation function.

print("Calculating global evaluation metrics (precision, recall and F1-score) ....")

output_file, metrics, correct = evaluate_predictions(
    pred_file=all_predictions_path,
    # Path to the TSV file containing predicted mappings with scores (before filtering).

    train_file=train_file_origin,
    # Path to the training reference file (used to exclude mappings involving train-only entities).

    test_file=test_file,
    # Path to the test reference file (used as the gold standard for evaluation).
)

# This function returns:
# - `output_file`: the path to the filtered and evaluated output file.
# - `metrics`: a tuple containing (Precision, Recall, F1-score).
# - `correct`: the number of correctly predicted mappings found in the gold standard.


# # **Metrics@1**

# Compute the top-1 most similar mappings using l2 distance
# and the similarity score is computed as the inverse of the l2 distance.
# Results are saved in a TSV file with columns: SrcEntity, TgtEntity, Score.

print("Calculating ranking metrics (precision@k, recall@k and F1-score@k) ....")

topk_faiss_l2(
    src_emb_path=Emb_final_src_cl,
    tgt_emb_path=Emb_final_tgt_cl,
    top_k=1,
    output_file=top_1_predictions
)


results = evaluate_topk(
    topk_file=top_1_predictions,
    # Path to the file containing the predicted mappings with scores.
    # This file may include unfiltered predictions (e.g., over all candidates).

    train_file=train_file_origin,
    # Path to the training reference mappings file.
    # Used to remove mappings that involve entities appearing only in training.

    test_file=test_file
    # Path to the test reference mappings file.
    # Ground-truth correspondences are extracted from this file for evaluation.
)


# # **Local MRR and Hit@k**

print("Calculating local ranking metrics (MRR and Hit@k) ....")

import pandas as pd

# === Step 1: Load input files ===

# Define paths to cleaned embedding files
src_emb_path = Emb_final_src_cl
tgt_emb_path = Emb_final_tgt_cl

# Load candidate mappings (SrcEntity, TgtEntity) and source/target embeddings
df_cands = pd.read_csv(cands_path)
src_emb_df = pd.read_csv(src_emb_path, sep="\t")
tgt_emb_df = pd.read_csv(tgt_emb_path, sep="\t")

# === Step 2: Extract unique source and target URIs from the candidate pairs ===

# Keep only distinct source and target entities (URIs) for which embeddings are needed
unique_src_df = pd.DataFrame(df_cands["SrcEntity"].unique(), columns=["Concept"])
unique_tgt_df = pd.DataFrame(df_cands["TgtEntity"].unique(), columns=["Concept"])

# === Step 3: Join embeddings for each concept based on the "Concept" URI ===

# Merge source entities with their corresponding embeddings (if available)
merged_src_df = pd.merge(unique_src_df, src_emb_df, on="Concept", how="left")

# Merge target entities with their corresponding embeddings (if available)
merged_tgt_df = pd.merge(unique_tgt_df, tgt_emb_df, on="Concept", how="left")

# === Step 4: Save the merged results to TSV files ===

# Save the source concepts and their embeddings to file
merged_src_df.to_csv(src_rank_emb, sep="\t", index=False)

# Save the target concepts and their embeddings to file
merged_tgt_df.to_csv(tgt_rank_emb, sep="\t", index=False)

topk_faiss_l2(
    # Path to the source entity embeddings (already filtered and linearly encoded)
    src_emb_path=src_rank_emb,

    # Path to the target entity embeddings (already filtered and linearly encoded)
    tgt_emb_path=tgt_rank_emb,

    # Number of top matches to retrieve per source entity (Top-K candidates)
    top_k=200,

    # Path to save the resulting Top-K mappings sorted by FAISS L2 distance (converted to similarity)
    output_file=mappings_mrr
)


# Format the prediction scores into ranked candidate lists per source entity,
# using the gold standard candidate file as reference. This prepares the output
# for MRR and Hits@k evaluation. The output is a TSV file with columns:
# SrcEntity, TgtEntity (ground truth), and TgtCandidates (ranked list with scores).
format_ranked_predictions_for_mrr(
    reference_file=test_cands,        # Gold reference with candidate sets
    predicted_file=mappings_mrr,  # Flat prediction scores (Src, Tgt, Score)
    output_file=formatted_predictions_path                             # Output path for ranked candidate format
)

# Evaluate ranking performance using standard metrics like MRR and Hits@K
# 'formatted_predictions_path' should point to a TSV file with columns: SrcEntity, TgtEntity, TgtCandidates
# This function computes how well the true targets are ranked among the candidates
results = ranking_eval(formatted_predictions_path, Ks=[1, 5, 10])

# Print the evaluation results for Hits@1, Hits@5, and Hits@10
print("Ranking Evaluation Results at K=1, 5, and 10:")
print(results)
