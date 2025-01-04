# Package Installation

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

# Define the source ontology name
src_ent = "omim"

# Define the target ontology name
tgt_ent = "ordo"

# Define the task name for this ontology matching process
task = "omim2ordo"

# Define the similarity threshold for validating matches
thres = 0.20

print(f"Matching {src_ent}.owl and {tgt_ent}.owl:")

# Sets the relative path to the root directory of the BioGITOM project.
# This path points three levels up from the current working directory
# and navigates to the 'biogitom' folder. Adjust this path if the 
# directory structure changes or the script is executed from a different location.

dir = "../biogitom"

# Define the directory for the dataset containing source and target ontologies
dataset_dir = f"{dir}/Datasets/{task}"

# Define the data directory for storing embeddings, adjacency matrices, and related files
data_dir = f"{dir}/Tasks/{task}/Data"

# Define the directory for storing the results
results_dir = f"{dir}/Tasks/{task}/Results"

# Load the Source ontology using the Ontology class from DeepOnto
# This initializes the source ontology by loading its .owl file.
src_onto = Ontology(f"{dataset_dir}/{src_ent}.owl")

# Load the Target ontology using the Ontology class from DeepOnto
# This initializes the target ontology by loading its .owl file.
tgt_onto = Ontology(f"{dataset_dir}/{tgt_ent}.owl")

# Define the file path for the Source embeddings CSV file
# Embeddings for the source ontology entities are stored in this file.
src_Emb = f"{data_dir}/{src_ent}_Sentence_SapBERT_emb.csv"

# Define the file path for the Target embeddings CSV file
# Embeddings for the target ontology entities are stored in this file.
tgt_Emb = f"{data_dir}/{tgt_ent}_Sentence_SapBERT_emb.csv"

# Define the file path for the Source adjacency matrix
# This file represents the relationships (edges) between entities in the source ontology.
src_Adjacence = f"{data_dir}/{src_ent}_adjacence.csv"

# Define the file path for the Target adjacency matrix
# This file represents the relationships (edges) between entities in the target ontology.
tgt_Adjacence = f"{data_dir}/{tgt_ent}_adjacence.csv"

# Define the file path for the JSON file containing the Source ontology class labels
# This file maps the source ontology entities to their labels or names.
src_class = f"{data_dir}/{src_ent}_classes.json"

# Define the file path for the JSON file containing the Target ontology class labels
# This file maps the target ontology entities to their labels or names.
tgt_class = f"{data_dir}/{tgt_ent}_classes.json"

# Define the file path for the train data
train_file = f"{data_dir}/{task}_train.csv"

# Define the file path for the test data
# The test file contains reference mappings (ground truth) between the source and target ontologies.
test_file = f"{dataset_dir}/refs_equiv/test.tsv"

# Define the file path for the candidate mappings used during testing
# This file includes the candidate pairs (source and target entities) for ranking and evaluation.
test_cands = f"{dataset_dir}/refs_equiv/test.cands.tsv"

# Define the file path for the candidate mappings between Source to Target entities
# This file contains cleaned, combined, and encoded candidates used for predictions.
candidates_Prediction = f"{data_dir}/{task}_candidates_prediction.csv"

# Define the file path for the candidate mappings between Source to Target entities for ranking-based metrics
# This file is used to compute ranking-based metrics like MRR and Hits@k.
candidates_Rank = f"{data_dir}/{task}_candidates.csv"

# Define the path where the prediction results will be saved in TSV format
# This file will store the final predictions (mappings) between source and target entities.
prediction_path = f"{results_dir}/{task}_matching_results.tsv"

# Define the path where all prediction results will be saved in TSV format
# This file will store detailed prediction results, including all candidate scores.
all_predictions_path = f"{results_dir}/{task}_all_predictions.tsv"

# Define the path where all ranking prediction results will be saved in TSV format
# This file will store predictions sorted by rank based on their scores.
all_predictions_path_ranked = f"{results_dir}/{task}_all_predictions_ranked.tsv"

# Define the path where formatted ranking predictions will be saved in TSV format
# This file will contain predictions formatted for evaluation using ranking-based metrics.
formatted_predictions_path = f"{results_dir}/{task}_formatted_predictions.tsv"

# GIT Architecture

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

# Gated Network Architecture

class GatedCombination(nn.Module):
    """
    A neural network module for combining embeddings using a gating mechanism
    and evaluating their similarity. This class is particularly useful for
    ontology matching tasks.
    
    Args:
        input_dim (int): Dimensionality of the input embeddings.
    """
    def __init__(self, input_dim):
        super(GatedCombination, self).__init__()
        
        # Fully connected layer for gating mechanism on the first embedding pair (x1, x2)
        self.gate_A_fc = nn.Linear(input_dim, input_dim)
        
        # Fully connected layer for gating mechanism on the second embedding pair (x3, x4)
        self.gate_B_fc = nn.Linear(input_dim, input_dim)
        
        # Fully connected layer for final similarity classification
        self.fc = nn.Linear(1, 1)

    def forward(self, x1, x2, x3, x4, return_embeddings=False):
        """
        Forward pass through the GatedCombination model.
        
        Args:
            x1 (torch.Tensor): First set of embeddings (e.g., updated source embeddings).
            x2 (torch.Tensor): Second set of embeddings (e.g., original source embeddings).
            x3 (torch.Tensor): Third set of embeddings (e.g., updated target embeddings).
            x4 (torch.Tensor): Fourth set of embeddings (e.g., original target embeddings).
            return_embeddings (bool): Whether to return the intermediate combined embeddings (a, b).
        
        Returns:
            torch.Tensor: Probability score for binary classification if return_embeddings is False.
            Tuple[torch.Tensor, torch.Tensor]: Intermediate embeddings (a, b) if return_embeddings is True.
        """
        # Compute gating weights for the first pair of embeddings (x1, x2)
        gate_values1 = torch.sigmoid(self.gate_A_fc(x1))
        
        # Blend x1 and x2 using the computed gating weights
        a = x1 * gate_values1 + x2 * (1 - gate_values1)

        # Compute gating weights for the second pair of embeddings (x3, x4)
        gate_values2 = torch.sigmoid(self.gate_B_fc(x3))
        
        # Blend x3 and x4 using the computed gating weights
        b = x3 * gate_values2 + x4 * (1 - gate_values2)

        # If return_embeddings is True, return the intermediate blended embeddings
        if return_embeddings:
            return a, b

        # Compute cosine similarity between the blended embeddings a and b
        x = torch.cosine_similarity(a, b, dim=1)
        
        # Apply a fully connected layer and sigmoid activation for classification
        out = torch.sigmoid(self.fc(x.unsqueeze(1)))
        return out

# Utility functions

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

def Prediction_with_candidates(model, X1_tt, X2_tt, X3_tt, X4_tt, src_entity_tensor_o, tgt_entity_tensor_o,
                                   indexed_dict_src, indexed_dict_tgt, all_predictions_path):
    """
    Evaluates the GatedCombination model using the given embeddings and candidate entity pairs.
    Saves the predictions and evaluation results to a file.

    Args:
        model: Trained GatedCombination model.
        X1_tt, X2_tt, X3_tt, X4_tt (torch.Tensor): Tensors of source and target entity embeddings (updated and original).
        src_entity_tensor_o, tgt_entity_tensor_o (torch.Tensor): Tensors of source and target entity indices.
        indexed_dict_src, indexed_dict_tgt (dict): Dictionaries mapping entity indices to URIs for source and target.
        output_file (str): Path to save the predictions and results.
        hits_at_k_values (list): List of k-values for which hits@k is evaluated.

    Returns:
        None
    """
    # Move the model to CPU and set it to evaluation mode
    model = model.to("cpu")
    model.eval()

    # Set batch size for evaluation
    batch_size_test = 32

    # Create a DataLoader for the evaluation data
    test_dataset = TensorDataset(X1_tt, X2_tt, X3_tt, X4_tt)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test)

    # Prepare for collecting predictions and results
    predictions = []
    results = []
    count_predictions = 0  # Counter for predictions above threshold (0.5)

    # Measure prediction time
    start_time = time.time()

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over batches and compute model predictions
        for batch_X1, batch_X2, batch_X3, batch_X4 in test_dataloader:
            outputs = model(batch_X1, batch_X2, batch_X3, batch_X4)
            predictions.extend(outputs.cpu().numpy())  # Collect predictions in CPU memory

    end_time = time.time()
    predicting_time = end_time - start_time
    print(f"Predicting time: {predicting_time:.2f} seconds")

    # Convert tensors to lists for easier iteration
    src_indices = src_entity_tensor_o.tolist()
    tgt_indices = tgt_entity_tensor_o.tolist()

    # Prepare results
    for i in range(len(predictions)):
        if predictions[i] >= 0.00:  # Consider only predictions greater than 0.5
            count_predictions += 1  # Increment the counter

            # Map the source and target entity indices to their URIs
            src_code = src_indices[i]
            tgt_code = tgt_indices[i]

            src_uri = indexed_dict_src.get(int(src_code), "Unknown URI")
            tgt_uri = indexed_dict_tgt.get(int(tgt_code), "Unknown URI")

            # Get the model's predicted score for the current pair
            score = predictions[i]

            # Append the results (with URIs instead of entity indices)
            results.append({
                'SrcEntity': src_uri,
                'TgtEntity': tgt_uri,
                'Score': score
            })

    # Convert the results into a pandas DataFrame
    df_results = pd.DataFrame(results)

    # Save the results to a TSV file
    df_results.to_csv(all_predictions_path, sep='\t', index=False)

    print(f"Predictions saved to {all_predictions_path}")

def filter_highest_predictions(input_file_path, output_file_path, threshold=thres):
    # Load the all predictions file
    df = pd.read_csv(input_file_path, sep='\t')

    # Extract the similarity score from the list in the 'Score' column
    df['Score'] = df['Score'].apply(lambda x: float(x.strip('[]')))

    # Sorting the dataframe by similarity score in descending order
    df_sorted = df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    # Initialize variables with threshold value
    source_concepts = set(df_sorted['SrcEntity'])
    target_concepts = set(df_sorted['TgtEntity'])
    matched_sources = set()
    matched_targets = set()
    result = []

    # Iterate through the sorted dataframe and find highest correspondences
    for _, row in df_sorted.iterrows():
        source, target, similarity = row['SrcEntity'], row['TgtEntity'], row['Score']

        # Check if the source or target has already been matched and if the similarity is above the threshold
        if source not in matched_sources and target not in matched_targets and similarity >= threshold:
            # Add the match to the result list
            result.append((source, target, similarity))
            # Mark the source and target as matched
            matched_sources.add(source)
            matched_targets.add(target)

    # Create a dataframe for the matching results with threshold applied
    matching_results_df_threshold = pd.DataFrame(result, columns=['SrcEntity', 'TgtEntity', 'Score'])

    # Save the matching results with the updated column names to a new TSV file
    matching_results_df_threshold.to_csv(output_file_path, sep='\t', index=False)

    # Print the number of predictions saved
    print(f"Number of Positive Predictions : {len(matching_results_df_threshold)}")

    return matching_results_df_threshold, len(matching_results_df_threshold)

def compute_mrr_and_hits(reference_file, predicted_file, output_file, k_values=[1, 5, 10]):
    """
    Compute MRR and Hits@k for ontology matching predictions based on a reference file.

    Args:
        reference_file (str): Path to the reference file (test.cands.tsv format).
        predicted_file (str): Path to the predictions file with scores.
        output_file (str): Path to save the scored results.
        k_values (list): List of k values for Hits@k.

    Returns:
        dict: A dictionary containing MRR and Hits@k metrics.
    """
    # Read the reference mappings
    test_candidate_mappings = read_table(reference_file).values.tolist()
    ranking_results = []

    # Read the predicted scores
    predicted_data = pd.read_csv(predicted_file, sep="\t")
    predicted_data["Score"] = predicted_data["Score"].apply(lambda x: float(x.strip("[]")))

    # Create a lookup dictionary for predicted scores
    score_lookup = {}
    for _, row in predicted_data.iterrows():
        score_lookup[(row["SrcEntity"], row["TgtEntity"])] = row["Score"]

    for src_ref_class, tgt_ref_class, tgt_cands in test_candidate_mappings:
        tgt_cands = eval(tgt_cands)  # Convert string to list of candidates
        scored_cands = []
        for tgt_cand in tgt_cands:
            # Retrieve score for each candidate, defaulting to a very low score if not found
            matching_score = score_lookup.get((src_ref_class, tgt_cand), -1e9)
            scored_cands.append((tgt_cand, matching_score))

        # Sort candidates by score in descending order
        scored_cands = sorted(scored_cands, key=lambda x: x[1], reverse=True)
        ranking_results.append((src_ref_class, tgt_ref_class, scored_cands))

    # Save the ranked results to a file
    pd.DataFrame(ranking_results, columns=["SrcEntity", "TgtEntity", "TgtCandidates"]).to_csv(output_file, sep="\t", index=False)

    # Compute MRR and Hits@k
    total_entities = len(ranking_results)
    reciprocal_ranks = []
    hits_at_k = {k: 0 for k in k_values}

    for src_entity, tgt_ref_class, tgt_cands in ranking_results:
        ranked_candidates = [candidate[0] for candidate in tgt_cands]
        if tgt_ref_class in ranked_candidates:
            rank = ranked_candidates.index(tgt_ref_class) + 1
            reciprocal_ranks.append(1 / rank)
            for k in k_values:
                if rank <= k:
                    hits_at_k[k] += 1
        else:
            reciprocal_ranks.append(0)

    mrr = sum(reciprocal_ranks) / total_entities
    hits_at_k = {k: hits / total_entities for k, hits in hits_at_k.items()}

    return {"MRR": mrr, "Hits@k": hits_at_k}

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

# Mappings Selector
print("Scoring Mappings...")
# Build an indexed dictionary for the source ontology classes
# src_class is the file path to the JSON file containing the source ontology classes
indexed_dict_src = build_indexed_dict(src_class)

# Build an indexed dictionary for the target ontology classes
# tgt_class is the file path to the JSON file containing the target ontology classes
indexed_dict_tgt = build_indexed_dict(tgt_class)

# Read the candidate pairs from a Candidates CSV file into a pandas DataFrame
df_embbedings = pd.read_csv(candidates_Prediction, index_col=0)

# Extract the 'SrcEntity' column (source entity indices) and convert it to a NumPy array of integers
tensor_term1 = df_embbedings['SrcEntity'].values.astype(int)

# Extract the 'TgtEntity' column (target entity indices) and convert it to a NumPy array of integers
tensor_term2 = df_embbedings['TgtEntity'].values.astype(int)

# Convert the source entity indices to a PyTorch LongTensor
src_entity_tensor_o = torch.from_numpy(tensor_term1).type(torch.LongTensor)

# Convert the target entity indices to a PyTorch LongTensor
tgt_entity_tenso_or = torch.from_numpy(tensor_term2).type(torch.LongTensor)

# Select rows from the updated source embeddings based on the indices in src_entity_tensor_o
X1_tt = select_rows_by_index(embeddings_src, src_entity_tensor_o)

# Select rows from the original source embeddings based on the indices in src_entity_tensor_o
X2_tt = select_rows_by_index(x_src, src_entity_tensor_o)

# Select rows from the updated target embeddings based on the indices in tgt_entity_tenso_or
X3_tt = select_rows_by_index(embeddings_tgt, tgt_entity_tenso_or)

# Select rows from the original target embeddings based on the indices in tgt_entity_tenso_or
X4_tt = select_rows_by_index(x_tgt, tgt_entity_tenso_or)

# Generate predictions for candidate mappings using the trained GatedCombination model
Prediction_with_candidates(
    model=trained_model,             # The trained GatedCombination model used to evaluate similarity
    X1_tt=X1_tt,                     # Updated source embeddings (after applying the GIT model)
    X2_tt=X2_tt,                     # Original source embeddings (before applying the GIT model)
    X3_tt=X3_tt,                     # Updated target embeddings (after applying the GIT model)
    X4_tt=X4_tt,                     # Original target embeddings (before applying the GIT model)
    src_entity_tensor_o=src_entity_tensor_o,  # Tensor of source entity indices used for evaluation
    tgt_entity_tensor_o=tgt_entity_tenso_or,  # Tensor of target entity indices used for evaluation
    indexed_dict_src=indexed_dict_src,        # Dictionary mapping source entity indices to their URIs
    indexed_dict_tgt=indexed_dict_tgt,        # Dictionary mapping target entity indices to their URIs
    all_predictions_path=all_predictions_path # Path to save all predictions with similarity scores in TSV format
)

# Filter the highest scoring predictions from the predictions file and save the results to a new file
print("Filtring Mappings...")
matching_results_df = filter_highest_predictions(
    all_predictions_path,  # Path to the file containing all predictions with scores for all candidate pairs
    prediction_path        # Path where the filtered predictions with highest scores will be saved
)

# Evaluation
print("Evaluation...")

# Global metrics calculation
print("Global metrics calculation...")

# Retrieve the indices of the ignored classes (from source and target ontologies)
ignored_class_index = get_ignored_class_index(src_onto)  # Get ignored class indices from source ontology
ignored_class_index.update(get_ignored_class_index(tgt_onto))  # Update with ignored class indices from target ontology

# Read the predicted mappings from the prediction results file
preds = EntityMapping.read_table_mappings(prediction_path)

# Read the reference mappings from the ground truth test file
refs = ReferenceMapping.read_table_mappings(test_file)

# Filter the predicted mappings to remove any mappings that involve ignored classes
preds = remove_ignored_mappings(preds, ignored_class_index)

# Compute the precision, recall, and F1-score by comparing predictions with the reference mappings
results = AlignmentEvaluator.f1(preds, refs)

preds2 = [p.to_tuple() for p in preds]
refs2 = [r.to_tuple() for r in refs]

correct= len(set(preds2).intersection(set(refs2)))

print(f"Number of Correct Predictions: {correct}")

# Print the computed precision, recall, and F1-score metrics
print(results)

# Ranked-based metrics calculation
print("Ranked-based metrics calculation...")
# Read the candidate pairs from a Candidates CSV file into a pandas DataFrame
df_embbedings = pd.read_csv(candidates_Rank, index_col=0)

# Extract the 'SrcEntity' column (source entity indices) and convert it to a NumPy array of integers
tensor_term1 = df_embbedings['SrcEntity'].values.astype(int)

# Extract the 'TgtEntity' column (target entity indices) and convert it to a NumPy array of integers
tensor_term2 = df_embbedings['TgtEntity'].values.astype(int)

# Convert the source entity indices to a PyTorch LongTensor
src_entity_tensor_o = torch.from_numpy(tensor_term1).type(torch.LongTensor)

# Convert the target entity indices to a PyTorch LongTensor
tgt_entity_tenso_or = torch.from_numpy(tensor_term2).type(torch.LongTensor)

# Select rows from the updated source embeddings based on the indices in src_entity_tensor_o
X1_tt = select_rows_by_index(embeddings_src, src_entity_tensor_o)

# Select rows from the original source embeddings based on the indices in src_entity_tensor_o
X2_tt = select_rows_by_index(x_src, src_entity_tensor_o)

# Select rows from the updated target embeddings based on the indices in tgt_entity_tenso_or
X3_tt = select_rows_by_index(embeddings_tgt, tgt_entity_tenso_or)

# Select rows from the original target embeddings based on the indices in tgt_entity_tenso_or
X4_tt = select_rows_by_index(x_tgt, tgt_entity_tenso_or)

# Perform ranking-based predictions using the trained GatedCombination model
# Generate predictions for candidate mappings using the trained GatedCombination model
Prediction_with_candidates(
    model=trained_model,             # The trained GatedCombination model used to evaluate similarity
    X1_tt=X1_tt,                     # Updated source embeddings (after applying the GIT model)
    X2_tt=X2_tt,                     # Original source embeddings (before applying the GIT model)
    X3_tt=X3_tt,                     # Updated target embeddings (after applying the GIT model)
    X4_tt=X4_tt,                     # Original target embeddings (before applying the GIT model)
    src_entity_tensor_o=src_entity_tensor_o,  # Tensor of source entity indices used for evaluation
    tgt_entity_tensor_o=tgt_entity_tenso_or,  # Tensor of target entity indices used for evaluation
    indexed_dict_src=indexed_dict_src,        # Dictionary mapping source entity indices to their URIs
    indexed_dict_tgt=indexed_dict_tgt,        # Dictionary mapping target entity indices to their URIs
    all_predictions_path=all_predictions_path_ranked, # Path where the ranked predictions will be saved in TSV format
)

# Compute MRR and Hits@k metrics
# This function evaluates the predicted rankings against the reference mappings
results = compute_mrr_and_hits(
    reference_file=test_cands,             # Reference file with true ranks
    predicted_file=all_predictions_path_ranked,             # File containing predicted rankings
    output_file=formatted_predictions_path,    # File path to save formatted predictions
    k_values=[1, 5, 10]                        # Evaluate Hits@1, Hits@5, and Hits@10
)

# Display the computed metrics
print("MRR and Hits@k Results:")
print(results)  # Output the Mean Reciprocal Rank (MRR) and Hits@k metrics

# Call the ranking evaluation function, passing the path to the formatted predictions file.
# Ks specifies the evaluation levels, checking if the correct target is within the top K candidates.
print("Ranked-based metrics calculation using DeepOnto...")
results = ranking_eval(formatted_predictions_path, Ks=[1, 5, 10])
print("Ranking Evaluation Results at K=1, 5, and 10:")
print(results)


