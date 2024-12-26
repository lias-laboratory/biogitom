import os
import json
import pandas as pd
import numpy as np


# Define the task and file paths
task = "pharm"  # Task name
base_dir = "../../../biogitom" 
dataset_dir = os.path.join(base_dir, "Datasets", task, "refs_equiv")
data_dir = os.path.join(base_dir, "Tasks", task, "Data")

# Define file paths
train_path = os.path.join(dataset_dir, "train.tsv")  # Training data
src_class = os.path.join(data_dir, "snomed.pharm_classes.json")  # Source ontology classes
tgt_class = os.path.join(data_dir, "ncit.pharm_classes.json")  # Target ontology classes
encoded_train_path = os.path.join(data_dir, f"{task}_train.encoded.csv")  # Output for encoded training data
src_emb = os.path.join(data_dir, "snomed.pharm_emb.csv")  # Source embeddings
tgt_emb = os.path.join(data_dir, "ncit.pharm_emb.csv")  # Target embeddings


# Helper Functions
def build_indexed_dict(file_path):
    """
    Build an indexed dictionary from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary mapping keys (URIs) to unique integer indices.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {key: index for index, key in enumerate(data.keys())}


def encode_uris(row, src_dict, tgt_dict):
    """
    Encode URIs into integer indices using provided dictionaries.

    Args:
        row (pd.Series): Row from a DataFrame with `SrcEntity` and `TgtEntity` columns.
        src_dict (dict): Dictionary for source entities.
        tgt_dict (dict): Dictionary for target entities.

    Returns:
        pd.Series: Encoded source and target entity indices.
    """
    uri_1, uri_2 = row['SrcEntity'], row['TgtEntity']
    encoded_uri_1 = src_dict.get(uri_1, -1)  # Default to -1 if not found
    encoded_uri_2 = tgt_dict.get(uri_2, -1)  # Default to -1 if not found
    return pd.Series([int(encoded_uri_1), int(encoded_uri_2)])


def extract_negatives(f1, f2, df, n_negatives):
    """
    Generate negative samples for training.

    Args:
        f1 (str): Path to source embeddings file.
        f2 (str): Path to target embeddings file.
        df (pd.DataFrame): DataFrame with positive samples.
        n_negatives (int): Number of negative samples to generate per source entity.

    Returns:
        pd.DataFrame: DataFrame containing negative samples.
    """
    embs1 = pd.read_csv(f1, index_col=0).to_numpy()  # Load source embeddings
    embs2 = pd.read_csv(f2, index_col=0).to_numpy()  # Load target embeddings
    him = df.to_numpy()  # Convert positive samples to NumPy array
    negative_samples = []

    for i, src in enumerate(df['SrcEntity'].values):
        # Find positive target entities for the current source entity
        already_positive = him[np.where(him[:, 0] == src), 1].astype(int).flatten()

        # Exclude positive entities from candidate targets
        candidate_tgt = np.setdiff1d(np.arange(embs2.shape[0]), already_positive)

        # Determine the number of negatives to sample
        current_n_negatives = min(len(candidate_tgt), n_negatives)

        # Randomly sample negatives from remaining candidates
        kept = np.random.choice(candidate_tgt, size=current_n_negatives, replace=False)

        for tgt in kept:
            negative_samples.append([src, tgt, 0])  # Negative score: 0

        print(f"{i + 1}/{df.shape[0]} : {(i + 1) / df.shape[0] * 100:.2f}%\t\t\t", end="\r")

    return pd.DataFrame(negative_samples, columns=["SrcEntity", "TgtEntity", "Score"])


# Main Execution
if __name__ == "__main__":
    # Encode the training data
    print("Encoding training data...")
    indexed_dict_src = build_indexed_dict(src_class)
    indexed_dict_tgt = build_indexed_dict(tgt_class)

    # Load train data
    entity_pairs_df = pd.read_csv(train_path, sep='\t')
    encoded_entity_pairs_df = entity_pairs_df.apply(
        encode_uris, axis=1, src_dict=indexed_dict_src, tgt_dict=indexed_dict_tgt
    )

    # Add Score and ID columns
    encoded_entity_pairs_df['Score'] = 1  # Positive samples have a score of 1
    encoded_entity_pairs_df['ID'] = range(len(encoded_entity_pairs_df))  # Unique IDs
    encoded_entity_pairs_df = encoded_entity_pairs_df[['ID', 'SrcEntity', 'TgtEntity', 'Score']]  # Reorder columns

    # Save encoded training data
    encoded_entity_pairs_df.to_csv(encoded_train_path, index=False)
    print(f"Encoded training set saved to: {encoded_train_path}")

    # Generate negative samples
    print("Generating negative samples...")
    df = pd.read_csv(encoded_train_path, sep=',')
    for nb_negs in [20, 50, 100, 200]:  # Generate negative sets with different sizes
        df_negs = extract_negatives(src_emb, tgt_emb, df, n_negatives=nb_negs)
        df_final = pd.concat([df, df_negs], axis=0).reset_index(drop=True)

        # Save the combined dataset with negative samples
        output_path = os.path.join(data_dir, f"{task}_train_2_{nb_negs}.csv")
        df_final.to_csv(output_path, index=False)
        print(f"Training set with {nb_negs} negatives saved to: {output_path}")
