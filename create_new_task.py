import os
import argparse
import shutil
import json
import pandas as pd
import numpy as np

# ================================
# 📁 Directory Initialization
# ================================
def create_directories(task_name):
    """
    Create directory structure for the new task:
    - Tasks/<task_name>/Data and Results
    - Datasets/<task_name>/refs_equiv
    """
    task_root = os.path.join("Tasks", task_name)
    data_dir = os.path.join(task_root, "Data")
    results_dir = os.path.join(task_root, "Results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    dataset_dir = os.path.join("Datasets", task_name)
    refs_dir = os.path.join(dataset_dir, "refs_equiv")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(refs_dir, exist_ok=True)

    print(f"📁 Task created: {task_root}")
    return data_dir, dataset_dir, refs_dir

# ================================
# 🧾 Ontology Upload
# ================================
def upload_ontology(prompt, dataset_dir):
    """
    Prompt user to upload a .owl ontology file and copy it to the dataset folder.
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
# 🔍 Locate Embeddings
# ================================
def find_embedding_files(data_dir, src_name, tgt_name):
    """
    Find embeddings files (.csv) for the source and target ontologies based on their names.
    """
    src_emb = [f for f in os.listdir(data_dir) if f.endswith("_emb.csv") and src_name in f]
    tgt_emb = [f for f in os.listdir(data_dir) if f.endswith("_emb.csv") and tgt_name in f]
    if not src_emb or not tgt_emb:
        raise FileNotFoundError("Embedding files not found for the given source/target names.")
    return os.path.join(data_dir, src_emb[0]), os.path.join(data_dir, tgt_emb[0])

# ================================
# 🧠 Encode & Generate Negatives
# ================================
def run_encoding_and_negatives(task_name, train_path, data_dir, src_name, tgt_name):
    """
    Encode entities from the training file using their indexed positions,
    then generate negative samples for training.
    """
    print("Encoding training data and generating negatives...")
    src_class = os.path.join(data_dir, f"{src_name}_classes.json")
    tgt_class = os.path.join(data_dir, f"{tgt_name}_classes.json")
    src_emb, tgt_emb = find_embedding_files(data_dir, src_name, tgt_name)
    encoded_train_path = os.path.join(data_dir, f"{task_name}_train.encoded.csv")

    # Build URI → index mapping
    def build_indexed_dict(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return {key: index for index, key in enumerate(data.keys())}

    # Convert URIs in dataframe to integer indices
    def encode_uris(row, src_dict, tgt_dict):
        encoded_uri_1 = src_dict.get(row['SrcEntity'], -1)
        encoded_uri_2 = tgt_dict.get(row['TgtEntity'], -1)
        return pd.Series({'SrcEntity': int(encoded_uri_1), 'TgtEntity': int(encoded_uri_2)})

    # Generate negative examples avoiding true positives
    def extract_negatives(f1, f2, df, n_negatives):
        embs1 = pd.read_csv(f1, index_col=0).to_numpy()
        embs2 = pd.read_csv(f2, index_col=0).to_numpy()
        him = df.to_numpy()
        negative_samples = []
        for i, src in enumerate(df['SrcEntity'].values):
            already_positive = him[np.where(him[:, 0] == src), 1].astype(int).flatten()
            candidate_tgt = np.setdiff1d(np.arange(embs2.shape[0]), already_positive)
            current_n_negatives = min(len(candidate_tgt), n_negatives)
            kept = np.random.choice(candidate_tgt, size=current_n_negatives, replace=False)
            for tgt in kept:
                negative_samples.append([src, tgt, 0])
            print(f"{i + 1}/{df.shape[0]} : {(i + 1) / df.shape[0] * 100:.2f}%\t\t\t", end="\r")
        return pd.DataFrame(negative_samples, columns=["SrcEntity", "TgtEntity", "Score"])

    # Load mappings and encode
    indexed_dict_src = build_indexed_dict(src_class)
    indexed_dict_tgt = build_indexed_dict(tgt_class)
    entity_pairs_df = pd.read_csv(train_path, sep='\t')
    encoded_entity_pairs_df = entity_pairs_df.apply(
        encode_uris, axis=1, src_dict=indexed_dict_src, tgt_dict=indexed_dict_tgt
    )
    encoded_entity_pairs_df['Score'] = 1
    encoded_entity_pairs_df['ID'] = range(len(encoded_entity_pairs_df))
    encoded_entity_pairs_df = encoded_entity_pairs_df[['ID', 'SrcEntity', 'TgtEntity', 'Score']]
    encoded_entity_pairs_df.to_csv(encoded_train_path, index=False)
    print(f"Encoded training set saved to: {encoded_train_path}")

    # Generate and save the final training set with negatives
    df = pd.read_csv(encoded_train_path, sep=',')
    df_negs = extract_negatives(src_emb, tgt_emb, df, n_negatives=50)
    df_final = pd.concat([df, df_negs], axis=0).reset_index(drop=True)
    output_path = os.path.join(data_dir, f"{task_name}_train.csv")
    df_final.to_csv(output_path, index=False)
    print(f"Training set with negatives saved to: {output_path}")

# ================================
# 📄 Training Reference File Upload
# ================================
def upload_and_copy_train(refs_dir, task_name, data_dir, src_name, tgt_name):
    """
    Ask user to upload the train.tsv file, copy it, and launch encoding/negative generation.
    """
    print("\n📄 Upload the training reference file (train.tsv)")
    for attempt in range(3):
        path = input("📄 Provide full path to training file (.tsv): ").strip()
        if os.path.exists(path) and path.endswith(".tsv"):
            train_path = os.path.join(refs_dir, "train.tsv")
            shutil.copy(path, train_path)
            print("✅ train.tsv copied to refs_equiv/")
            run_encoding_and_negatives(task_name, train_path, data_dir, src_name, tgt_name)
            return
        else:
            print("❌ File not found or invalid format.")
            if attempt < 2 and input("Try again? (y/n): ").strip().lower() != "y":
                exit(1)
    print("🚫 Too many failed attempts.")
    exit(1)

# ================================
# 🧾 Script Generators
# ================================
def create_cfe_script(task_name, src_name, tgt_name):
    """
    Create the concept feature encoding script (from template).
    """
    script_path = os.path.join("Tasks", task_name, f"{task_name}_cfe.py")
    content_path = os.path.join("Tasks", "template_cfe.py")
    if not os.path.exists(script_path):
        if os.path.exists(content_path):
            with open(content_path, "r", encoding="utf-8") as source:
                content = source.read()
                content = content.replace("src_name", src_name)
                content = content.replace("tgt_name", tgt_name)
                content = content.replace("task_name", task_name)
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"🖍️ Created script: {script_path}")
        else:
            print("⚠️ Template script not found at Tasks/template_script.py. No script created.")

def create_task_script(task_name, src_name, tgt_name):
    """
    Create the training/evaluation script (from template) for the given task.
    """
    script_path = os.path.join("Tasks", task_name, f"{task_name}.py")
    content_path = os.path.join("Tasks", "template_script.py")
    if not os.path.exists(script_path):
        if os.path.exists(content_path):
            with open(content_path, "r", encoding="utf-8") as source:
                content = source.read()
                content = content.replace("src_name", src_name)
                content = content.replace("tgt_name", tgt_name)
                content = content.replace("task_name", task_name)
                
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"🖍️ Created script: {script_path}")
        else:
            print("⚠️ Template script not found at Tasks/template_script.py. No script created.")

# ================================
# 🚀 Task Preparation Pipeline
# ================================
def prepare_task(task_name):
    """
    Full pipeline for creating a new BioGITOM task.
    - Create directories
    - Upload ontologies
    - Generate scripts
    - Encode and generate negatives
    - Run the task script
    """
    print(f"\n🚀 Creating task: {task_name}")
    data_dir, dataset_dir, refs_dir = create_directories(task_name)

    src_filename = upload_ontology("🧴 Upload the SOURCE ontology (.owl)", dataset_dir)
    tgt_filename = upload_ontology("🧖 Upload the TARGET ontology (.owl)", dataset_dir)
    src_name = os.path.splitext(src_filename)[0]
    tgt_name = os.path.splitext(tgt_filename)[0]

    create_cfe_script(task_name, src_name, tgt_name)
    
    print(f"\n🚀 Running Concept Features Encoder (CFE): Launching script: Tasks/{task_name}/{task_name}_cfe.py")

    os.system(f"python Tasks/{task_name}/{task_name}_cfe.py")

    upload_and_copy_train(refs_dir, task_name, data_dir, src_name, tgt_name)
 
    create_task_script(task_name, src_name, tgt_name)

    print(f"\n🚀 Launching script: Tasks/{task_name}/{task_name}.py")
    os.system(f"python Tasks/{task_name}/{task_name}.py")


# ================================
# 🏁 Entry Point
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new BioGITOM task")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., 'omim2ordo')")
    args = parser.parse_args()
    prepare_task(args.task)
