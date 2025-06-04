# Guide: Creating a New Ontology Matching Task in BioGITOM (Full Pipeline)

This guide explains how to create a new ontology matching task in BioGITOM using the script create_new_task.py.
This version assumes that you will upload the source and target ontology files (in .owl format) and a training reference file (train.tsv). The system will then automatically:

- Prepare the required directory structure and task-specific scripts

- Generate embeddings for all ontology concepts

- Create adjacency matrices to train the GIT model

- Extract class files (_classes.json) to encode concept URIs

- Automatically generate negative training examples

- Launch the complete matching pipeline

This workflow provides a streamlined setup for new ontology alignment tasks.

---

## Requirements

Before running the script, ensure you have:

* Source and target ontologies in `.owl` format
* A training file in `train.tsv` format with `SrcEntity` and `TgtEntity`


---

## Step-by-Step Usage

### 1. Launch the Script

```bash
python biogitom/create_new_task.py --task <task_name>
```
Replace `<task_name>` with your task identifier, e.g., `ncit2mondo`.

Example:

```bash
python biogitom/create_new_task.py --task ncit2mondo
```

---

### 2. Upload Ontology Files
During the setup, you will be prompted to upload the two ontologies required for the task:

- The source ontology (e.g., ncit.owl)

- The target ontology (e.g., mondo.owl)

📦 Format Requirements:

- Each file must be in .owl (RDF/XML) format.


These will be copied to:

```
Datasets/<task_name>/
```
---

### 3. Automatically Generate Embeddings

Once the source and target ontologies are uploaded, a script will be automatically generated from a predefined template:

- `Tasks/<task_name>/<task_name>_cfe.py`  
  → This script launches the **Concept Feature Encoder (CFE)**, which generates the necessary input features for the matching model. Specifically, it performs the following operations:

#### 🔧 Automated Steps:

- **Semantic Embedding Generation:**  
  Computes dense vector representations for all ontology concepts using the pretrained model **Sentence-SapBERT**.

- **Adjacency Matrix Construction:**  
  Builds CSV-formatted adjacency matrices based on `subClassOf` relationships, which are essential for the **Graph Isomorphism Transformer (GIT)** model.

- **Class File Extraction:**  
  Creates `*_classes.json` files that map each concept URI to its associated labels and synonyms, used for encoding.

#### 📁 Output Location:
All generated files are saved under:

```
Tasks/<task_name>/Data/
```

 **Note:** This step may take a significant amount of time depending on the number of ontology concepts to process.

---

### 4. Upload `train.tsv`

You will be prompted to upload a tab-separated file containing positive mappings:

```
SrcEntity[TAB]TgtEntity
```

Example:

```
http://ncit.org/C123	http://mondo.org/M456
```

It will be copied to:

```
Datasets/<task_name>/refs_equiv/train.tsv
```

---

### 5. Automatic Encoding and Negative Sampling

The script performs:

* Encoding of `SrcEntity` and `TgtEntity` as integers using `classes.json`
* Generation of 50 random negative mappings per source entity

It produces:

* `<task>_train.encoded.csv`
* `<task>_train.csv` (positives + negatives)

---

### 6. Auto-Generate Matching Script

A Python script will be automatically generated:

- `Tasks/<task_name>/<task_name>.py`  
  → This script orchestrates the training pipeline for the ontology matching task. It includes:

  - Training the **Graph Isomorphism Transformer (GIT)** model to capture structural features.
  - Training the **GatedCombination** model to fuse semanctic and graph-enhanced embeddings.
  - Generating mappings between source and target ontologies.

---

### 7. Automatic Execution

The script will then run:

```bash
python Tasks/<task_name>/<task_name>.py
```

This launches the full pipeline including model training and evaluation.

---

### 8. Set the Value of k
You will be prompted to enter a value for k, which controls the number of top-ranked target candidates retrieved for each source concept based on embedding similarity.

- The top-k candidates are selected using exact nearest neighbor search with FAISS, based on the L2 (Euclidean) distance between vectors.

- This step is crucial for generating a manageable and relevant candidate set for alignment.

- A larger k increases recall but may reduce precision or slow down further processing steps.

---

## 9. Generate top-k Candidate Mappings Using FAISS L2

After training, top-k candidate mappings are generated:

- Output saved to TSV with: `SrcEntity`, `TgtEntity`, `Score`

---

## 10.  Choose Mapping Selection Strategy
You are prompted to choose from:

- Greedy 1-to-1
- Relaxed Top-1 (margin-based)
- Both strategies (with optional evaluation if test set is available)


All generated mappings are saved to:

```
Tasks/<task_name>/Results/
```
 

This completes the end-to-end process for launching a new ontology matching task using BioGITOM with precomputed embeddings.

---

## Example Directory Structure After Task Creation
```
BiogitomFolder/
└── biogitom/
    ├── Tasks/
    │   └── ncit2mondo/
    │       ├── Data/
    │       │   ├── ncit_emb.csv
    │       │   ├── ncit_classes.json
    │       │   ├── mondo_emb.csv
    │       │   ├── mondo_classes.json
    │       │   ├── ncit_adjacence.csv
    │       │   ├── mondo_adjacence.csv
    │       │   ├── ncit2mondo_train.csv
    │       │   └── ncit2mondo_train.encoded.csv
    │       ├── Results/
    │       └── ncit2mondo.py
    ├── Datasets/
    │   └── ncit2mondo/
    │       ├── ncit.owl
    │       ├── mondo.owl
    │       └── refs_equiv/
    │           └── train.tsv
    └── create_new_task.py
    └── create_new_task_with_embeddings.py

```


## Notes

* Scripts are only generated if the target paths do not already exist.

---

## Troubleshooting

* **Missing columns**: Ensure `train.tsv` is tab-separated with correct headers.

* **Ontology load failure**: Make sure `.owl` files are in RDF/XML format.

