# Guide: Creating a New Ontology Matching Task in BioGITOM (Full Pipeline)

This guide explains how to create a new ontology matching task in BioGITOM using the script create_new_task.py.
This version assumes that you will upload the source and target ontology files (in .owl format) and a training reference file (train.tsv). The system will then automatically:

- Prepare the required directory structure and task-specific scripts

- Generate embeddings for all ontology concepts

- Create adjacency matrices to train the GIT model

- Extract class files (_classes.json) to encode concept URIs

- Automatically generate negative training examples

- Launch the complete matching pipeline

This workflow provides a streamlined setup for new ontology alignment tasks, ensuring consistency and scalability across experiments.

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

### 2. Upload Ontologies

You will be prompted to provide paths to:

* The source ontology (e.g., `ncit.owl`)
* The target ontology (e.g., `mondo.owl`)

These will be copied to:

```
Datasets/<task_name>/
```

---

### 3.  Automatically Generate Embeddings

Once the ontologies are uploaded, the script will automatically perform the following steps:

- Generate semantic embeddings for all ontology concepts using a pretrained encoder (Sentence-SapBERT)

- Create adjacency matrices (CSV) based on subClassOf relations to train the GIT model

- Extract class files (_classes.json) to map concept URIs to their labels/synonyms for encoding

These files are generated and saved in:

```
Tasks/<task_name>/Data/
```

Note: This step may take a significant amount of time depending on the number of ontology concepts to process.

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

### 6. Set Top-k Value

You will be asked to provide the value of **k**, used later for retrieving top-k candidates using FAISS.

---

### 7. Auto-Generate Python Scripts

Two scripts will be generated from templates:

* `Tasks/<task_name>/<task_name>_cfe.py`
* `Tasks/<task_name>/<task_name>.py`

The placeholders in the templates will be automatically replaced.

---

### 8. Automatic Execution

The script will then run:

```bash
python Tasks/<task_name>/<task_name>.py
```

This launches the full pipeline including model training and evaluation.

---

## Directory Structure

```
BioGITOM/
├── Tasks/
│   └── ncit2mondo/
│       ├── Data/
│       │   ├── ncit_emb.csv
│       │   ├── ncit_classes.json
│       │   ├── mondo_emb.csv
│       │   ├── mondo_classes.json
│       │   ├── ncit2mondo_train.csv
│       │   └── ncit2mondo_train.encoded.csv
│       ├── Results/
│       ├── ncit2mondo.py
│       └── ncit2mondo_cfe.py
├── Datasets/
│   └── ncit2mondo/
│       ├── ncit.owl
│       ├── mondo.owl
│       └── refs_equiv/
│           └── train.tsv
└── biogitom/
    └── create_new_task.py
```

---

## Notes

* Scripts are only generated if the target paths do not already exist.

---

## Troubleshooting

* **Missing columns**: Ensure `train.tsv` is tab-separated with correct headers.

* **Ontology load failure**: Make sure `.owl` files are in RDF/XML format.

