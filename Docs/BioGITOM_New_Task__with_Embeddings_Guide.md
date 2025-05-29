
# Guide: Creating a New Ontology Matching Task in BioGITOM Using Precomputed Embeddings

TThis guide describes how to create a new ontology matching task in BioGITOM using the script `create_new_task_with_embeddings.py`. This version assumes that you will upload the ontologies, class files, and embeddings manually.

---

## Requirements

Before running the script, ensure you have:

* Source and target ontologies in `.owl` format
* Class files in `.json` format mapping concept URIs to labels
* Embedding files (`*_emb.csv`) for both source and target
* A training file in `train.tsv` 

## Step-by-Step Instructions

### 1. Launch the Task Creation Script

```bash
python biogitom/create_new_task_with_embeddings.py --task <task_name>
```
Replace `<task_name>` with your task identifier, e.g., `ncit2mondo`.

---
Example:

```bash
python biogitom/create_new_task_with_embeddings.py --task ncit2mondo
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

### 3. Upload Class Files (`_classes.json`)
For each ontology (source and target), you need to upload a file named like:
- `ncit_classes.json`
- `mondo_classes.json`

**Format:**
```json
{
  "http://example.org/concept/001": ["Cancer", "Tumor"],
  "http://example.org/concept/002": ["Leukemia"]
}
```
Each key is a concept URI, and the value is a list of labels or synonyms.

---

### 4. Upload Embeddings Files
For each ontology (source and target), you must provide a .csv file containing the embeddings.

#### File Naming
The file must follow the naming convention:
ontologyname_emb.csv
(e.g., ncit_emb.csv, mondo_emb.csv, doid_emb.csv)

Each file will be automatically copied into the Tasks/<task_name>/Data/ folder created by the script.


**Format:**
The file must contain a header row with the following:
- The first column is an index generated using one ot encoding corresponding to classe.json file.
- Remaining columns must be floating-point values representing the embedding vector.

Example:
```csv
,0,1,2,...,767
0,0.12,0.05,0.33,...,0.09
1,0.11,0.02,0.36,...,0.08
```

---

### 5. Upload Training File

You will be prompted to upload the `train.tsv` file, containing known positive mappings.

**Format:** Tab-separated file with at least two columns:
```
SrcEntity<TAB>TgtEntity
```
Example:
```
http://ncit/123	http://mondo/987
```

It will be copied to:

```
Datasets/<task_name>/refs_equiv/train.tsv
```

---

### 6. Automatically Encodes Training Data & Generates Negatives

The script performs:

* Encoding of `SrcEntity` and `TgtEntity` as integers using `classes.json`
* Generation of 50 random negative mappings per source entity

It produces:

* `<task>_train.encoded.csv`
* `<task>_train.csv` (positives + negatives)

---

### 7. Enter `k` Value
You will be asked to provide the value of **k**, used later for retrieving top-k candidates using FAISS.

---

### 8. Auto-Generate Task Script

A file like `Tasks/<task>/<task>.py` is created from a template, automatically filled in with:
- Task name
- Source and target names
- The value of `k`

---

### 9. Auto-Launch the Task

The script will then run:

```bash
python Tasks/<task_name>/<task_name>.py
```

This launches the full pipeline including model training and evaluation.

---

## Example Directory Structure After Task Creation
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
│       └── ncit2mondo.py
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

* Embeddings must have the same ordering as the keys in `classes.json`.
* The encoded IDs correspond to row indices of embeddings.
* Scripts are only generated if the target paths do not already exist.

---

## Tips
- Always verify file naming: embeddings must include the ontology name and end with `_emb.csv`.
- Ensure concept URIs in `train.tsv` match those in `classes.json`.
- You can run the generated task script again later using:

```bash
python Tasks/<task>/<task>.py
```

---

## Troubleshooting

* **Missing columns**: Ensure `train.tsv` is tab-separated with correct headers.
* **Embedding files not found**: Filenames must match and end with `_emb.csv`.
* **Ontology load failure**: Make sure `.owl` files are in RDF/XML format.

