
---

# GNN-Match: A Modular Neural Framework for Ontology Matching

GNN-Match is a modular and fully configurable ontology matching system designed to address the increasing complexity and heterogeneity of ontologies. It combines semantic embeddings—generated using user-selected pretrained language models (PLMs) such as Sentence-BERT, BioBERT, SapBERT, or any other compatible encoder—with structural features learned through configurable graph neural network (GNN) architectures, including GCN (Graph Convolutional Network), GAT (Graph Attention Network), GIN (Graph Isomorphism Network), and GTN (Graph Transformer Network). The resulting semantic and structural representations are subsequently fused through a user-defined combination module, such as gated combination, concatenation, or addition.

This high degree of configurability enables researchers to tailor each stage of the pipeline—semantic encoding, structural propagation, and representation fusion—to their specific requirements. 

---

## 📁 Repository Structure

### `Datasets/`

This folder includes the Bio-ML track ontologies along with their corresponding reference alignments.


### `Tasks/`

Each ontology matching task is organized in a dedicated subdirectory containing:

* **Data**: ontologies, reference alignments, auxiliary files
* **Scripts**: `.py` 

### `download_data.py`

Automatically downloads, extracts, and organizes all required datasets for each task.

### `run_gnn_match.py`

Main entry point for executing task pipelines.
Automatically:

* detects available tasks,
* loads the corresponding configuration,
* and runs the end-to-end matching process.

### `requirements.txt`

Lists all Python dependencies (version-pinned) required to execute GNN-Match.

### `dictionary.json`

Maps ontology filenames to:

* namespace URIs
* synonym properties
  Ensures standardized synonym extraction across heterogeneous ontologies.

---

## 🚀 Installation

For detailed setup instructions, please refer to:

👉 **[Installation Guide](\Docs\GNN-Match_Installation.md)**

This guide includes:

* Environment configuration
* Dependency installation
* Data download & preparation

---

## 🔁 Reproducibility

All steps required to reproduce published results are detailed in:

👉 **[Usage Guide](\Docs\GNN-Match_Usage.md)**

Includes:

* Example commands


---

## 📜 License

**GNN-Match** is distributed under the **MIT License**.
See the full license text in **[LICENSE](LICENSE)**.

---

## 👥 Contributors

* **Samira Oulefki**, Dep. of AI & Data Sciences, USTHB, Algeria
* **Lamia Berkani**, Dep. of AI & Data Sciences, USTHB, Algeria
* **[Ladjel Bellatreche](https://www.lias-lab.fr/members/bellatreche/)**, LIAS, ISAE-ENSMA, France
---

## 📩 Contact

For questions or collaboration inquiries:
📧 **[soulefki@usthb.dz](mailto:soulefki@usthb.dz)**

---

