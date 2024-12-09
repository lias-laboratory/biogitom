# BioGITOM: A Graph-Based Ontology Matching System for Biomedical Data Interoperability



BioGITOM is a specialized OM system developed to address the increasing complexity and heterogeneity in biomedical ontologies. Its core purpose is to ensure effective integration and alignment of disparate ontologies, which is essential for improving data interoperability in biomedical research. BioGITOM is particularly designed to manage the unique challenges posed by the biomedical field, where ontologies often differ in structure and semantics. By combining advanced graph-based techniques, BioGITOM is able to produce more accurate mappings between concepts from different ontologies, thereby supporting enhanced data sharing and collaboration across systems.

## Repository Structure

- `Codes/`: This directory contains Jupyter notebooks for key stages of the workflow, including candidate generation, training set creation, and embeddings generation using SapBERT.
- `Datasets/`: This folder includes the Bio-ML datasets, comprising ontologies and their corresponding reference alignments.
= `Experiments/`: This section holds the experimental results, demonstrating the rationale behind the selection of the BERT model, the optimal number of negative examples in the training set, and findings from the ablation study.
- Tasks/`: contains on task-specific data, results, and notebooks.
- `download_data.py`: Script to download the required data for the project.
- `run_biogitom.py`: Main script to execute the full pipeline.
- `requirements.txt`: Required libraries for the project.
- `dictionary.json`: Contains the links to entities.


## Installation

BioGITOM can be installed using the following steps:

1. Clone the BioGITOM repository from GitHub:

```bash
git clone https://github.com/lias-laboratory/biogitom.git
```

2. Change the directory to the BioGITOM folder:

```bash
cd biogitom
```

3. Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

4. Download the required data using the following command:

    - Run the following script to download and extract the required data files:

    ```bash
    python download_data.py
    ```
    - This script will:
        - Download a compressed archive of the required data files from the remote server.
        - Extract the files into a temporary temp/ directory.
        - Automatically move the files to their correct locations within the repository (e.g., Datasets/, Tasks/).
        - Delete the temporary temp/ directory.
5. Run the main script to execute the task file:

```bash
python run_biogitom.py --task <task> --src_ent <src_ent> --tgt_ent <tgt_ent>
```

example:

```bash
python run_biogitom.py --task omim2ordo --src_ent omim --tgt_ent ordo
```

## List of Tasks and Entities

The following tasks and entities are supported in BioGITOM (there are 5):

- `omim2ordo`: **omim** to **ordo**
- `body`: **snomed.body** to **fma.body**
- `ncit2doid`: **ncit** to **doid**
- `neoplas`: **snomed.neoplas** to **ncit.neoplas**
- `pharm`: 

## Usage

BioGITOM can be used to perform ontology matching tasks in the biomedical domain. The system is designed to support various stages of the ontology matching process, including candidate generation, training, and integration with SapBERT. The main script `run_biogitom.py` can be used to execute the full pipeline for ontology matching tasks. The script takes input parameters such as the source and target ontologies, the type of matching task (e.g., equivalence or subsumption).

## License

BioGITOM is released under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact

For more information about BioGITOM, please contact [email](email).
