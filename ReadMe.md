# BioGITOM: Matching Biomedical Ontologies with Graph Isomorphism Transformer

**BioGITOM** is a specialized ontology matching system designed to tackle the increasing complexity and heterogeneity of biomedical ontologies. While these ontologies are essential for organizing and standardizing domain knowledge, their structural, semantic, and terminological disparities often pose significant challenges to data integration and knowledge alignment.

To address these challenges, BioGITOM enhances domain-specific embeddings extracted using the SapBERT model with structural features derived from the Graph Isomorphism Transformer (GIT)—a hybrid model that combines Graph Neural Networks (GNNs) and Graph Transformers. This dual-focus approach effectively captures both structural and semantic intricacies, enabling the generation of more accurate concept representations and, consequently, relevaant and precise ontology mappings.

## Repository Structure

- `Codes/`: This directory contains Jupyter notebooks for key stages of the workflow, including candidate generation, training set creation, and embeddings generation using SapBERT.
- `Datasets/`: This folder includes the Bio-ML datasets, comprising ontologies and their corresponding reference alignments.
- `Experiments/`: This section holds the experimental results, demonstrating the rationale behind the selection of the BERT model, the optimal number of negative examples in the training set, and findings from the ablation study.
- `Tasks/`: This directory contains the predefined tasks of the Bio-ML track, with each task organized into a subdirectory that includes:
  - Task-specific data files: Input data required for the task.
  - Scripts: Implementations in Python (`.py`) and Jupyter Notebook (`.ipynb`) formats.
  - Results: Evaluation outputs documented in Markdown (`.md`) files.

- `download_data.py`: This script automates the process of downloading, extracting, and organizing the required data for the project from a remote server. 
- `run_biogitom.py`: This script facilitates the execution of task-specific Python scripts in the BioGITOM framework. Each task resides in its dedicated directory within the Tasks/ folder, and the script dynamically loads and executes the relevant task script based on user input.
- `requirements.txt`: This file lists all the Python packages and their specific versions required to run the BioGITOM framework. It ensures compatibility and consistency across environments.
- `dictionary.json`: This JSON maps ontology files to their respective namespace URIs and synonym properties, enabling standardized synonym extraction for ontology processing tasks.


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

4. Download the required dataset using the following command:

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

## List of Predefined Tasks

BioGITOM includes the following predefined ontology matching tasks:

1. **`omim2ordo`**: Aligns the **OMIM** ontology with the **ORDO** ontology.  
2. **`body`**: Matches the **SNOMED.body** ontology with the **FMA.body** ontology.  
3. **`ncit2doid`**: Maps the **NCIT** ontology to the **DOID** ontology.  
4. **`neoplas`**: Aligns the **SNOMED.neoplas** ontology with the **NCIT.neoplas** ontology.  
5. **`pharm`**: Matches the **SNOMED.pharm** ontology with the **NCIT.pharm** ontology.
   
Each task is organized under the Tasks/ directory, containing all necessary resources such as implementation scripts, and results.

## Usage

BioGITOM can be used to perform ontology matching tasks in the biomedical domain. The system is designed to support various stages of the ontology matching process, including candidate generation, training, and integration with SapBERT. The main script `run_biogitom.py` can be used to execute the full pipeline for ontology matching tasks. The script takes input parameters such as the source and target ontologies, the type of matching task (e.g., equivalence or subsumption).

## License

BioGITOM is released under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact

For more information about BioGITOM, please contact [soulefki@usthb.dz](mailto:soulefki@usthb.dz).
