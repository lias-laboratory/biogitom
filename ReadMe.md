# BioGITOM: Matching Biomedical Ontologies with Graph Isomorphism Transformer

**BioGITOM** is a specialized ontology matching system designed to tackle the increasing complexity and heterogeneity of biomedical ontologies. While these ontologies are essential for organizing and standardizing domain knowledge, their structural, semantic, and terminological disparities often pose significant challenges to data integration and knowledge alignment.

To address these challenges, BioGITOM enhances domain-specific embeddings extracted using the SapBERT model with structural features derived from the Graph Isomorphism Transformer (GIT)—a hybrid model that integrates Graph Isomorphism Networks (GINs) and Graph Transformers (GTs). This dual-focus approach effectively captures both structural and semantic intricacies, enabling the generation of more accurate concept representations and, consequently, relevant and precise ontology mappings.

## Repository Structure

- `Scripts/`: This directory contains scripts for candidate generation, training set construction, semantic embedding generation using SapBERT, and the implementation and training processes for the GIT and Gated Network architectures.
- `Datasets/`: This folder includes the Bio-ML datasets, comprising ontologies and their corresponding reference alignments.
- `Experiments/`: This section holds the experimental results, demonstrating the rationale behind the selection of the BERT model (i.e., SapBERT), the optimal number of negative examples in the training set, and findings from the ablation study.
- `Tasks/`: This directory contains the predefined tasks of the Bio-ML track, each organized into a dedicated subdirectory that includes:
   - Task-specific data files: Input data necessary for the task execution.
   - Scripts: Implementations provided in Python (.py) and Jupyter Notebook (.ipynb) formats, along with execution logs documented in Markdown (.md) files.
   - Results: Execution outputs, including all predictions and filtered prediction results.
- `download_data.py`: This script automates the process of downloading, extracting, and organizing the required data for each task from a remote server. 
- `run_biogitom.py`: This script serves as a central interface for executing task-specific Python scripts. Each task is organized in its designated subdirectory within the Tasks/ folder. The script dynamically identifies and executes the appropriate task script based on user-provided input parameters.
- `requirements.txt`: This file lists all the Python packages and their specific versions required to run the BioGITOM framework. It ensures compatibility and consistency across environments.
- `dictionary.json`: This JSON maps ontology files to their respective namespace URIs and synonym properties, enabling standardized synonym extraction for ontology processing tasks.

## Installation

For comprehensive instructions on installing and setting up BioGITOM, please consult the [Installation Guide](./Docs/BioGITOM_Installation.md). The guide offers detailed, step-by-step directions for installing all required dependencies, configuring the environment, and downloading as well as extracting the necessary data files.

## Reproducibility

To reproduce the published results of BioGITOM, please consult the [Reproducibility Guide](./Docs/BioGITOM_Usage.md), which provides comprehensive usage instructions and illustrative examples.

## License

BioGITOM is released under the MIT License. See [LICENSE](LICENSE) for more information.

## Contributors

- Samira Oulefki, Dep. of Artificial Intelligence and Data Sciences, USTHB, Algeria
- Lamia Berkani, Dep. of Artificial Intelligence and Data Sciences, USTHB, Algeria
- Nassim Boudjenah, Dep. of Artificial Intelligence and Data Sciences, USTHB, Algeria
- [Ladjel Bellatreche](https://www.lias-lab.fr/fr/members/bellatreche/), LIAS, ISAE-ENSMA, France
- Aicha Mokhtari, Dep. of Artificial Intelligence and Data Sciences, USTHB, Algeria

## Contact

For more information about BioGITOM, please contact [soulefki@usthb.dz](mailto:soulefki@usthb.dz).
