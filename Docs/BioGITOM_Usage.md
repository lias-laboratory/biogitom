# BioGITOM Task Configuration Guide

This documentation outlines how to work with ontology matching tasks in **BioGITOM**, whether by launching predefined tasks or creating new ones.

---

## Predefined Matching Tasks

BioGITOM includes several ready-to-run ontology matching tasks:

| Task ID       | Description                                             |
|---------------|---------------------------------------------------------|
| `omim2ordo`   | Aligns the **OMIM** ontology with the **ORDO** ontology |
| `body`        | Matches **SNOMED (body)** with **FMA (body)**           |
| `ncit2doid`   | Maps the **NCIT** ontology to the **DOID** ontology     |
| `neoplas`     | Aligns **SNOMED (neoplasms)** with **NCIT (neoplasms)** |
| `pharm`       | Matches **SNOMED (pharma)** with **NCIT (pharma)**      |

Each task is organized under the `Tasks/<task_name>/` directory and includes all necessary components:
- Implementation scripts
- Embeddings
- Reference mappings

**Execution:** To execute a specific task, run the following command in your terminal:

   ```bash
   python run_biogitom.py --task <task>
   ```
   Replace `<task>` with the appropriate task name. 

   Example: To run the omim2ordo task:

   ```bash
   python run_biogitom.py --task omim2ordo 
   ```
---

## Create a New Ontology Matching Task

BioGITOM provides two flexible modes to create your own task:

###  Option 1: Create a Task from Ontologies *(Full Pipeline)*

If you **only have the ontologies and a training set**, BioGITOM will handle all steps automatically:

**You need:**
- Source ontology (.owl)
- Target ontology (.owl)
- Reference mappings file (`train.tsv`)

**The system will:**
- Generate concept embeddings using Sentence-SapBERT
- Create adjacency matrices (`_adjacence.csv`)
- Launch the full matching pipeline

📗 [Guide: Create Task from Raw Ontologies](./BioGITOM_New_Task.md)

---

###  Option 2: Use Precomputed Embeddings

If you **already have some preprocessed data**, you can skip embedding and graph generation.

**You need:**
- `*_emb.csv` files for both ontologies
- `*_classes.json` files for both ontologies
- `*_adjacence.csv` files
- Reference mappings file (`train.tsv`)

**The system will:**
- Encode training data using your `classes.json`
- Generate negative samples
- Run the matching script pipeline

📗 [Guide: Create Task with Precomputed Embeddings](./BioGITOM_New_Task__with_Embeddings_Guide.md)

---

## Notes

- `train.tsv` must use tab as separator and contain `SrcEntity` and `TgtEntity` columns

---

## **Outputs and Results Visualization**

Executing a task generates the following output files in TSV format:
    
  - {task}_all-predictions: Includes all candidate mappings with their respective mapping scores.
  - {task}_matching_results: Contains the filtered mappings based on a predefined threshold.
  - {task}_all_predictions_ranked: Lists all candidate mappings considered for rank-based metrics calculation, along with their mapping scores.
  - {task}_formatted_predictions: Provides predictions reformatted specifically for calculating rank-based metrics.

**Instructions for Evaluating Ontology Matching Results**
   
 **Evaluating Global Metrics (Precision, Recall, F1-Score) and ranking metrics (Precision@1, Recall@1, F1-Score@1)** :
      To evaluate the generated mappings included in the {task}_matching_results.tsv file in terms of global metrics such as Precision, Recall, and F1-Score, use the following command:
  
  ```bash
  python Scripts/Evaluation/evaluate_global_metrics.py --task <task> --src_ent <src_ent> --tgt_ent <tgt_ent>

  ```
  Replace `<task>`, `<src_ent>`, and `<tgt_ent>` with the corresponding task name, source ontology name, and target ontology name.

  Example:  

  ```bash
 python Scripts/Evaluation/evaluate_global_metrics.py --task omim2ordo --src_ent omim --tgt_ent ordo
  ```
**Evaluating Local Ranked-Based Metrics (MRR and Hits@k)**: To assess the {task}_matching_results.tsv file in terms of ranked-based metrics such as Mean Reciprocal Rank (MRR) and Hits@k, use the following command:
   
```bash
python Scripts/Evaluation/evaluate_ranked_based_metrics.py --task <task> --src_ent <src_ent> --tgt_ent <tgt_ent> 
```
Replace `<task>`, `<src_ent>`, and `<tgt_ent>` with the corresponding task name, source ontology name, and target ontology name.

Example:
```bash
python Scripts/evaluate_ranked_based_metrics.py --task omim2ordo --src_ent omim --tgt_ent ordo
```
**Viewing Logs for a Specific Task :** Task execution logs can be found in the Tasks/<task> directory. Each task folder includes a file named Results_log_{task}.md, which contains detailed logs of the task's execution.
