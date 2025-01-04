## Reproducibility
1. **Usage** 

   To execute a specific task using BioGITOM, run the following command in your terminal:

   ```bash
   python run_biogitom.py --task <task>
   ```
   Replace `<task>` with the appropriate task name. 

   Example: To run the omim2ordo task:

   ```bash
   python run_biogitom.py --task omim2ordo 
   ```

   ### List of Predefined Tasks

     BioGITOM includes the following predefined ontology matching tasks:

      1. **`omim2ordo`**: Aligns the **omim** ontology with the **ordo** ontology.  
      2. **`body`**: Matches the **snomed.body** ontology with the **fma.body** ontology.  
      3. **`ncit2doid`**: Maps the **ncit** ontology to the **doid** ontology.  
      4. **`neoplas`**: Aligns the **snomed.neoplas** ontology with the **ncit.neoplas** ontology.  
      5. **`pharm`**: Matches the **snomed.pharm** ontology with the **ncit.pharm** ontology.
   
     Each task is organized under the Tasks/ directory, containing all necessary resources such as implementation scripts, and results.
     
2. **Outputs and Results Visualization**

   Executing a task generates the following output files in TSV format:
    
    - {task}_all-predictions: Includes all candidate mappings with their respective mapping scores.
    - {task}_matching_results: Contains the filtered mappings based on a predefined threshold.
    - {task}_all_predictions_ranked: Lists all candidate mappings considered for rank-based metrics calculation, along with their mapping scores.
    - {task}_formatted_predictions: Provides predictions reformatted specifically for calculating rank-based metrics.

   **Instructions for Evaluating Ontology Matching Results**
   
    - Evaluating Global Metrics (Precision, Recall, F1-Score)
      To evaluate the generated mappings included in the {task}_matching_results.tsv file in terms of global metrics such as Precision, Recall, and F1-Score, use the following command:
  
       ```bash
       python Scripts/evaluate_global_metrics.py --task <task> --src_ent <src_ent> --tgt_ent <tgt_ent>

       ```
      Replace `<task>`, `<src_ent>`, and `<tgt_ent>` with the corresponding task name, source ontology name, and target ontology name.

      Example:  
      ```bash
       python Scripts/evaluate_global_metrics.py --task omim2ordo --src_ent omim --tgt_ent ordo
      ```
    - Evaluating Ranked-Based Metrics (MRR and Hits@k)
      To assess the {task}_matching_results.tsv file in terms of ranked-based metrics such as Mean Reciprocal Rank (MRR) and Hits@k, use the following command:
   
       ```bash
        python Scripts/evaluate_ranked_based_metrics.py --task <task> --src_ent <src_ent> --tgt_ent <tgt_ent> 
       ```
      Replace `<task>`, `<src_ent>`, and `<tgt_ent>` with the corresponding task name, source ontology name, and target ontology name.
      Example:
      ```bash
       python Scripts/evaluate_ranked_based_metrics.py --task omim2ordo --src_ent omim --tgt_ent ordo
      ```
    - Viewing Logs for a Specific Task
        Task execution logs can be found in the Tasks/<task> directory. Each task folder includes a file named Results_log_{task}.md, which contains detailed logs of the task's execution.