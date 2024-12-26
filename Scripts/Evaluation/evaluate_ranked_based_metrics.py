import pandas as pd
import argparse  # Import argparse for command-line argument parsing

try:
    # Import necessary modules from DeepOnto
    from deeponto.align.evaluation import AlignmentEvaluator
    from deeponto.utils import read_table

    print("Necessary imports successful!")

    def compute_mrr_and_hits(reference_file, predicted_file, k_values=[1, 5, 10]):
        """
        Compute MRR and Hits@k for ontology matching predictions based on a reference file.

        Args:
            reference_file (str): Path to the reference file (test.cands.tsv format).
            predicted_file (str): Path to the predictions file with scores.
            k_values (list): List of k values for Hits@k.

        Returns:
            dict: A dictionary containing MRR and Hits@k metrics.
        """
        # Load the reference mappings from the specified file.
        test_candidate_mappings = read_table(reference_file).values.tolist()

        # Load the predicted scores from the predictions file.
        predicted_data = pd.read_csv(predicted_file, sep="\t")
        predicted_data["Score"] = predicted_data["Score"].apply(lambda x: float(x.strip("[]")))

        # Create a dictionary to quickly look up scores for (source, target) pairs.
        score_lookup = {}
        for _, row in predicted_data.iterrows():
            score_lookup[(row["SrcEntity"], row["TgtEntity"])] = row["Score"]

        ranking_results = []

        # Process each reference mapping in the test candidate set.
        for src_ref_class, tgt_ref_class, tgt_cands in test_candidate_mappings:
            tgt_cands = eval(tgt_cands)  # Convert string representation to list
            scored_cands = []
            for tgt_cand in tgt_cands:
                matching_score = score_lookup.get((src_ref_class, tgt_cand), -1e9)
                scored_cands.append((tgt_cand, matching_score))

            # Sort candidates by score in descending order.
            scored_cands = sorted(scored_cands, key=lambda x: x[1], reverse=True)
            ranking_results.append((src_ref_class, tgt_ref_class, scored_cands))

        # Compute MRR and Hits@k metrics.
        total_entities = len(ranking_results)
        reciprocal_ranks = []
        hits_at_k = {k: 0 for k in k_values}

        for src_entity, tgt_ref_class, tgt_cands in ranking_results:
            ranked_candidates = [candidate[0] for candidate in tgt_cands]
            if tgt_ref_class in ranked_candidates:
                rank = ranked_candidates.index(tgt_ref_class) + 1
                reciprocal_ranks.append(1 / rank)
                for k in k_values:
                    if rank <= k:
                        hits_at_k[k] += 1
            else:
                reciprocal_ranks.append(0)

        mrr = sum(reciprocal_ranks) / total_entities
        hits_at_k = {k: hits / total_entities for k, hits in hits_at_k.items()}

        return {"MRR": mrr, "Hits@k": hits_at_k}

    def evaluate_ranked_metrics(task, src_ent, tgt_ent):
        """
        Run evaluation for a specific ontology matching task.

        Args:
            task (str): Task name.
            src_ent (str): Source ontology entity.
            tgt_ent (str): Target ontology entity.
        """
        reference_file = f"Datasets/{task}/refs_equiv/test.cands.tsv"
        predicted_file = f"Tasks/{task}/Results/{task}_all_predictions_ranked.tsv"

        print(f"Starting evaluation for ranked-based metrics in the ontology matching task: {task}...")
        metrics = compute_mrr_and_hits(reference_file, predicted_file)
        print(f"MRR: {metrics['MRR']}")
        print(f"Hits@k: {metrics['Hits@k']}")

    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Run BioGITOM Ontology Matching Evaluation")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., 'omim2ordo')")
    parser.add_argument("--src_ent", type=str, required=True, help="Source ontology name (e.g., 'omim')")
    parser.add_argument("--tgt_ent", type=str, required=True, help="Target ontology name (e.g., 'ordo')")

    # Parse the arguments
    args = parser.parse_args()

    # Call the evaluation function with parsed arguments
    evaluate_ranked_metrics(args.task, args.src_ent, args.tgt_ent)

except Exception as e:
    # Handle and report any errors that occur during execution.
    print(f"An error occurred: {e}")
