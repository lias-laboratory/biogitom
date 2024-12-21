import sys
import os
import argparse  # Import argparse for command-line argument parsing
try:
    # Import necessary modules from DeepOnto after starting the JVM
    from deeponto.onto import Ontology
    from deeponto.align.oaei import get_ignored_class_index, remove_ignored_mappings
    from deeponto.align.evaluation import AlignmentEvaluator
    from deeponto.align.mapping import ReferenceMapping, EntityMapping

    print("All imports successful!")

    # Function to evaluate global metrics for ontology matching
    def evaluate_Global_Metrics(task, src_ent, tgt_ent):
        """
        Evaluate global metrics (Precision, Recall, F1) for ontology matching.

        Args:
            task (str): Name of the task (e.g., ncit2doid).
            src_ent (str): Source ontology file name (without extension).
            tgt_ent (str): Target ontology file name (without extension).
        """
        print(f"Starting evaluation for global metrics in the ontology matching task: {task}...")

        # Define paths to dataset and result files
        dataset_dir = f"Datasets/{task}"  # Directory containing the task-specific dataset
        prediction_path = f"Tasks/{task}/Results/{task}_matching_results.tsv"  # Path to the predictions file
        test_file = f"Datasets/{task}/refs_equiv/test.tsv"  # Path to the reference mappings (ground truth)

        # Load the source and target ontologies using DeepOnto's Ontology class
        src_onto = Ontology(f"{dataset_dir}/{src_ent}.owl")  # Source ontology
        tgt_onto = Ontology(f"{dataset_dir}/{tgt_ent}.owl")  # Target ontology

        # Check if the predictions file exists
        if not os.path.exists(prediction_path):
            print(f"Error: Predictions file not found at {prediction_path}")
            return

        # Check if the reference file (ground truth) exists
        if not os.path.exists(test_file):
            print(f"Error: Test file not found at {test_file}")
            return

        # Print absolute paths for debugging purposes
        print(f"Prediction Path: {os.path.abspath(prediction_path)}")
        print(f"Test File Path: {os.path.abspath(test_file)}")

        # Retrieve ignored classes from both source and target ontologies
        ignored_class_index = get_ignored_class_index(src_onto)  # Get ignored classes from source ontology
        ignored_class_index.update(get_ignored_class_index(tgt_onto))  # Add ignored classes from target ontology

        # Read predicted mappings from the predictions file
        preds = EntityMapping.read_table_mappings(prediction_path)

        # Read reference mappings (ground truth) from the test file
        refs = ReferenceMapping.read_table_mappings(test_file)

        # Remove mappings that involve ignored classes
        preds = remove_ignored_mappings(preds, ignored_class_index)

        # Calculate evaluation metrics: Precision, Recall, F1-Score
        results = AlignmentEvaluator.f1(preds, refs)

        # Convert predictions and references to tuples for intersection calculation
        preds2 = [p.to_tuple() for p in preds]
        refs2 = [r.to_tuple() for r in refs]
        correct = len(set(preds2).intersection(set(refs2)))  # Count correct predictions

        # Print the evaluation results
        print(f"Number of Correct Predictions: {correct}")
        print(f"Precision, Recall, F1-Score: {results}")

    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Run BioGITOM Ontology Matching Evaluation")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., 'omim2ordo')")
    parser.add_argument("--src_ent", type=str, required=True, help="Source ontology name (e.g., 'omim')")
    parser.add_argument("--tgt_ent", type=str, required=True, help="Target ontology name (e.g., 'ordo')")

    # Parse the arguments
    args = parser.parse_args()

    # Call the evaluation function with parsed arguments
    evaluate_Global_Metrics(args.task, args.src_ent, args.tgt_ent)

except Exception as e:
    # Handle and display any errors that occur during execution
    print(f"An error occurred: {e}")

