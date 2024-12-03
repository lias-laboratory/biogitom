import os
import argparse

def execute_task_script(task_name, src_ent, tgt_ent):
    """
    Executes a task-specific script with the given arguments.

    Args:
        task_name (str): Name of the task folder (e.g., 'omim2ordo').
        src_ent (str): Source ontology name (e.g., 'omim').
        tgt_ent (str): Target ontology name (e.g., 'ordo').
    """
    script_path = f"Tasks/{task_name}/{task_name}.py"
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Task script '{script_path}' not found.")
    
    # print the task name and path
    print(f"Executing task '{task_name}' from '{script_path}'...")

    # Build the script as a string with argument injection
    with open(script_path, "r") as script_file:
        script_content = script_file.read()

    # Inject variables
    exec_globals = {
        "src_ent": src_ent,
        "tgt_ent": tgt_ent,
        "task": task_name,
    }
    exec(script_content, exec_globals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BioGITOM tasks")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., 'omim2ordo')")
    parser.add_argument("--src_ent", type=str, required=True, help="Source ontology name (e.g., 'omim')")
    parser.add_argument("--tgt_ent", type=str, required=True, help="Target ontology name (e.g., 'ordo')")
    args = parser.parse_args()

    execute_task_script(args.task, args.src_ent, args.tgt_ent)
