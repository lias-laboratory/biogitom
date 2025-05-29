import os
import argparse

def execute_task_script(task_name):
    task_path = f"Tasks/{task_name}/{task_name}.py"
    if not os.path.isfile(task_path):
        print(f"Error: Task file '{task_path}' not found.")
        return

    with open(task_path, encoding='utf-8') as script_file:  # 🔧 FIX HERE
        script_content = script_file.read()

    exec(script_content, globals())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BioGITOM tasks")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., 'omim2ordo')")
    args = parser.parse_args()

    execute_task_script(args.task)
