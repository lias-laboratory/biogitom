import os
import tarfile
from tqdm import tqdm

# Set the root directory and output tar file
root_dir = "./"
output_tar = "data_biogitom.tar.gz"

# Define allowed extensions
allowed_extensions = {".owl", ".tsv", ".csv", ".json"}

# Create a list of files to add to the tar file
files_to_add = []

# Walk through the root directory and collect files with allowed extensions
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if any(filename.endswith(ext) for ext in allowed_extensions):
            file_path = os.path.join(dirpath, filename)
            files_to_add.append(file_path)

# Create a tarfile object
with tarfile.open(output_tar, "w:gz") as tar:
    # Initialize tqdm for progress bar
    with tqdm(total=len(files_to_add), desc="Compressing files") as pbar:
        # Add files to the tar archive
        for file_path in files_to_add:
            tar.add(file_path, arcname=os.path.relpath(file_path, root_dir))
            pbar.update(1)  # Update the progress bar after each file

print(f"Compressed files to {output_tar}.")
