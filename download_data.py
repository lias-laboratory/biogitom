import os
import requests
from tqdm import tqdm
import tarfile
import shutil


# URL of the compressed file
DATA_URL = "https://www.lias-lab.fr/ftppublic/research/biogitom/data_biogitom.tar.gz"
ZIP_FILE = "data_biogitom.tar.gz"
TARGET_DIR = "temp"

def download_file(url, dest):
    print("Downloading data files...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        content_type = r.headers.get('Content-Type')
        if 'gzip' not in content_type:
            raise ValueError(f"Expected a gzip file but got {content_type}. Check the URL.")
        
        total_size = int(r.headers.get('content-length', 0))  # Get total file size
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as progress_bar:
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress_bar.update(len(chunk))
    print("Download completed.")

def extract_file(tar_file, target_dir):
    print("Extracting files...")
    with tarfile.open(tar_file, 'r:gz') as tar:
        members = tar.getmembers()  # List of files in the tarball
        # Initialize tqdm progress bar
        with tqdm(total=len(members), unit='file', desc="Extracting") as progress_bar:
            for member in members:
                tar.extract(member, target_dir)
                progress_bar.update(1)  # Update progress bar for each file
    print("Extraction completed.")


def move_files(source_dir):
    # Walk through all files in the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            old_path = os.path.join(root, file)  # Full path to the file
            # Strip "temp/" prefix to get the relative path
            if old_path.startswith(source_dir + os.sep):
                new_path = old_path[len(source_dir + os.sep):]
            else:
                new_path = old_path  # Fallback, though unlikely

            # Create the target directory if it doesn't exist
            target_path = os.path.join(".", new_path)  # Move to current working directory structure
            target_dir = os.path.dirname(target_path)
            os.makedirs(target_dir, exist_ok=True)

            # Check if the file already exists at the new location
            if os.path.exists(target_path):
                print(f"File already exists, removing: {target_path}")
                os.remove(target_path)

            # Move the file
            shutil.move(old_path, target_path)
            print(f"{old_path} -> {target_path}")

    # Remove the now-empty source directory
    shutil.rmtree(source_dir)
    print(f"Removed directory: {source_dir}")


def main():

    # Download and extract the data files
    if not os.path.exists(ZIP_FILE):
        download_file(DATA_URL, ZIP_FILE)
    else:
        print("Zip file already exists. Skipping download.")

    # Create the target directory if it doesn't exist
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    extract_file(ZIP_FILE, TARGET_DIR)

    # Move the files to their correct locations
    move_files(TARGET_DIR)

    # Remove the zip file after extraction
    os.remove(ZIP_FILE)

    print("Data download and extraction completed.")

if __name__ == "__main__":
    main()
