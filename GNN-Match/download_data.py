import os
import requests
from tqdm import tqdm
import tarfile
import shutil
import time

# URL of the compressed file
DATA_URL = " https://forge.lias-lab.fr/datasets/biogitom/data_biogitom.tar.gz"
ZIP_FILE = "data_biogitom.tar.gz"
TARGET_DIR = "temp"

def download_file(url, dest, max_retries=5):
    print("Downloading data files...")
    retries = 0

    while retries < max_retries:
        try:
            # Vérifier si un fichier partiel existe déjà
            local_file_size = 0
            if os.path.exists(dest):
                local_file_size = os.path.getsize(dest)

            # En-têtes HTTP pour reprendre le téléchargement
            headers = {"Range": f"bytes={local_file_size}-"}
            response = requests.get(url, stream=True, headers=headers, timeout=(10, 60))
            response.raise_for_status()

            # Vérifier si le serveur supporte la reprise
            if response.status_code == 206:
                print(f"Resuming download from byte {local_file_size}...")
            elif local_file_size > 0:
                print("Fichier partiel trouvé, mais le serveur ne supporte pas la reprise. Réinitialisation.")
                os.remove(dest)
                return download_file(url, dest, max_retries)

            # Obtenir la taille totale
            total_size = int(response.headers.get("content-length", 0)) + local_file_size

            # Télécharger avec une barre de progression
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading", initial=local_file_size) as progress_bar:
                with open(dest, "ab") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))

            print("Download completed.")
            return  # Succès, sortir de la boucle

        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            retries += 1
            print(f"Error during download: {e}")
            print(f"Retrying download... ({retries}/{max_retries})")

    # Si le maximum de retries est atteint
    print("Maximum retries reached. Aborting download.")
    raise Exception("Failed to download the file after multiple retries.")

def extract_file(tar_file, target_dir):
    print("Extracting files...")
    with tarfile.open(tar_file, 'r:gz') as tar:
        members = tar.getmembers()  # List files in the archive
        with tqdm(total=len(members), unit="file", desc="Extracting") as progress_bar:
            for member in members:
                tar.extract(member, target_dir)
                progress_bar.update(1)  # Update the progress bar
    print("Extraction completed.")

def move_files(source_dir):
    for root, _, files in os.walk(source_dir):
        for file in files:
            old_path = os.path.join(root, file)
            if old_path.startswith(source_dir + os.sep):
                new_path = old_path[len(source_dir + os.sep):]
            else:
                new_path = old_path

            target_path = os.path.join(".", new_path)
            target_dir = os.path.dirname(target_path)
            os.makedirs(target_dir, exist_ok=True)

            if os.path.exists(target_path):
                print(f"File already exists, removing: {target_path}")
                os.remove(target_path)

            shutil.move(old_path, target_path)
            print(f"{old_path} -> {target_path}")

    shutil.rmtree(source_dir)
    print(f"Removed directory: {source_dir}")

def main():
    # Download and extract the data files
    if not os.path.exists(ZIP_FILE):
        download_file(DATA_URL, ZIP_FILE)
    else:
        print("Zip file already exists. Skipping download.")
    
    # Create target directory if necessary
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    extract_file(ZIP_FILE, TARGET_DIR)

    # Move files to their proper locations
    move_files(TARGET_DIR)

    # Remove the ZIP file after extraction
    os.remove(ZIP_FILE)

    print("Data download and extraction completed.")

if __name__ == "__main__":
    main()
