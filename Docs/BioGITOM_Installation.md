## Installation

Follow the steps below to install and set up **BioGITOM**:

---

### 1. Clone the BioGITOM Repository

Clone the repository from GitHub to your local system:

```bash
git clone https://github.com/lias-laboratory/biogitom.git
```

---

### 2. Navigate to the Project Directory

Navigate to the directory where the repository was cloned:

```bash
cd biogitom
```

---

### 3. Set Up a Virtual Environment (Recommended)

Using a virtual environment ensures that dependencies do not conflict with other Python projects. Create and activate a virtual environment as follows:

- **Create the Environment:**

  ```bash
  python -m venv env
  ```

- **Activate the Environment:**

  - On **Linux/Mac**:
    ```bash
    source env/bin/activate
    ```
  - On **Windows**:
    ```bash
    env\Scripts\activate
    ```

---

### 4. Install Dependencies

Once the virtual environment is activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

---

### 5. Download the Required Data

Run the following script to download and extract the required data files:

```bash
python download_data.py
```

- **What this script does:**
  - Downloads a compressed archive of the required data files from the remote server.
  - Extracts the files into a temporary `temp/` directory.
  - Moves the files to the appropriate locations within the repository (e.g., `Datasets/`, `Tasks/`).
  - Deletes the temporary `temp/` directory after extraction.

---

By following these steps, **BioGITOM** will be ready for use! For guidance on executing tasks and running experiments, refer to the [Reproducibility Guide](./BioGITOM_Usage.md) section.
