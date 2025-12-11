## Installation

Follow the steps below to install and set up **GNN-Match**:

---

### 1. Clone the GNN-Match Repository

Clone the repository from GitHub to your local system:

```bash
git clone https://github.com/lias-laboratory/biogitom.git
```

---

### 2. Navigate to the Project Directory

Navigate to the directory where the repository was cloned:

```bash
cd biogitom\GNN-Match
```

---

### 3. Set Up a Virtual Environment (Recommended)

Using a virtual environment ensures that dependencies do not conflict with other Python projects.

* **Create the Environment:**

  ```bash
  python -m venv env
  ```

* **Activate the Environment:**

  * On **Linux/Mac**:

    ```bash
    source env/bin/activate
    ```
  * On **Windows**:

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

### 5. Download the Required Data (If Applicable)

GNN-Match includes automated data preparation, run the corresponding script:

```bash
python download_data.py
```

* **This script typically:**

  * Downloads the required datasets or preprocessed resources.
  * Extracts and organizes them into relevant directories (e.g., `Datasets/`, `Tasks/`).
  * Cleans up temporary folders after installation.


---

By following these steps, **GNN-Match** will be ready for use!
For guidance on executing tasks and running experiments, refer to the [Usage Guide](GNN-Match_Usage.md).

---