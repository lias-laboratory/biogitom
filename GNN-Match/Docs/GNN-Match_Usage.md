# **GNN-Match Task Execution Guide**

This documentation outlines how to work with ontology matching tasks in **GNN-Match**.

---

## **Predefined Matching Tasks**

GNN-Match includes two ready-to-run ontology matching tasks:

| Task ID     | Description                                             |
| ----------- | ------------------------------------------------------- |
|           |
| `ncit2doid` | Maps the **NCIT** ontology to the **DOID** ontology     |
| `neoplas`   | Aligns **SNOMED (neoplasms)** with **NCIT (neoplasms)** |
|      |

Each task is organized under the `Tasks/<task_name>/` directory and includes all necessary components:

* Implementation scripts
* Embeddings
* Reference mappings

**Execution:** To execute a specific task, run the following command in your terminal:

```bash
python run_gnn-match.py --task <task>
```

Example: To run the `ncit2doid_GCN` task:

```bash
python run_gnn-match.py --task ncit2doid_GCN
```

---

