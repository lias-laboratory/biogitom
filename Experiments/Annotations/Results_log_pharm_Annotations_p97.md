# **Package Installation**


```python
# Reinstall a specific version of PyTorch (v2.6.0) and torchvision (v0.21.0)
# The "--force-reinstall" flag ensures that the packages are reinstalled even if the correct version is already present.
# This is useful to resolve environment issues or when dependencies need to be reset.
!pip install torch==2.6.0 torchvision==0.21.0 --force-reinstall
```

    Collecting torch==2.6.0
      Downloading torch-2.6.0-cp311-cp311-manylinux1_x86_64.whl.metadata (28 kB)
    Collecting torchvision==0.21.0
      Downloading torchvision-0.21.0-cp311-cp311-manylinux1_x86_64.whl.metadata (6.1 kB)
    Collecting filelock (from torch==2.6.0)
      Downloading filelock-3.18.0-py3-none-any.whl.metadata (2.9 kB)
    Collecting typing-extensions>=4.10.0 (from torch==2.6.0)
      Downloading typing_extensions-4.13.2-py3-none-any.whl.metadata (3.0 kB)
    Collecting networkx (from torch==2.6.0)
      Downloading networkx-3.4.2-py3-none-any.whl.metadata (6.3 kB)
    Collecting jinja2 (from torch==2.6.0)
      Downloading jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
    Collecting fsspec (from torch==2.6.0)
      Downloading fsspec-2025.5.1-py3-none-any.whl.metadata (11 kB)
    Collecting nvidia-cuda-nvrtc-cu12==12.4.127 (from torch==2.6.0)
      Downloading nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cuda-runtime-cu12==12.4.127 (from torch==2.6.0)
      Downloading nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cuda-cupti-cu12==12.4.127 (from torch==2.6.0)
      Downloading nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cudnn-cu12==9.1.0.70 (from torch==2.6.0)
      Downloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cublas-cu12==12.4.5.8 (from torch==2.6.0)
      Downloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cufft-cu12==11.2.1.3 (from torch==2.6.0)
      Downloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-curand-cu12==10.3.5.147 (from torch==2.6.0)
      Downloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cusolver-cu12==11.6.1.9 (from torch==2.6.0)
      Downloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cusparse-cu12==12.3.1.170 (from torch==2.6.0)
      Downloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cusparselt-cu12==0.6.2 (from torch==2.6.0)
      Downloading nvidia_cusparselt_cu12-0.6.2-py3-none-manylinux2014_x86_64.whl.metadata (6.8 kB)
    Collecting nvidia-nccl-cu12==2.21.5 (from torch==2.6.0)
      Downloading nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl.metadata (1.8 kB)
    Collecting nvidia-nvtx-cu12==12.4.127 (from torch==2.6.0)
      Downloading nvidia_nvtx_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.7 kB)
    Collecting nvidia-nvjitlink-cu12==12.4.127 (from torch==2.6.0)
      Downloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting triton==3.2.0 (from torch==2.6.0)
      Downloading triton-3.2.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.4 kB)
    Collecting sympy==1.13.1 (from torch==2.6.0)
      Downloading sympy-1.13.1-py3-none-any.whl.metadata (12 kB)
    Collecting numpy (from torchvision==0.21.0)
      Downloading numpy-2.2.6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m62.0/62.0 kB[0m [31m1.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting pillow!=8.3.*,>=5.3.0 (from torchvision==0.21.0)
      Downloading pillow-11.2.1-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (8.9 kB)
    Collecting mpmath<1.4,>=1.1.0 (from sympy==1.13.1->torch==2.6.0)
      Downloading mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
    Collecting MarkupSafe>=2.0 (from jinja2->torch==2.6.0)
      Downloading MarkupSafe-3.0.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)
    Downloading torch-2.6.0-cp311-cp311-manylinux1_x86_64.whl (766.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m766.7/766.7 MB[0m [31m445.9 kB/s[0m eta [36m0:00:00[0m
    [?25hDownloading torchvision-0.21.0-cp311-cp311-manylinux1_x86_64.whl (7.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.2/7.2 MB[0m [31m67.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl (363.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m363.4/363.4 MB[0m [31m2.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (13.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.8/13.8 MB[0m [31m117.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (24.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m24.6/24.6 MB[0m [31m81.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (883 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m883.7/883.7 kB[0m [31m41.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl (664.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m664.8/664.8 MB[0m [31m482.2 kB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl (211.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m211.5/211.5 MB[0m [31m5.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl (56.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.3/56.3 MB[0m [31m23.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl (127.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m127.9/127.9 MB[0m [31m8.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl (207.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m207.5/207.5 MB[0m [31m5.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cusparselt_cu12-0.6.2-py3-none-manylinux2014_x86_64.whl (150.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m150.1/150.1 MB[0m [31m7.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl (188.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m188.7/188.7 MB[0m [31m5.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.1/21.1 MB[0m [31m103.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_nvtx_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (99 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m99.1/99.1 kB[0m [31m7.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading sympy-1.13.1-py3-none-any.whl (6.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.2/6.2 MB[0m [31m120.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading triton-3.2.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (253.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m253.2/253.2 MB[0m [31m4.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pillow-11.2.1-cp311-cp311-manylinux_2_28_x86_64.whl (4.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.6/4.6 MB[0m [31m99.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading typing_extensions-4.13.2-py3-none-any.whl (45 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m45.8/45.8 kB[0m [31m3.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading filelock-3.18.0-py3-none-any.whl (16 kB)
    Downloading fsspec-2025.5.1-py3-none-any.whl (199 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m199.1/199.1 kB[0m [31m14.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading jinja2-3.1.6-py3-none-any.whl (134 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.9/134.9 kB[0m [31m9.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading networkx-3.4.2-py3-none-any.whl (1.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m70.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading numpy-2.2.6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m16.8/16.8 MB[0m [31m110.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading MarkupSafe-3.0.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (23 kB)
    Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m536.2/536.2 kB[0m [31m32.5 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: triton, nvidia-cusparselt-cu12, mpmath, typing-extensions, sympy, pillow, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, MarkupSafe, fsspec, filelock, nvidia-cusparse-cu12, nvidia-cudnn-cu12, jinja2, nvidia-cusolver-cu12, torch, torchvision
      Attempting uninstall: mpmath
        Found existing installation: mpmath 1.3.0
        Uninstalling mpmath-1.3.0:
          Successfully uninstalled mpmath-1.3.0
      Attempting uninstall: typing-extensions
        Found existing installation: typing_extensions 4.13.2
        Uninstalling typing_extensions-4.13.2:
          Successfully uninstalled typing_extensions-4.13.2
      Attempting uninstall: sympy
        Found existing installation: sympy 1.13.1
        Uninstalling sympy-1.13.1:
          Successfully uninstalled sympy-1.13.1
      Attempting uninstall: pillow
        Found existing installation: pillow 11.2.1
        Uninstalling pillow-11.2.1:
          Successfully uninstalled pillow-11.2.1
      Attempting uninstall: numpy
        Found existing installation: numpy 2.0.2
        Uninstalling numpy-2.0.2:
          Successfully uninstalled numpy-2.0.2
      Attempting uninstall: networkx
        Found existing installation: networkx 3.4.2
        Uninstalling networkx-3.4.2:
          Successfully uninstalled networkx-3.4.2
      Attempting uninstall: MarkupSafe
        Found existing installation: MarkupSafe 3.0.2
        Uninstalling MarkupSafe-3.0.2:
          Successfully uninstalled MarkupSafe-3.0.2
      Attempting uninstall: fsspec
        Found existing installation: fsspec 2025.5.0
        Uninstalling fsspec-2025.5.0:
          Successfully uninstalled fsspec-2025.5.0
      Attempting uninstall: filelock
        Found existing installation: filelock 3.18.0
        Uninstalling filelock-3.18.0:
          Successfully uninstalled filelock-3.18.0
      Attempting uninstall: jinja2
        Found existing installation: Jinja2 3.1.6
        Uninstalling Jinja2-3.1.6:
          Successfully uninstalled Jinja2-3.1.6
      Attempting uninstall: torch
        Found existing installation: torch 2.6.0+cpu
        Uninstalling torch-2.6.0+cpu:
          Successfully uninstalled torch-2.6.0+cpu
      Attempting uninstall: torchvision
        Found existing installation: torchvision 0.21.0+cpu
        Uninstalling torchvision-0.21.0+cpu:
          Successfully uninstalled torchvision-0.21.0+cpu
    Successfully installed MarkupSafe-3.0.2 filelock-3.18.0 fsspec-2025.5.1 jinja2-3.1.6 mpmath-1.3.0 networkx-3.4.2 numpy-2.2.6 nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-cusparselt-cu12-0.6.2 nvidia-nccl-cu12-2.21.5 nvidia-nvjitlink-cu12-12.4.127 nvidia-nvtx-cu12-12.4.127 pillow-11.2.1 sympy-1.13.1 torch-2.6.0 torchvision-0.21.0 triton-3.2.0 typing-extensions-4.13.2





```python
# === Base Libraries ===
!pip install numpy --upgrade
!pip install pandas
!pip install optuna

# === FAISS (for Approximate Nearest Neighbor Search) ===
!pip install faiss-cpu        # CPU version (recommended unless using GPU)
# !pip install faiss-gpu      # Uncomment if running on CUDA-enabled GPU

# === PyTorch Geometric and dependencies ===
!pip install torch-geometric==2.4.0
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
# Optional: latest dev version from GitHub
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

# === DeepOnto (Ontology Matching Toolkit) ===
!pip install deeponto
# Optionally install custom version from a GitHub repository
# !pip install git+https://github.com/<username>/deeponto.git

```

    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (2.2.6)
    Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (2.2.2)
    Requirement already satisfied: numpy>=1.23.2 in /usr/local/lib/python3.11/dist-packages (from pandas) (2.2.6)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas) (2025.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
    Requirement already satisfied: optuna in /usr/local/lib/python3.11/dist-packages (4.3.0)
    Requirement already satisfied: alembic>=1.5.0 in /usr/local/lib/python3.11/dist-packages (from optuna) (1.16.1)
    Requirement already satisfied: colorlog in /usr/local/lib/python3.11/dist-packages (from optuna) (6.9.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from optuna) (2.2.6)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from optuna) (25.0)
    Requirement already satisfied: sqlalchemy>=1.4.2 in /usr/local/lib/python3.11/dist-packages (from optuna) (2.0.41)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from optuna) (4.67.1)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.11/dist-packages (from optuna) (6.0.2)
    Requirement already satisfied: Mako in /usr/lib/python3/dist-packages (from alembic>=1.5.0->optuna) (1.1.3)
    Requirement already satisfied: typing-extensions>=4.12 in /usr/local/lib/python3.11/dist-packages (from alembic>=1.5.0->optuna) (4.13.2)
    Requirement already satisfied: greenlet>=1 in /usr/local/lib/python3.11/dist-packages (from sqlalchemy>=1.4.2->optuna) (3.2.2)
    Requirement already satisfied: faiss-cpu in /usr/local/lib/python3.11/dist-packages (1.11.0)
    Requirement already satisfied: numpy<3.0,>=1.25.0 in /usr/local/lib/python3.11/dist-packages (from faiss-cpu) (2.2.6)
    Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from faiss-cpu) (25.0)
    Requirement already satisfied: torch-geometric==2.4.0 in /usr/local/lib/python3.11/dist-packages (2.4.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from torch-geometric==2.4.0) (4.67.1)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from torch-geometric==2.4.0) (2.2.6)
    Requirement already satisfied: scipy in /usr/local/lib/python3.11/dist-packages (from torch-geometric==2.4.0) (1.15.3)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch-geometric==2.4.0) (3.1.6)
    Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from torch-geometric==2.4.0) (2.32.3)
    Requirement already satisfied: pyparsing in /usr/local/lib/python3.11/dist-packages (from torch-geometric==2.4.0) (3.2.3)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.11/dist-packages (from torch-geometric==2.4.0) (1.6.1)
    Requirement already satisfied: psutil>=5.8.0 in /usr/local/lib/python3.11/dist-packages (from torch-geometric==2.4.0) (5.9.5)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch-geometric==2.4.0) (3.0.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->torch-geometric==2.4.0) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->torch-geometric==2.4.0) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->torch-geometric==2.4.0) (2.4.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->torch-geometric==2.4.0) (2025.4.26)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn->torch-geometric==2.4.0) (1.5.0)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn->torch-geometric==2.4.0) (3.6.0)
    Looking in links: https://data.pyg.org/whl/torch-2.0.0+cpu.html
    Collecting torch-scatter
      Using cached https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_scatter-2.1.2%2Bpt20cpu-cp311-cp311-linux_x86_64.whl (494 kB)
    Collecting torch-sparse
      Using cached https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_sparse-0.6.18%2Bpt20cpu-cp311-cp311-linux_x86_64.whl (1.2 MB)
    Collecting torch-cluster
      Using cached https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_cluster-1.6.3%2Bpt20cpu-cp311-cp311-linux_x86_64.whl (750 kB)
    Collecting torch-spline-conv
      Using cached https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_spline_conv-1.2.2%2Bpt20cpu-cp311-cp311-linux_x86_64.whl (208 kB)
    Requirement already satisfied: scipy in /usr/local/lib/python3.11/dist-packages (from torch-sparse) (1.15.3)
    Requirement already satisfied: numpy<2.5,>=1.23.5 in /usr/local/lib/python3.11/dist-packages (from scipy->torch-sparse) (2.2.6)
    Installing collected packages: torch-spline-conv, torch-scatter, torch-sparse, torch-cluster
    Successfully installed torch-cluster-1.6.3+pt20cpu torch-scatter-2.1.2+pt20cpu torch-sparse-0.6.18+pt20cpu torch-spline-conv-1.2.2+pt20cpu
      Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m194.8/194.8 kB[0m [31m4.3 MB/s[0m eta [36m0:00:00[0m
    [?25h  Building wheel for torch-geometric (pyproject.toml) ... [?25l[?25hdone
    Collecting deeponto
      Downloading deeponto-0.9.3-py3-none-any.whl.metadata (16 kB)
    Collecting JPype1 (from deeponto)
      Downloading jpype1-1.5.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)
    Collecting yacs (from deeponto)
      Downloading yacs-0.1.8-py3-none-any.whl.metadata (639 bytes)
    Requirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages (from deeponto) (2.6.0)
    Collecting anytree (from deeponto)
      Downloading anytree-2.13.0-py3-none-any.whl.metadata (8.0 kB)
    Requirement already satisfied: click in /usr/local/lib/python3.11/dist-packages (from deeponto) (8.2.1)
    Collecting dill (from deeponto)
      Downloading dill-0.4.0-py3-none-any.whl.metadata (10 kB)
    Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (from deeponto) (2.2.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from deeponto) (2.2.6)
    Requirement already satisfied: scikit_learn in /usr/local/lib/python3.11/dist-packages (from deeponto) (1.6.1)
    Requirement already satisfied: transformers[torch] in /usr/local/lib/python3.11/dist-packages (from deeponto) (4.52.2)
    Collecting datasets (from deeponto)
      Downloading datasets-3.6.0-py3-none-any.whl.metadata (19 kB)
    Requirement already satisfied: spacy in /usr/local/lib/python3.11/dist-packages (from deeponto) (3.8.6)
    Collecting pprintpp (from deeponto)
      Downloading pprintpp-0.4.0-py2.py3-none-any.whl.metadata (7.9 kB)
    Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from deeponto) (3.4.2)
    Collecting lxml (from deeponto)
      Downloading lxml-5.4.0-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (3.5 kB)
    Collecting textdistance (from deeponto)
      Downloading textdistance-4.6.3-py3-none-any.whl.metadata (18 kB)
    Requirement already satisfied: ipywidgets in /usr/local/lib/python3.11/dist-packages (from deeponto) (7.7.1)
    Requirement already satisfied: ipykernel in /usr/local/lib/python3.11/dist-packages (from deeponto) (6.17.1)
    Collecting enlighten (from deeponto)
      Downloading enlighten-1.14.1-py2.py3-none-any.whl.metadata (18 kB)
    Collecting rdflib (from deeponto)
      Downloading rdflib-7.1.4-py3-none-any.whl.metadata (11 kB)
    Requirement already satisfied: nltk in /usr/local/lib/python3.11/dist-packages (from deeponto) (3.9.1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from datasets->deeponto) (3.18.0)
    Requirement already satisfied: pyarrow>=15.0.0 in /usr/local/lib/python3.11/dist-packages (from datasets->deeponto) (20.0.0)
    Collecting dill (from deeponto)
      Downloading dill-0.3.8-py3-none-any.whl.metadata (10 kB)
    Requirement already satisfied: requests>=2.32.2 in /usr/local/lib/python3.11/dist-packages (from datasets->deeponto) (2.32.3)
    Requirement already satisfied: tqdm>=4.66.3 in /usr/local/lib/python3.11/dist-packages (from datasets->deeponto) (4.67.1)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.11/dist-packages (from datasets->deeponto) (3.5.0)
    Collecting multiprocess<0.70.17 (from datasets->deeponto)
      Downloading multiprocess-0.70.16-py311-none-any.whl.metadata (7.2 kB)
    Collecting fsspec<=2025.3.0,>=2023.1.0 (from fsspec[http]<=2025.3.0,>=2023.1.0->datasets->deeponto)
      Downloading fsspec-2025.3.0-py3-none-any.whl.metadata (11 kB)
    Requirement already satisfied: huggingface-hub>=0.24.0 in /usr/local/lib/python3.11/dist-packages (from datasets->deeponto) (0.31.4)
    Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from datasets->deeponto) (25.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.11/dist-packages (from datasets->deeponto) (6.0.2)
    Collecting blessed>=1.17.7 (from enlighten->deeponto)
      Downloading blessed-1.21.0-py2.py3-none-any.whl.metadata (13 kB)
    Collecting prefixed>=0.3.2 (from enlighten->deeponto)
      Downloading prefixed-0.9.0-py2.py3-none-any.whl.metadata (11 kB)
    Requirement already satisfied: debugpy>=1.0 in /usr/local/lib/python3.11/dist-packages (from ipykernel->deeponto) (1.8.0)
    Requirement already satisfied: ipython>=7.23.1 in /usr/local/lib/python3.11/dist-packages (from ipykernel->deeponto) (7.34.0)
    Requirement already satisfied: jupyter-client>=6.1.12 in /usr/local/lib/python3.11/dist-packages (from ipykernel->deeponto) (6.1.12)
    Requirement already satisfied: matplotlib-inline>=0.1 in /usr/local/lib/python3.11/dist-packages (from ipykernel->deeponto) (0.1.7)
    Requirement already satisfied: nest-asyncio in /usr/local/lib/python3.11/dist-packages (from ipykernel->deeponto) (1.6.0)
    Requirement already satisfied: psutil in /usr/local/lib/python3.11/dist-packages (from ipykernel->deeponto) (5.9.5)
    Requirement already satisfied: pyzmq>=17 in /usr/local/lib/python3.11/dist-packages (from ipykernel->deeponto) (24.0.1)
    Requirement already satisfied: tornado>=6.1 in /usr/local/lib/python3.11/dist-packages (from ipykernel->deeponto) (6.4.2)
    Requirement already satisfied: traitlets>=5.1.0 in /usr/local/lib/python3.11/dist-packages (from ipykernel->deeponto) (5.7.1)
    Requirement already satisfied: ipython-genutils~=0.2.0 in /usr/local/lib/python3.11/dist-packages (from ipywidgets->deeponto) (0.2.0)
    Requirement already satisfied: widgetsnbextension~=3.6.0 in /usr/local/lib/python3.11/dist-packages (from ipywidgets->deeponto) (3.6.10)
    Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from ipywidgets->deeponto) (3.0.15)
    Requirement already satisfied: joblib in /usr/local/lib/python3.11/dist-packages (from nltk->deeponto) (1.5.0)
    Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.11/dist-packages (from nltk->deeponto) (2024.11.6)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas->deeponto) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas->deeponto) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas->deeponto) (2025.2)
    Requirement already satisfied: pyparsing<4,>=2.1.0 in /usr/local/lib/python3.11/dist-packages (from rdflib->deeponto) (3.2.3)
    Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.11/dist-packages (from scikit_learn->deeponto) (1.15.3)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from scikit_learn->deeponto) (3.6.0)
    Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.11 in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (3.0.12)
    Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (1.0.5)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (1.0.12)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (2.0.11)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (3.0.9)
    Requirement already satisfied: thinc<8.4.0,>=8.3.4 in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (8.3.6)
    Requirement already satisfied: wasabi<1.2.0,>=0.9.1 in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (1.1.3)
    Requirement already satisfied: srsly<3.0.0,>=2.4.3 in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (2.5.1)
    Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (2.0.10)
    Requirement already satisfied: weasel<0.5.0,>=0.1.0 in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (0.4.1)
    Requirement already satisfied: typer<1.0.0,>=0.3.0 in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (0.15.3)
    Requirement already satisfied: pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4 in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (2.11.4)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (3.1.6)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (75.2.0)
    Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in /usr/local/lib/python3.11/dist-packages (from spacy->deeponto) (3.5.0)
    Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (4.13.2)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (12.4.127)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (12.4.127)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (12.4.127)
    Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (9.1.0.70)
    Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (12.4.5.8)
    Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (11.2.1.3)
    Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (10.3.5.147)
    Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (11.6.1.9)
    Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (12.3.1.170)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (0.6.2)
    Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (2.21.5)
    Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (12.4.127)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (12.4.127)
    Requirement already satisfied: triton==3.2.0 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (3.2.0)
    Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch->deeponto) (1.13.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch->deeponto) (1.3.0)
    Requirement already satisfied: tokenizers<0.22,>=0.21 in /usr/local/lib/python3.11/dist-packages (from transformers[torch]->deeponto) (0.21.1)
    Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.11/dist-packages (from transformers[torch]->deeponto) (0.5.3)
    Requirement already satisfied: accelerate>=0.26.0 in /usr/local/lib/python3.11/dist-packages (from transformers[torch]->deeponto) (1.7.0)
    Requirement already satisfied: wcwidth>=0.1.4 in /usr/local/lib/python3.11/dist-packages (from blessed>=1.17.7->enlighten->deeponto) (0.2.13)
    Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /usr/local/lib/python3.11/dist-packages (from fsspec[http]<=2025.3.0,>=2023.1.0->datasets->deeponto) (3.11.15)
    Collecting jedi>=0.16 (from ipython>=7.23.1->ipykernel->deeponto)
      Downloading jedi-0.19.2-py2.py3-none-any.whl.metadata (22 kB)
    Requirement already satisfied: decorator in /usr/local/lib/python3.11/dist-packages (from ipython>=7.23.1->ipykernel->deeponto) (5.2.1)
    Requirement already satisfied: pickleshare in /usr/local/lib/python3.11/dist-packages (from ipython>=7.23.1->ipykernel->deeponto) (0.7.5)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /usr/local/lib/python3.11/dist-packages (from ipython>=7.23.1->ipykernel->deeponto) (3.0.51)
    Requirement already satisfied: pygments in /usr/local/lib/python3.11/dist-packages (from ipython>=7.23.1->ipykernel->deeponto) (2.19.1)
    Requirement already satisfied: backcall in /usr/local/lib/python3.11/dist-packages (from ipython>=7.23.1->ipykernel->deeponto) (0.2.0)
    Requirement already satisfied: pexpect>4.3 in /usr/local/lib/python3.11/dist-packages (from ipython>=7.23.1->ipykernel->deeponto) (4.9.0)
    Requirement already satisfied: jupyter-core>=4.6.0 in /usr/local/lib/python3.11/dist-packages (from jupyter-client>=6.1.12->ipykernel->deeponto) (5.7.2)
    Requirement already satisfied: language-data>=1.2 in /usr/local/lib/python3.11/dist-packages (from langcodes<4.0.0,>=3.2.0->spacy->deeponto) (1.3.0)
    Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.11/dist-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy->deeponto) (0.7.0)
    Requirement already satisfied: pydantic-core==2.33.2 in /usr/local/lib/python3.11/dist-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy->deeponto) (2.33.2)
    Requirement already satisfied: typing-inspection>=0.4.0 in /usr/local/lib/python3.11/dist-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy->deeponto) (0.4.1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas->deeponto) (1.17.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests>=2.32.2->datasets->deeponto) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests>=2.32.2->datasets->deeponto) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests>=2.32.2->datasets->deeponto) (2.4.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests>=2.32.2->datasets->deeponto) (2025.4.26)
    Requirement already satisfied: blis<1.4.0,>=1.3.0 in /usr/local/lib/python3.11/dist-packages (from thinc<8.4.0,>=8.3.4->spacy->deeponto) (1.3.0)
    Requirement already satisfied: confection<1.0.0,>=0.0.1 in /usr/local/lib/python3.11/dist-packages (from thinc<8.4.0,>=8.3.4->spacy->deeponto) (0.1.5)
    Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.11/dist-packages (from typer<1.0.0,>=0.3.0->spacy->deeponto) (1.5.4)
    Requirement already satisfied: rich>=10.11.0 in /usr/local/lib/python3.11/dist-packages (from typer<1.0.0,>=0.3.0->spacy->deeponto) (14.0.0)
    Requirement already satisfied: cloudpathlib<1.0.0,>=0.7.0 in /usr/local/lib/python3.11/dist-packages (from weasel<0.5.0,>=0.1.0->spacy->deeponto) (0.21.1)
    Requirement already satisfied: smart-open<8.0.0,>=5.2.1 in /usr/local/lib/python3.11/dist-packages (from weasel<0.5.0,>=0.1.0->spacy->deeponto) (7.1.0)
    Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.11/dist-packages (from widgetsnbextension~=3.6.0->ipywidgets->deeponto) (6.5.7)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->spacy->deeponto) (3.0.2)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets->deeponto) (2.6.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets->deeponto) (1.3.2)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets->deeponto) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets->deeponto) (1.6.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets->deeponto) (6.4.4)
    Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets->deeponto) (0.3.1)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets->deeponto) (1.20.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.4 in /usr/local/lib/python3.11/dist-packages (from jedi>=0.16->ipython>=7.23.1->ipykernel->deeponto) (0.8.4)
    Requirement already satisfied: platformdirs>=2.5 in /usr/local/lib/python3.11/dist-packages (from jupyter-core>=4.6.0->jupyter-client>=6.1.12->ipykernel->deeponto) (4.3.8)
    Requirement already satisfied: marisa-trie>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from language-data>=1.2->langcodes<4.0.0,>=3.2.0->spacy->deeponto) (1.2.1)
    Requirement already satisfied: argon2-cffi in /usr/local/lib/python3.11/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (23.1.0)
    Requirement already satisfied: nbformat in /usr/local/lib/python3.11/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (5.10.4)
    Requirement already satisfied: nbconvert>=5 in /usr/local/lib/python3.11/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (7.16.6)
    Requirement already satisfied: Send2Trash>=1.8.0 in /usr/local/lib/python3.11/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.8.3)
    Requirement already satisfied: terminado>=0.8.3 in /usr/local/lib/python3.11/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.18.1)
    Requirement already satisfied: prometheus-client in /usr/local/lib/python3.11/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.22.0)
    Requirement already satisfied: nbclassic>=0.4.7 in /usr/local/lib/python3.11/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.3.1)
    Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.11/dist-packages (from pexpect>4.3->ipython>=7.23.1->ipykernel->deeponto) (0.7.0)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.11/dist-packages (from rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy->deeponto) (3.0.0)
    Requirement already satisfied: wrapt in /usr/local/lib/python3.11/dist-packages (from smart-open<8.0.0,>=5.2.1->weasel<0.5.0,>=0.1.0->spacy->deeponto) (1.17.2)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.11/dist-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy->deeponto) (0.1.2)
    Requirement already satisfied: notebook-shim>=0.2.3 in /usr/local/lib/python3.11/dist-packages (from nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.2.4)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.11/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (4.13.4)
    Requirement already satisfied: bleach!=5.0.0 in /usr/local/lib/python3.11/dist-packages (from bleach[css]!=5.0.0->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (6.2.0)
    Requirement already satisfied: defusedxml in /usr/local/lib/python3.11/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.7.1)
    Requirement already satisfied: jupyterlab-pygments in /usr/local/lib/python3.11/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.3.0)
    Requirement already satisfied: mistune<4,>=2.0.3 in /usr/local/lib/python3.11/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (3.1.3)
    Requirement already satisfied: nbclient>=0.5.0 in /usr/local/lib/python3.11/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.10.2)
    Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.11/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.5.1)
    Requirement already satisfied: fastjsonschema>=2.15 in /usr/local/lib/python3.11/dist-packages (from nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (2.21.1)
    Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.11/dist-packages (from nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (4.23.0)
    Requirement already satisfied: argon2-cffi-bindings in /usr/local/lib/python3.11/dist-packages (from argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (21.2.0)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.11/dist-packages (from bleach!=5.0.0->bleach[css]!=5.0.0->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.5.1)
    Requirement already satisfied: tinycss2<1.5,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from bleach[css]!=5.0.0->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.4.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (2025.4.1)
    Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.36.2)
    Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.25.1)
    Requirement already satisfied: jupyter-server<3,>=1.8 in /usr/local/lib/python3.11/dist-packages (from notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.16.0)
    Requirement already satisfied: cffi>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.17.1)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.11/dist-packages (from beautifulsoup4->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (2.7)
    Requirement already satisfied: pycparser in /usr/local/lib/python3.11/dist-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (2.22)
    Requirement already satisfied: anyio>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (4.9.0)
    Requirement already satisfied: websocket-client in /usr/local/lib/python3.11/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.8.0)
    Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.11/dist-packages (from anyio>=3.1.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.3.1)
    Downloading deeponto-0.9.3-py3-none-any.whl (89.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m89.7/89.7 MB[0m [31m13.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading anytree-2.13.0-py3-none-any.whl (45 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m45.1/45.1 kB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading datasets-3.6.0-py3-none-any.whl (491 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m491.5/491.5 kB[0m [31m8.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading dill-0.3.8-py3-none-any.whl (116 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m116.3/116.3 kB[0m [31m7.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading enlighten-1.14.1-py2.py3-none-any.whl (42 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m42.3/42.3 kB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading jpype1-1.5.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (494 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m494.1/494.1 kB[0m [31m31.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading lxml-5.4.0-cp311-cp311-manylinux_2_28_x86_64.whl (4.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.9/4.9 MB[0m [31m116.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pprintpp-0.4.0-py2.py3-none-any.whl (16 kB)
    Downloading rdflib-7.1.4-py3-none-any.whl (565 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m565.1/565.1 kB[0m [31m32.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading textdistance-4.6.3-py3-none-any.whl (31 kB)
    Downloading yacs-0.1.8-py3-none-any.whl (14 kB)
    Downloading blessed-1.21.0-py2.py3-none-any.whl (84 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m84.7/84.7 kB[0m [31m6.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading fsspec-2025.3.0-py3-none-any.whl (193 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m193.6/193.6 kB[0m [31m13.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading multiprocess-0.70.16-py311-none-any.whl (143 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m143.5/143.5 kB[0m [31m10.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading prefixed-0.9.0-py2.py3-none-any.whl (13 kB)
    Downloading jedi-0.19.2-py2.py3-none-any.whl (1.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m63.2 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: prefixed, pprintpp, yacs, textdistance, rdflib, lxml, JPype1, jedi, fsspec, dill, blessed, anytree, multiprocess, enlighten, datasets, deeponto
      Attempting uninstall: fsspec
        Found existing installation: fsspec 2025.5.1
        Uninstalling fsspec-2025.5.1:
          Successfully uninstalled fsspec-2025.5.1
    Successfully installed JPype1-1.5.2 anytree-2.13.0 blessed-1.21.0 datasets-3.6.0 deeponto-0.9.3 dill-0.3.8 enlighten-1.14.1 fsspec-2025.3.0 jedi-0.19.2 lxml-5.4.0 multiprocess-0.70.16 pprintpp-0.4.0 prefixed-0.9.0 rdflib-7.1.4 textdistance-4.6.3 yacs-0.1.8



```python
# Import pandas for working with tabular data (e.g., CSV, TSV files)
import pandas as pd

# Import numpy for numerical operations and efficient array handling
import numpy as np

# Import json for reading and writing JSON-formatted files (useful for config or ontology structures)
import json

# Import pickle for serializing and deserializing Python objects (e.g., saving models or processed data)
import pickle

# Import warnings to control or suppress warning messages during runtime
import warnings

# Import gc (garbage collector) for managing memory manually when dealing with large datasets
import gc

# Ignore all warning messages to keep the output clean
warnings.filterwarnings('ignore')
```


```python
# Import PyTorch core library for tensor operations and model definition
import torch

# Import commonly used PyTorch components
from torch import Tensor, optim  # Tensor type and optimization algorithms (e.g., SGD, Adam)

# Import PyTorch's neural network module (base class for defining models)
import torch.nn as nn

# Import PyTorch's functional API for operations like activations and loss functions
import torch.nn.functional as F

# Import DataLoader utilities for batching and loading datasets during training
from torch.utils.data import DataLoader, TensorDataset

# === PyTorch Geometric (PyG) modules for graph-based learning ===

# Basic graph data structure from PyG
from torch_geometric.data import Data

# PyG-specific DataLoader for batching graphs
from torch_geometric.loader import DataLoader as GeoDataLoader

# Import graph convolution layers and pooling functions from PyG
from torch_geometric.nn import (
    GCNConv,             # Graph Convolutional Network layer
    GINConv,             # Graph Isomorphism Network convolution
    global_mean_pool,    # Global mean pooling over node embeddings
    global_add_pool,     # Global sum pooling over node embeddings
    MessagePassing       # Base class for defining custom GNN layers
)

# Explicitly re-import MessagePassing (optional if already above)
from torch_geometric.nn.conv import MessagePassing

# Graph utility functions from PyG
from torch_geometric.utils import (
    to_undirected,       # Converts a directed graph to undirected
    softmax              # Softmax over edges (e.g., for attention)
)

# Initialization utilities for GNN layers
from torch_geometric.nn.inits import (
    reset,               # Reset parameters
    glorot,              # Glorot (Xavier) weight initialization
    zeros                # Zero initialization
)

# Typing utilities from PyG for adjacency and tensor specifications
from torch_geometric.typing import (
    Adj, OptTensor, PairTensor, SparseTensor
)

# Dense linear transformation layer from PyG (alternative to torch.nn.Linear)
from torch_geometric.nn.dense.linear import Linear

# Additional PyTorch neural network components
from torch.nn import (
    Linear,             # Fully connected (dense) layer
    PReLU,              # Parametric ReLU activation
    Sequential,         # Layer container for building sequential models
    BatchNorm1d,        # Batch normalization for 1D inputs
    Dropout             # Dropout regularization
)
```


```python
# Import matplotlib for creating visualizations (e.g., loss curves, evaluation metrics, embedding projections)
import matplotlib.pyplot as plt
```


```python
# Import function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import encoder to convert categorical labels into integer values (useful for classification tasks)
from sklearn.preprocessing import LabelEncoder

# Import evaluation metrics for classification and regression tasks
from sklearn.metrics import (
    f1_score,            # Harmonic mean of precision and recall; useful for imbalanced classification
    precision_score,     # Measures the proportion of true positives among all predicted positives
    accuracy_score,      # Measures overall correctness of predictions (classification)
    mean_squared_error,  # Measures average squared difference between predicted and actual values (regression)
    mean_absolute_error  # Measures average absolute difference between predicted and actual values (regression)
)
```


```python
# Import the Ontology class for loading and manipulating OWL ontologies
from deeponto.onto import Ontology

# Import all components related to OAEI (Ontology Alignment Evaluation Initiative) benchmarking
from deeponto.align.oaei import *

# Import data structures for representing mappings between ontology entities
from deeponto.align.mapping import EntityMapping, ReferenceMapping
# - EntityMapping: represents a predicted alignment (one or more mappings)
# - ReferenceMapping: represents the gold standard/reference alignments

# Import the evaluator to compute Precision, Recall, and F1-score for alignments
from deeponto.align.evaluation import AlignmentEvaluator

# Utility function to read TSV/CSV tables as mapping or data frames
from deeponto.utils import read_table
```

    Please enter the maximum memory located to JVM [8g]: 8g
    



```python
# Import Optuna, a hyperparameter optimization framework for automating model tuning using strategies like Bayesian optimization
import optuna
```


```python
# Import the math module for mathematical functions (e.g., sqrt, log, exp)
import math

# Import the time module for measuring execution time of code blocks or functions
import time

# Import typing annotations for function signatures and code clarity
from typing import Optional, Tuple, Union, Callable
# - Optional[T]: denotes a value that could be of type T or None
# - Tuple: fixed-size ordered collection of elements
# - Union: allows multiple possible types (e.g., Union[int, str])
# - Callable: represents a function or method type
```


```python
# Import Python's built-in random module for generating pseudo-random numbers
import random

# Set the seed for PyTorch's random number generator to ensure reproducibility
import torch
torch.manual_seed(42)

# Set the seed for NumPy's random number generator to ensure reproducibility
import numpy as np
np.random.seed(42)

# Set the seed for Python's built-in random module to ensure reproducibility
random.seed(42)
```

# **Paths Definition**


```python
# Importing the 'drive' module from Google Colab to interact with Google Drive
from google.colab import drive

# Mount the user's Google Drive to the Colab environment
# After running this, a link will appear to authorize access, and Google Drive will be mounted at '/content/gdrive'
drive.mount('/content/gdrive')

```

    Mounted at /content/gdrive



```python
# Define the source ontology name
src_ent = "snomed.pharm"

# Define the target ontology name
tgt_ent = "ncit.pharm"

# Define the task name for this ontology matching process
task = "pharm"
```


```python
dir = "/content/gdrive/My Drive/BioGITOM-VLDB/"

# Define the directory for the dataset containing source and target ontologies
dataset_dir = f"{dir}/Datasets/{task}"

# Define the data directory for storing embeddings, adjacency matrices, and related files
data_dir = f"{dir}/{task}/Data"

# Define the directory for storing the results
results_dir = f"{dir}/{task}/Results"
```


```python
# Load the Source ontology using the Ontology class from DeepOnto
# This initializes the source ontology by loading its .owl file.
src_onto = Ontology(f"{dataset_dir}/{src_ent}.owl")

# Load the Target ontology using the Ontology class from DeepOnto
# This initializes the target ontology by loading its .owl file.
tgt_onto = Ontology(f"{dataset_dir}/{tgt_ent}.owl")

# Load the Source ontology using the Ontology class from DeepOnto
# This initializes the source ontology by loading its .owl file.
src_Emb = f"{data_dir}/enriched_snomed.pharm_BERT_Hybrid_emb_Hamiti.csv"

# Define the file path for the Target embeddings CSV file
# Embeddings for the target ontology entities are stored in this file.
tgt_Emb = f"{data_dir}/enriched_ncit.pharm_Hamiti_with_p97_emb_SentenceSapBERT.csv"

# Define the file path for the Source adjacency matrix
# This file represents the relationships (edges) between entities in the source ontology.
src_Adjacence = f"{data_dir}/{src_ent}_adjacence_Hamiti.csv"

# Define the file path for the Target adjacency matrix
# This file represents the relationships (edges) between entities in the target ontology.
tgt_Adjacence = f"{data_dir}/{tgt_ent}_adjacence_Hamiti.csv"

# Define the file path for the JSON file containing the Source ontology class labels
# This file maps the source ontology entities to their labels or names.
src_class = f"{data_dir}/{src_ent}_classes.json"

# Define the file path for the JSON file containing the Target ontology class labels
# This file maps the target ontology entities to their labels or names.
tgt_class = f"{data_dir}/{tgt_ent}_classes.json"

# Define the file path for the train data
train_file = f"{data_dir}/pharm__rdm_train_tgt_50_Hamiti_enriched.csv"

# Define the file path for the test data
# The test file contains reference mappings (ground truth) between the source and target ontologies.
test_file = f"{dataset_dir}/refs_equiv/test.tsv"

# Define the file path for the candidate mappings used during testing
# This file includes the candidate pairs (source and target entities) for ranking based metrics.
test_cands = f"{dataset_dir}/refs_equiv/test.cands.tsv"

# Reformatted candidate file derived from test.cands.tsv
# It contains the same mappings (SrcEntity, TgtEntity, CandidateTgtEntities),
# but in a structure optimized for scoring (e.g., using FAISS or embedding-based similarity).
cands_path = f"{data_dir}/{task}_cands.csv"

# Define the path where the prediction results will be saved in TSV format
# This file will store the final predictions (mappings) between source and target entities.
prediction_path = f"{results_dir}/{task}_matching_results.tsv"

# Define the path where all prediction results will be saved in TSV format
# This file will store detailed prediction results, including all candidate scores.
all_predictions_path = f"{results_dir}/{task}_all_predictions.tsv"

# Define the path where formatted ranking predictions will be saved in TSV format
# This file will contain predictions formatted for evaluation using ranking-based metrics.
formatted_predictions_path = f"{results_dir}/{task}_formatted_predictions.tsv"
```

# **GIT Architecture**



```python
# RGIT class definition which inherits from PyTorch Geometric's MessagePassing class
class RGIT(MessagePassing):

    _alpha: OptTensor  # Define _alpha as an optional tensor for storing attention weights

    def __init__(
        self,
        nn: Callable,  # Neural network to be used in the final layer of the GNN
        in_channels: Union[int, Tuple[int, int]],  # Input dimension, can be a single or pair of integers
        out_channels: int,  # Output dimension of the GNN
        eps: float = 0.,  # GIN parameter: epsilon for GIN aggregation
        train_eps: bool = False,  # GIN parameter: whether epsilon should be learnable
        heads: int = 1,  # Transformer parameter: number of attention heads
        dropout: float = 0.,  # Dropout rate for attention weights
        edge_dim: Optional[int] = None,  # Dimension for edge attributes (optional)
        bias: bool = True,  # Whether to use bias in linear layers
        root_weight: bool = True,  # GIN parameter: whether to apply root weight in aggregation
        **kwargs,  # Additional arguments passed to the parent class
    ):
        # Set the aggregation type to 'add' and initialize the parent class with node_dim=0
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        # Initialize input/output dimensions, neural network, and GIN/transformer parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn  # Neural network used by the GNN
        self.initial_eps = eps  # Initial value of epsilon for GIN

        # Set epsilon to be learnable or fixed
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))  # Learnable epsilon
        else:
            self.register_buffer('eps', torch.empty(1))  # Non-learnable epsilon (fixed)

        # Initialize transformer-related parameters
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None  # Placeholder for attention weights

        # Handle case where in_channels is a single integer or a tuple
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # Define the linear layers for key, query, and value for the transformer mechanism
        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)

        # Define linear transformation for edge embeddings if provided
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        # Reset all parameters to their initial values
        self.reset_parameters()

    # Function to reset model parameters
    def reset_parameters(self):
        super().reset_parameters()  # Call parent class reset method
        self.lin_key.reset_parameters()  # Reset key linear layer
        self.lin_query.reset_parameters()  # Reset query linear layer
        self.lin_value.reset_parameters()  # Reset value linear layer
        if self.edge_dim:
            self.lin_edge.reset_parameters()  # Reset edge linear layer if used
        reset(self.nn)  # Reset the neural network provided
        self.eps.data.fill_(self.initial_eps)  # Initialize epsilon with the starting value

    # Forward function defining how the input data flows through the model
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        # Unpack number of heads and output channels
        H, C = self.heads, self.out_channels

        # If x is a tensor, treat it as a pair of tensors (source and target embeddings)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # Extract source node embeddings
        x_t = x[0]

        # Apply linear transformations and reshape query, key, and value for multi-head attention
        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # Propagate messages through the graph using the propagate function
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        # Retrieve attention weights and reset them
        alpha = self._alpha
        self._alpha = None  # Reset _alpha after use
        out = out.mean(dim=1)  # Take the mean over all attention heads

        # Apply GIN aggregation by adding epsilon-scaled original node embeddings
        out = out + (1 + self.eps) * x_t
        return self.nn(out)  # Pass through the neural network

    # Message passing function which calculates attention and combines messages
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # If edge attributes are used, apply linear transformation and add them to the key
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr

        # Calculate attention (alpha) using the dot product between query and key
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)  # Apply softmax to normalize attention
        self._alpha = alpha  # Store attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # Apply dropout

        # Calculate the output message by applying attention to the value
        out = value_j
        if edge_attr is not None:
            out = out + edge_attr  # Add edge embeddings to the output if present
        out = out * alpha.view(-1, self.heads, 1)  # Scale by attention weights
        return out

    # String representation function for debugging or printing
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
```


```python
# Define the RGIT_mod class, a multi-layer GNN that uses both RGIT and linear layers
class RGIT_mod(torch.nn.Module):
    """Multi-layer RGIT with optional linear layers"""

    # Initialize the model with hidden dimension, number of RGIT layers, and number of linear layers
    def __init__(self, dim_h, num_layers, num_linear_layers=1):
        super(RGIT_mod, self).__init__()
        self.num_layers = num_layers  # Number of RGIT layers
        self.num_linear_layers = num_linear_layers  # Number of linear layers
        self.linears = torch.nn.ModuleList()  # List to store linear layers
        self.rgit_layers = torch.nn.ModuleList()  # List to store RGIT layers

        # Create a list of Linear and PReLU layers (for encoding entity names)
        for _ in range(num_linear_layers):
            self.linears.append(Linear(dim_h, dim_h))  # Linear transformation layer
            self.linears.append(PReLU(num_parameters=dim_h))  # Parametric ReLU activation function

        # Create a list of RGIT layers
        for _ in range(num_layers):
            self.rgit_layers.append(RGIT(  # Each RGIT layer contains a small MLP with Linear and PReLU
                Sequential(Linear(dim_h, dim_h), PReLU(num_parameters=dim_h),
                           Linear(dim_h, dim_h), PReLU(num_parameters=dim_h)), dim_h, dim_h))

    # Forward pass through the model
    def forward(self, x, edge_index):
        # Apply the linear layers first to the input
        for layer in self.linears:
            x = layer(x)

        # Then apply the RGIT layers for message passing
        for layer in self.rgit_layers:
            x = layer(x, edge_index)

        return x  # Return the final node embeddings after all layers

```

# **Gated Network Architecture**


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedCombination(nn.Module):
    def __init__(self, input_dim):
        super(GatedCombination, self).__init__()
        self.gate_A_fc = nn.Linear(input_dim, input_dim)
        self.gate_B_fc = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(1, 1)

    def euclidean_distance(self, a, b):
        """
        Compute the Euclidean distance between two tensors.
        Args:
            a: Tensor of shape [batch, dim]
            b: Tensor of shape [batch, dim]
        Returns:
            Tensor of shape [batch] representing the L2 distance.
        """
        return torch.norm(a - b, p=2, dim=1)

    def forward(self, x1, x2, x3, x4, return_embeddings=False):
        gate_values1 = torch.sigmoid(self.gate_A_fc(x1))
        a = x1 * gate_values1 + x2 * (1 - gate_values1)

        gate_values2 = torch.sigmoid(self.gate_B_fc(x3))
        b = x3 * gate_values2 + x4 * (1 - gate_values2)

        if return_embeddings:
            return a, b

        # Utilisation de la distance Euclidienne
        distance = self.euclidean_distance(a, b)

        # Passage dans couche de classification
        out = torch.sigmoid(self.fc(distance.unsqueeze(1)))
        return out

```

# **Utility functions**


```python
def adjacency_matrix_to_undirected_edge_index(adjacency_matrix):
    """
    Converts an adjacency matrix into an undirected edge index for use in graph-based neural networks.

    Args:
        adjacency_matrix: A 2D list or array representing the adjacency matrix of a graph.

    Returns:
        edge_index_undirected: A PyTorch tensor representing the undirected edges.
    """
    # Convert each element in the adjacency matrix to an integer (from boolean or float)
    adjacency_matrix = [[int(element) for element in sublist] for sublist in adjacency_matrix]

    # Convert the adjacency matrix into a PyTorch LongTensor (used for indexing)
    edge_index = torch.tensor(adjacency_matrix, dtype=torch.long)

    # Transpose the edge_index tensor so that rows represent edges in the form [source, target]
    edge_index = edge_index.t().contiguous()

    # Convert the directed edge_index into an undirected edge_index, meaning both directions are added (i.e., (i, j) and (j, i))
    edge_index_undirected = to_undirected(edge_index)

    return edge_index_undirected  # Return the undirected edge index
```


```python
def build_indexed_dict(file_path):
    """
    Builds a dictionary with numeric indexes for each key from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        indexed_dict (dict): A new dictionary where each key from the JSON file is assigned a numeric index.
    """
    # Load the JSON file into a Python dictionary
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Create a new dictionary with numeric indexes as keys and the original JSON keys as values
    indexed_dict = {index: key for index, key in enumerate(data.keys())}

    return indexed_dict  # Return the newly created dictionary
```


```python
def select_rows_by_index(embedding_vector, index_vector):
    """
    Select rows from an embedding vector using an index vector.

    Args:
        embedding_vector (torch.Tensor): 2D tensor representing the embedding vector with shape [num_rows, embedding_size].
        index_vector (torch.Tensor): 1D tensor representing the index vector.

    Returns:
        torch.Tensor: New tensor with selected rows from the embedding vector.
    """
    # Use torch.index_select to select the desired rows
    new_tensor = torch.index_select(embedding_vector, 0, index_vector)

    return new_tensor
```


```python
def contrastive_loss(source_embeddings, target_embeddings, labels, margin=1.0):
    """
    Computes the contrastive loss, a type of loss function used to train models in tasks like matching or similarity learning.

    Args:
        source_embeddings (torch.Tensor): Embeddings of the source graphs, shape [batch_size, embedding_size].
        target_embeddings (torch.Tensor): Embeddings of the target graphs, shape [batch_size, embedding_size].
        labels (torch.Tensor): Binary labels indicating if the pairs are matched (1) or not (0), shape [batch_size].
        margin (float): Margin value for the contrastive loss. Defaults to 1.0.

    Returns:
        torch.Tensor: The contrastive loss value.
    """
    # Calculate the pairwise Euclidean distance between source and target embeddings
    distances = F.pairwise_distance(source_embeddings, target_embeddings)

    # Compute the contrastive loss:
    # - For matched pairs (label == 1), the loss is the squared distance between embeddings.
    # - For non-matched pairs (label == 0), the loss is based on how far apart the embeddings are,
    #   but penalizes them only if the distance is less than the margin.
    loss = torch.mean(
        labels * 0.4 * distances.pow(2) +  # For positive pairs, minimize the distance (squared)
        (1 - labels) * 0.4 * torch.max(torch.zeros_like(distances), margin - distances).pow(2)  # For negative pairs, maximize the distance (up to the margin)
    )

    return loss  # Return the computed contrastive loss

```


```python
def save_gated_embeddings(gated_model, embeddings_src, x_src, embeddings_tgt, x_tgt,
                          indexed_dict_src, indexed_dict_tgt,
                          output_file_src, output_file_tgt):
    """
    Compute and save the final entity embeddings generated by the GatedCombination model
    for both source and target ontologies. Outputs include entity URIs and their final vectors.
    Measures and prints the execution time of the entire operation.

    Args:
        gated_model (nn.Module): The trained GatedCombination model.
        embeddings_src (Tensor): Structural embeddings for the source ontology.
        x_src (Tensor): Semantic embeddings for the source ontology.
        embeddings_tgt (Tensor): Structural embeddings for the target ontology.
        x_tgt (Tensor): Semantic embeddings for the target ontology.
        indexed_dict_src (dict): Index-to-URI mapping for the source ontology.
        indexed_dict_tgt (dict): Index-to-URI mapping for the target ontology.
        output_file_src (str): Path to save source embeddings (TSV).
        output_file_tgt (str): Path to save target embeddings (TSV).
    """
    import pandas as pd
    import torch
    import time

    start_time = time.time()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gated_model = gated_model.to(device)
    gated_model.eval()

    # Move inputs to the same device
    embeddings_src = embeddings_src.to(device)
    x_src = x_src.to(device)
    embeddings_tgt = embeddings_tgt.to(device)
    x_tgt = x_tgt.to(device)

    with torch.no_grad():
        # === Source ontology ===
        gate_src = torch.sigmoid(gated_model.gate_A_fc(embeddings_src))
        final_src = embeddings_src * gate_src + x_src * (1 - gate_src)
        final_src = final_src.cpu().numpy()

        # === Target ontology ===
        gate_tgt = torch.sigmoid(gated_model.gate_B_fc(embeddings_tgt))
        final_tgt = embeddings_tgt * gate_tgt + x_tgt * (1 - gate_tgt)
        final_tgt = final_tgt.cpu().numpy()

    # Create DataFrames with Concept URI and embedding values
    df_src = pd.DataFrame(final_src)
    df_src.insert(0, "Concept", [indexed_dict_src[i] for i in range(len(df_src))])

    df_tgt = pd.DataFrame(final_tgt)
    df_tgt.insert(0, "Concept", [indexed_dict_tgt[i] for i in range(len(df_tgt))])

    # Save embeddings to file
    df_src.to_csv(output_file_src, sep='\t', index=False)
    df_tgt.to_csv(output_file_tgt, sep='\t', index=False)

    elapsed_time = time.time() - start_time
    print(f"✅ Gated embeddings saved:\n- Source: {output_file_src}\n- Target: {output_file_tgt}")
    print(f"⏱️ Execution time: {elapsed_time:.2f} seconds")

```


```python
import pandas as pd

def format_ranked_predictions_for_mrr(reference_file, predicted_file, output_file):
    """
    Format predicted scores into ranked candidate lists per source entity,
    in a structure compatible with MRR and Hits@k evaluation.

    Args:
        reference_file (str): Path to the reference test candidate file (e.g., 'test.cands.tsv'),
                              with columns: SrcEntity, TgtEntity (gold), CandidateTgtEntities (list)
        predicted_file (str): Path to the flat prediction file with columns: SrcEntity, TgtEntity, Score
        output_file (str): Path to save the formatted ranked candidates (for evaluation)

    Returns:
        str: Path to the formatted output file (TSV with columns: SrcEntity, TgtEntity, TgtCandidates)
    """

    # Load reference candidates (test.cands.tsv format)
    reference_data = pd.read_csv(reference_file, sep='\t').values.tolist()

    # Load predictions and ensure scores are floats
    predicted_data = pd.read_csv(predicted_file, sep="\t")
    predicted_data["Score"] = predicted_data["Score"].apply(
        lambda x: float(x.strip("[]")) if isinstance(x, str) else float(x)
    )

    # Build a dictionary for quick score lookup
    score_lookup = {
        (row["SrcEntity"], row["TgtEntity"]): row["Score"]
        for _, row in predicted_data.iterrows()
    }

    ranking_results = []

    # For each source entity, rank its candidate targets by predicted score
    for src_entity, tgt_gold, tgt_cands in reference_data:
        try:
            raw = eval(tgt_cands)
            candidates = list(raw) if isinstance(raw, (list, tuple)) else []
        except:
            candidates = []

        # Score each candidate (default to very low score if missing)
        scored_cands = [
            (cand, score_lookup.get((src_entity, cand), -1e9))
            for cand in candidates
        ]

        # Sort by score descending
        ranked = sorted(scored_cands, key=lambda x: x[1], reverse=True)

        # Append the ranking result
        ranking_results.append((src_entity, tgt_gold, ranked))

    # Save to TSV file (used later for MRR / Hits@k computation)
    pd.DataFrame(ranking_results, columns=["SrcEntity", "TgtEntity", "TgtCandidates"]).to_csv(
        output_file, sep="\t", index=False
    )

    print(f"✅ Ranked predictions saved for evaluation: {output_file}")
    return output_file

```


```python
import pandas as pd

def filter_ignored_class(src_emb_path, tgt_emb_path, src_onto, tgt_onto):
    """
    Filters the source and target embedding files by removing concepts considered "ignored classes"
    (e.g., owl:Thing, deprecated entities, etc.) based on both source and target ontologies.

    Args:
        src_emb_path (str): Path to the TSV file containing source embeddings with 'Concept' column.
        tgt_emb_path (str): Path to the TSV file containing target embeddings with 'Concept' column.
        src_onto (Ontology): Source ontology object loaded with DeepOnto.
        tgt_onto (Ontology): Target ontology object loaded with DeepOnto.

    Returns:
        (str, str): Paths to the cleaned source and target embedding files.
    """

    # === Load the embedding files ===
    df_src = pd.read_csv(src_emb_path, sep='\t', dtype=str)
    print(f"🔍 Initial source file: {len(df_src)} rows")

    df_tgt = pd.read_csv(tgt_emb_path, sep='\t', dtype=str)
    print(f"🔍 Initial target file: {len(df_tgt)} rows")

    # === Step 1: Retrieve ignored classes from both ontologies ===
    ignored_class_index = get_ignored_class_index(src_onto)  # e.g., owl:Thing, non-usable classes
    ignored_class_index.update(get_ignored_class_index(tgt_onto))  # Merge with target ontology's ignored classes
    ignored_uris = set(str(uri).strip() for uri in ignored_class_index)

    # === Step 2: Remove rows where the 'Concept' column matches ignored URIs ===
    df_src_cleaned = df_src[~df_src['Concept'].isin(ignored_uris)].reset_index(drop=True)
    df_tgt_cleaned = df_tgt[~df_tgt['Concept'].isin(ignored_uris)].reset_index(drop=True)

    print(f"✅ Source after removing ignored classes: {len(df_src_cleaned)} rows")
    print(f"✅ Target after removing ignored classes: {len(df_tgt_cleaned)} rows")

    # === Step 3: Save the cleaned embedding files ===
    output_file_src = src_emb_path.replace(".tsv", "_cleaned.tsv")
    output_file_tgt = tgt_emb_path.replace(".tsv", "_cleaned.tsv")

    df_src_cleaned.to_csv(output_file_src, sep='\t', index=False)
    df_tgt_cleaned.to_csv(output_file_tgt, sep='\t', index=False)

    print(f"📁 Cleaned source file saved to: {output_file_src}")
    print(f"📁 Cleaned target file saved to: {output_file_tgt}")

    return output_file_src, output_file_tgt
```

# **FAISS Similarity**


```python
import pandas as pd
import numpy as np
import faiss
import time

def load_embeddings(src_emb_path, tgt_emb_path):
    """
    Load the embeddings for the source and target ontologies from TSV files.

    Args:
        src_emb_path (str): Path to the source embeddings file.
        tgt_emb_path (str): Path to the target embeddings file.

    Returns:
        uris_src (np.ndarray): URIs of source entities.
        uris_tgt (np.ndarray): URIs of target entities.
        src_vecs (np.ndarray): Embedding vectors for source entities.
        tgt_vecs (np.ndarray): Embedding vectors for target entities.
    """
    df_src = pd.read_csv(src_emb_path, sep='\t')  # Read source embeddings
    df_tgt = pd.read_csv(tgt_emb_path, sep='\t')  # Read target embeddings
    uris_src = df_src["Concept"].values           # Extract source URIs
    uris_tgt = df_tgt["Concept"].values           # Extract target URIs
    src_vecs = df_src.drop(columns=["Concept"]).values.astype('float32')  # Extract and convert source vectors
    tgt_vecs = df_tgt.drop(columns=["Concept"]).values.astype('float32')  # Extract and convert target vectors
    return uris_src, uris_tgt, src_vecs, tgt_vecs

def save_results(uris_src, uris_tgt, indices, scores, output_file, top_k):
    """
    Save the top-k mapping results to a TSV file.

    Args:
        uris_src (np.ndarray): URIs of source entities.
        uris_tgt (np.ndarray): URIs of target entities.
        indices (np.ndarray): Indices of top-k matched target entities.
        scores (np.ndarray): Corresponding similarity scores.
        output_file (str): Output TSV file path.
        top_k (int): Number of top matches per source entity.
    """
    rows = []
    for i, (ind_row, score_row) in enumerate(zip(indices, scores)):
        src_uri = uris_src[i]
        for j, tgt_idx in enumerate(ind_row):
            tgt_uri = uris_tgt[tgt_idx]
            score = score_row[j]
            rows.append((src_uri, tgt_uri, score))  # Store each top-k match
    df_result = pd.DataFrame(rows, columns=["SrcEntity", "TgtEntity", "Score"])
    df_result.to_csv(output_file, sep='\t', index=False)  # Save to file
    print(f"Top-{top_k} FAISS similarity results saved to: {output_file}")

def topk_faiss_l2(src_emb_path, tgt_emb_path, top_k=15, output_file="topk_l2.tsv"):
    """
    Compute the top-k most similar target entities for each source entity using FAISS with L2 distance.

    Args:
        src_emb_path (str): Path to the source embeddings file.
        tgt_emb_path (str): Path to the target embeddings file.
        top_k (int): Number of top matches to retrieve.
        output_file (str): Path to save the top-k results.
    """
    print("🔹 Using L2 (Euclidean) distance with FAISS")
    start = time.time()  # Start timing

    # Load embeddings
    uris_src, uris_tgt, src_vecs, tgt_vecs = load_embeddings(src_emb_path, tgt_emb_path)

    # Build FAISS index using L2 distance
    dim = src_vecs.shape[1]
    index = faiss.IndexFlatL2(dim)  # Create FAISS index for L2 distance
    index.add(tgt_vecs)             # Add target vectors to index

    # Perform nearest neighbor search
    distances, indices = index.search(src_vecs, top_k)

    # Convert distances to similarity scores (optional: inverse of distance)
    similarity_scores = 1 / (1 + distances)

    # Save the results
    save_results(uris_src, uris_tgt, indices, similarity_scores, output_file, top_k)

    # Display execution time
    print(f"⏱️ Execution time: {time.time() - start:.2f} seconds")


```

# **Mappings Evaluation Functions**

# **Precision, Recall, F1**

### Evaluation Strategy and Filtering Justification

### Filtering Justification

To ensure that evaluation metrics such as Precision, Recall, and F1-score accurately reflect the model's true performance, we apply two carefully designed filtering steps in the evaluate_predictions function. These filters are specifically crafted to focus the evaluation on verifiable predictions without unnecessarily penalizing the model or distorting the top-k candidate structure.

#### 1. Filtering Out Training-Only Entities

We remove all predicted mappings that involve entities (either source or target) that are present exclusively in the training set and do not appear in the test set.

This is a crucial step because:

- In datasets like Bio-ML, entities often appear in both training and test sets but are aligned with different targets. Filtering based solely on mappings would eliminate valuable generalization examples.

- Predictions involving entities that are not part of the test set cannot be evaluated and could unfairly skew precision and recall.

Importantly, we do not remove all mappings seen during training. Instead, we only discard those that involve non-testable entities. This distinction ensures that we retain valuable mappings between shared entities that can still play a meaningful role during prediction and ranking.

#### 2. Filtering on `SrcEntity` present in the test set

The second step keeps only the predictions where the `SrcEntity` is included in the test reference set.

- This eliminates **non-evaluable false positives**, i.e., predicted mappings for source entities that do not appear in the test set and therefore have no ground-truth correspondences. Including such predictions **unfairly penalizes precision and F1-score**, even though they are technically not verifiable errors.

- It focuses the evaluation on entities with defined ground-truth mappings, which is critical for computing metrics such as :

$P_{\text{test}} = \frac{|\mathcal{M}_{\text{out}} \cap \mathcal{M}_{\text{test}}|}{|\mathcal{M}_{\text{out}} \setminus (\mathcal{M}_{\text{ref}} \setminus \mathcal{M}_{\text{test}})|}$.

---

### 📌 Why This Works

Let’s illustrate the rationale using a simplified **Bio-ML** scenario:

| Dataset | Mappings                                      | Entities       |
|---------|-----------------------------------------------|----------------|
| Train   | (A:Cancer, B:Melanoma), (C:Radiation, D:Therapy) | A, B, C, D     |
| Test    | (A:Cancer, E:Carcinoma), (F:Skin, B:Melanoma)    | A, B, E, F     |

After applying our filtering strategy:

- **Removed**: (C:Radiation, D:Therapy) → both `C` and `D` are exclusive to train
- **Kept**: (A:Cancer, B:Melanoma) → `A` and `B` also appear in test

This means we preserve mappings that involve entities **shared between train and test**, even if the specific mapping was seen during training and is not part of the test reference set.

---

### ✅ Key Advantages

- **Preserves semantic context**  
  Keeping *(A, B)* helps the model calibrate similarity scores for other test mappings involving `A` or `B`, such as *(A, E)* or *(F, B)*.

- **Maintains fair competition in ranking**  
  Removing all training mappings would delete useful distractors (e.g., *(A, B)*), which could **artificially promote** weaker candidates (e.g., *(A, E)*) to the top rank, simply due to lack of strong alternatives.

---

This strategy strikes a balance between **evaluation fairness** and **preservation of top-k integrity**, ensuring that the ranking dynamics remain realistic and reflective of the model’s true generalization ability.



```python
def select_best_candidates_per_src_with_margin(df, score_margin=0.01):
    """
    For each SrcEntity, retain all candidate mappings whose similarity score is
    within 99% of the best score (default margin = 0.01).

    Args:
        df (pd.DataFrame): DataFrame containing columns ['SrcEntity', 'TgtEntity', 'Score'].
        score_margin (float): Score margin. 0.01 means keep scores ≥ 99% of best score.

    Returns:
        pd.DataFrame: Filtered DataFrame with multiple high-quality candidates per SrcEntity.
    """
    selected_rows = []

    for src, group in df.groupby("SrcEntity"):
        group_sorted = group.sort_values(by="Score", ascending=False)
        best_score = group_sorted.iloc[0]["Score"]
        threshold = best_score * (1 - score_margin)

        # Keep all target entities with score >= threshold
        close_matches = group_sorted[group_sorted["Score"] >= threshold]
        selected_rows.append(close_matches)

    result_df = pd.concat(selected_rows).reset_index(drop=True)
    print(f"🏆 Selected candidates within {(1 - score_margin) * 100:.1f}% of best score per SrcEntity: {len(result_df)} rows")
    return result_df

```


```python
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def evaluate_predictions(
    pred_file, train_file, test_file,
    threshold=0.0, margin_ratio=0.997
):
    """
    Evaluate predicted mappings by applying filtering, thresholding, top-1 selection with margin,
    and computing precision, recall, and F1-score against the test set.
    """

    # Step 1: Load prediction, train, and test data
    df = pd.read_csv(pred_file, sep='\t')
    train_df = pd.read_csv(train_file, sep='\t')
    test_df = pd.read_csv(test_file, sep='\t')

    # ✅ Step 2: Remove entities that appear only in the training set
    train_uris = set(train_df['SrcEntity']) | set(train_df['TgtEntity'])
    test_uris = set(test_df['SrcEntity']) | set(test_df['TgtEntity'])
    uris_to_exclude = train_uris - test_uris
    df = df[~df['SrcEntity'].isin(uris_to_exclude) & ~df['TgtEntity'].isin(uris_to_exclude)]

    # Step 3: Keep only predictions where SrcEntity is part of the test set
    test_src_entities = set(test_df['SrcEntity'])
    df = df[df['SrcEntity'].isin(test_src_entities)]

    # Step 5: Save filtered predictions to file
    df.to_csv(all_predictions_path, sep='\t', index=False)

    # Step 6: Select best predictions per SrcEntity using a relaxed top-1 margin
    df_topk = select_best_candidates_per_src_with_margin(df, score_margin=0.003)

    # Step 7: Save the top-1 filtered predictions
    df_topk.to_csv(prediction_path, sep='\t', index=False)

    print(f"   ➤ Mappings file:   {prediction_path}")

    # === Step 8: Evaluate against reference mappings
    preds = EntityMapping.read_table_mappings(prediction_path)
    refs = ReferenceMapping.read_table_mappings(test_file)

    preds_set = {p.to_tuple() for p in preds}
    refs_set = {r.to_tuple() for r in refs}
    correct = len(preds_set & refs_set)

    results = AlignmentEvaluator.f1(preds, refs)

    # === Step 9: Print evaluation metrics
    print("\n🎯 Evaluation Summary:")
    print(f"   - Correct mappings:     {correct}")
    print(f"   - Total predictions:    {len(preds)}")
    print(f"   - Total references:     {len(refs)}")
    print(f"📊 Precision:              {results['P']:.3f}")
    print(f"📊 Recall:                 {results['R']:.3f}")
    print(f"📊 F1-score:               {results['F1']:.3f}\n")

    return prediction_path, results, correct
```

# **Precision@k, Recall@k, F1@k**


```python
import pandas as pd
from collections import defaultdict

def evaluate_topk(topk_file, train_file, test_file, k=1, threshold=0.0):
    """
    Evaluate Top-K predictions using Precision, Recall, and F1-score,
    after filtering out training-only URIs, keeping only test sources, and applying 1-1 constraint.

    Args:
        topk_file (str): Path to the top-k prediction file (TSV with SrcEntity, TgtEntity, Score)
        train_file (str): Path to the training mappings file (TSV)
        test_file (str): Path to the test mappings file (TSV)
        k (int): Value of K for top-k evaluation
        threshold (float): Minimum score to consider a prediction valid

    Returns:
        dict: Dictionary containing Precision@K, Recall@K, and F1@K
    """

    # === Step 1: Load input files ===
    df = pd.read_csv(topk_file, sep='\t', dtype=str)
    train_df = pd.read_csv(train_file, sep='\t', dtype=str)
    test_df = pd.read_csv(test_file, sep='\t', dtype=str)

    # === Step 2: Remove URIs only present in the training set ===
    train_uris = set(train_df['SrcEntity']) | set(train_df['TgtEntity'])
    test_uris = set(test_df['SrcEntity']) | set(test_df['TgtEntity'])
    uris_to_exclude = train_uris - test_uris
    df = df[~(df['SrcEntity'].isin(uris_to_exclude) | df['TgtEntity'].isin(uris_to_exclude))].reset_index(drop=True)

    # === Step 3: Keep only source entities from the test set ===
    src_entities_test = set(test_df['SrcEntity'])
    df = df[df['SrcEntity'].isin(src_entities_test)].reset_index(drop=True)

    # === Step 4: Convert score column to float and sort ===
    df['Score'] = df['Score'].apply(lambda x: float(x.strip("[]")) if isinstance(x, str) else float(x))
    df_sorted = df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    # === Step 5: Apply 1-to-1 constraint (greedy strategy with optional threshold)
    matched_sources = set()
    matched_targets = set()
    result = []

    for _, row in df_sorted.iterrows():
        src, tgt, score = row['SrcEntity'], row['TgtEntity'], row['Score']
        if src not in matched_sources and tgt not in matched_targets and score >= threshold:
            result.append((src, tgt, score))
            matched_sources.add(src)
            matched_targets.add(tgt)

    # === Step 6: Create and save Top-K prediction dataframe
    matching_results_df = pd.DataFrame(result, columns=['SrcEntity', 'TgtEntity', 'Score'])
    output_file = topk_file.replace(".tsv", "_predictions.tsv")
    matching_results_df.to_csv(output_file, sep='\t', index=False)

    # === Step 7: Build reference dictionary from test set
    ref_dict = defaultdict(set)
    for _, row in test_df.iterrows():
        ref_dict[row['SrcEntity']].add(row['TgtEntity'])

    # === Step 8: Select Top-K predictions for each source entity
    matching_results_df['Score'] = matching_results_df['Score'].astype(float)
    topk_df = matching_results_df.sort_values(by='Score', ascending=False).groupby('SrcEntity').head(k)

    # === Step 9: Compute Precision@K, Recall@K, F1@K
    total_tp = total_pred = total_ref = 0


    for src, group in topk_df.groupby('SrcEntity'):
        predicted = set(group['TgtEntity'])
        true = ref_dict.get(src, set())
        tp = len(predicted & true)
        total_tp += tp
        total_pred += len(predicted)
        total_ref += len(true)

    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_ref if total_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if precision + recall > 0 else 0.0

    # === Step 10: Print metrics

    print(f"📊 Precision@{k}:            {precision:.3f}")
    print(f"📊 Recall@{k}:               {recall:.3f}")
    print(f"📊 F1@{k}:                   {f1:.3f}\n")

    return {
        f'Precision@{k}': round(precision, 3),
        f'Recall@{k}': round(recall, 3),
        f'F1@{k}': round(f1, 3)
    }


```

# **Main Code**






# Reading semantic node embeddings provided by the ENE


```python
# Read the source embeddings from a CSV file into a pandas DataFrame
df_embbedings_src = pd.read_csv(src_Emb, index_col=0)

# Convert the DataFrame to a NumPy array, which will remove the index and store the data as a raw matrix
numpy_array = df_embbedings_src.to_numpy()

# Convert the NumPy array into a PyTorch FloatTensor, which is the format required for PyTorch operations
x_src = torch.FloatTensor(numpy_array)
```


```python
# Read the target embeddings from a CSV file into a pandas DataFrame
df_embbedings_tgt = pd.read_csv(tgt_Emb, index_col=0)

# Convert the DataFrame to a NumPy array, which removes the index and converts the data to a raw matrix
numpy_array = df_embbedings_tgt.to_numpy()

# Convert the NumPy array into a PyTorch FloatTensor, which is required for PyTorch operations
x_tgt = torch.FloatTensor(numpy_array)
```

# Reading adjacency Matrix


```python
# Read the source adjacency matrix from a CSV file into a pandas DataFrame
df_ma1 = pd.read_csv(src_Adjacence, index_col=0)

# Convert the DataFrame to a list of lists (Python native list format)
ma1 = df_ma1.values.tolist()
```


```python
# Read the target adjacency matrix from a CSV file into a pandas DataFrame
df_ma2 = pd.read_csv(tgt_Adjacence, index_col=0)

# Convert the DataFrame to a list of lists (Python native list format)
ma2 = df_ma2.values.tolist()
```

# Convert Adjacency matrix (in list format) to an undirected edge index


```python
# Convert the source adjacency matrix (in list format) to an undirected edge index for PyTorch Geometric
edge_src = adjacency_matrix_to_undirected_edge_index(ma1)

# Convert the target adjacency matrix (in list format) to an undirected edge index for PyTorch Geometric
edge_tgt = adjacency_matrix_to_undirected_edge_index(ma2)
```

# GIT Training


```python
def train_model_gnn(model, x_src, edge_src, x_tgt, edge_tgt,
                    tensor_term1, tensor_term2, tensor_score,
                    learning_rate, weight_decay_value, num_epochs, print_interval=10):
    """
    Trains a graph neural network (GNN) model using source and target embeddings and contrastive loss.

    Args:
        model: The GNN model to be trained.
        x_src (torch.Tensor): Source node embeddings.
        edge_src (torch.Tensor): Source graph edges.
        x_tgt (torch.Tensor): Target node embeddings.
        edge_tgt (torch.Tensor): Target graph edges.
        tensor_term1 (torch.Tensor): Indices of the source nodes to be compared.
        tensor_term2 (torch.Tensor): Indices of the target nodes to be compared.
        tensor_score (torch.Tensor): Labels indicating if the pairs are matched (1) or not (0).
        learning_rate (float): Learning rate for the optimizer.
        weight_decay_value (float): Weight decay (L2 regularization) value for the optimizer.
        num_epochs (int): Number of epochs for training.
        print_interval (int): Interval at which training progress is printed (every `print_interval` epochs).

    Returns:
        model: The trained GNN model.
    """

    # Step 1: Set device (GPU or CPU) for computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 2: Move the model and all inputs to the selected device
    model.to(device)
    x_tgt = x_tgt.to(device)               # Target node embeddings
    edge_tgt = edge_tgt.to(device)         # Target graph edges
    x_src = x_src.to(device)               # Source node embeddings
    edge_src = edge_src.to(device)         # Source graph edges
    tensor_term1 = tensor_term1.to(device) # Indices for source nodes
    tensor_term2 = tensor_term2.to(device) # Indices for target nodes
    tensor_score = tensor_score.to(device) # Ground truth labels

    # Step 3: Define optimizer with learning rate and regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)

    # Step 4: Initialize list to store training losses
    train_losses = []

    # Record the start time of training
    start_time = time.time()

    # Step 5: Training loop
    for epoch in range(num_epochs):
        # Zero out gradients from the previous iteration
        optimizer.zero_grad()

        # Forward pass: Compute embeddings for source and target graphs
        out1 = model(x_src, edge_src)  # Updated source embeddings
        out2 = model(x_tgt, edge_tgt)  # Updated target embeddings

        # Extract specific rows of embeddings for terms being compared
        src_embeddings = select_rows_by_index(out1, tensor_term1)
        tgt_embeddings = select_rows_by_index(out2, tensor_term2)

        # Compute contrastive loss based on the embeddings and ground truth labels
        loss = contrastive_loss(src_embeddings, tgt_embeddings, tensor_score)

        # Backward pass: Compute gradients
        loss.backward()

        # Update the model's parameters
        optimizer.step()

        # Append the loss for this iteration to the list
        train_losses.append(loss.item())

        # Print loss every `print_interval` epochs
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item()}")

    # Step 6: Record end time of training
    end_time = time.time()

    # Step 7: Plot the training loss over time
    plt.semilogy(range(1, num_epochs + 1), train_losses, label="Training Loss", marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Print the total training time
    training_time = end_time - start_time
    print(f"Training complete! Total training time: {training_time:.2f} seconds")

    # Step 8: Return the trained model
    return model
```


```python
# Initialize the GIT_mod model with the dimensionality of the target embeddings
# The first argument is the dimensionality of the target node embeddings (x_tgt.shape[1])
# The second argument (1) represents the number of RGIT layers in the model
GIT_model = RGIT_mod(x_tgt.shape[1], 1)
```


```python
# Reading the training pairs from a CSV file into a pandas DataFrame
df_embbedings = pd.read_csv(train_file, index_col=0)

# Extract the 'SrcEntity' and 'TgtEntity' columns as NumPy arrays and convert them to integers
tensor_term1 = df_embbedings['SrcEntity'].values.astype(int)  # Source entity indices
tensor_term2 = df_embbedings['TgtEntity'].values.astype(int)  # Target entity indices

# Extract the 'Score' column as a NumPy array and convert it to floats
tensor_score = df_embbedings['Score'].values.astype(float)  # Scores (labels) indicating if pairs match (1) or not (0)

# Convert the NumPy arrays to PyTorch LongTensors (for indices) and FloatTensors (for scores)
tensor_term1_o = torch.from_numpy(tensor_term1).type(torch.LongTensor)  # Source entity tensor
tensor_term2_o = torch.from_numpy(tensor_term2).type(torch.LongTensor)  # Target entity tensor
tensor_score_o = torch.from_numpy(tensor_score).type(torch.FloatTensor)  # Score tensor
```


```python
# Train the GNN model using the provided source and target graph embeddings, edges, and training data
trained_model = train_model_gnn(
    model=GIT_model,                # The GNN model to be trained (initialized earlier)
    x_src=x_src,                    # Source node embeddings (tensor for source graph)
    edge_src=edge_src,              # Source graph edges (undirected edge index for source graph)
    x_tgt=x_tgt,                    # Target node embeddings (tensor for target graph)
    edge_tgt=edge_tgt,              # Target graph edges (undirected edge index for target graph)
    tensor_term1=tensor_term1_o,    # Indices of source entities for training
    tensor_term2=tensor_term2_o,    # Indices of target entities for training
    tensor_score=tensor_score_o,    # Scores (labels) indicating if pairs match (1) or not (0)
    learning_rate=0.0001,            # Learning rate for the Adam optimizer
    weight_decay_value=1e-4,        # Weight decay for L2 regularization to prevent overfitting
    num_epochs=1000,                # Number of training epochs
    print_interval=10               # Interval at which to print training progress (every 10 epochs)
)
```

    Epoch [10/1000], Training Loss: 0.003323144977912307
    Epoch [20/1000], Training Loss: 0.002758753951638937
    Epoch [30/1000], Training Loss: 0.002405626932159066
    Epoch [40/1000], Training Loss: 0.0021553668193519115
    Epoch [50/1000], Training Loss: 0.0019685907755047083
    Epoch [60/1000], Training Loss: 0.001820491743274033
    Epoch [70/1000], Training Loss: 0.0016976859187707305
    Epoch [80/1000], Training Loss: 0.0015936832642182708
    Epoch [90/1000], Training Loss: 0.0015044499887153506
    Epoch [100/1000], Training Loss: 0.0014269719831645489
    Epoch [110/1000], Training Loss: 0.0013590113958343863
    Epoch [120/1000], Training Loss: 0.0012983985943719745
    Epoch [130/1000], Training Loss: 0.0012442731531336904
    Epoch [140/1000], Training Loss: 0.0011956796515733004
    Epoch [150/1000], Training Loss: 0.0011515321675688028
    Epoch [160/1000], Training Loss: 0.0011111898347735405
    Epoch [170/1000], Training Loss: 0.001073987688869238
    Epoch [180/1000], Training Loss: 0.001039916998706758
    Epoch [190/1000], Training Loss: 0.0010080942884087563
    Epoch [200/1000], Training Loss: 0.000978390802629292
    Epoch [210/1000], Training Loss: 0.0009506894275546074
    Epoch [220/1000], Training Loss: 0.0009247888810932636
    Epoch [230/1000], Training Loss: 0.0009003327577374876
    Epoch [240/1000], Training Loss: 0.0008772015571594238
    Epoch [250/1000], Training Loss: 0.0008555636159144342
    Epoch [260/1000], Training Loss: 0.0008350859279744327
    Epoch [270/1000], Training Loss: 0.0008157147094607353
    Epoch [280/1000], Training Loss: 0.0007974862237460911
    Epoch [290/1000], Training Loss: 0.0007802325417287648
    Epoch [300/1000], Training Loss: 0.0007637517992407084
    Epoch [310/1000], Training Loss: 0.0007480462663806975
    Epoch [320/1000], Training Loss: 0.0007330261287279427
    Epoch [330/1000], Training Loss: 0.0007186235743574798
    Epoch [340/1000], Training Loss: 0.0007048145635053515
    Epoch [350/1000], Training Loss: 0.0006914790719747543
    Epoch [360/1000], Training Loss: 0.0006785988807678223
    Epoch [370/1000], Training Loss: 0.0006662089726887643
    Epoch [380/1000], Training Loss: 0.0006542041664943099
    Epoch [390/1000], Training Loss: 0.0006425530882552266
    Epoch [400/1000], Training Loss: 0.000631225120741874
    Epoch [410/1000], Training Loss: 0.0006201670621521771
    Epoch [420/1000], Training Loss: 0.0006093672709539533
    Epoch [430/1000], Training Loss: 0.0005988220800645649
    Epoch [440/1000], Training Loss: 0.0005885244463570416
    Epoch [450/1000], Training Loss: 0.0005785101093351841
    Epoch [460/1000], Training Loss: 0.0005687700468115509
    Epoch [470/1000], Training Loss: 0.0005593542009592056
    Epoch [480/1000], Training Loss: 0.0005501640262082219
    Epoch [490/1000], Training Loss: 0.0005412035970948637
    Epoch [500/1000], Training Loss: 0.000532445206772536
    Epoch [510/1000], Training Loss: 0.0005239159218035638
    Epoch [520/1000], Training Loss: 0.0005155563703738153
    Epoch [530/1000], Training Loss: 0.0005074359942227602
    Epoch [540/1000], Training Loss: 0.0004995737108401954
    Epoch [550/1000], Training Loss: 0.0004919569473713636
    Epoch [560/1000], Training Loss: 0.0004846010124310851
    Epoch [570/1000], Training Loss: 0.00047750596422702074
    Epoch [580/1000], Training Loss: 0.0004706147010438144
    Epoch [590/1000], Training Loss: 0.0004637359525077045
    Epoch [600/1000], Training Loss: 0.0004573380865622312
    Epoch [610/1000], Training Loss: 0.00045107121695764363
    Epoch [620/1000], Training Loss: 0.00044467890984378755
    Epoch [630/1000], Training Loss: 0.00043841268052347004
    Epoch [640/1000], Training Loss: 0.00043217980419285595
    Epoch [650/1000], Training Loss: 0.00042686803499236703
    Epoch [660/1000], Training Loss: 0.0004212624626234174
    Epoch [670/1000], Training Loss: 0.00041546719148755074
    Epoch [680/1000], Training Loss: 0.0004096776247024536
    Epoch [690/1000], Training Loss: 0.00040498876478523016
    Epoch [700/1000], Training Loss: 0.0003994737344328314
    Epoch [710/1000], Training Loss: 0.00039359796210192144
    Epoch [720/1000], Training Loss: 0.0003883528697770089
    Epoch [730/1000], Training Loss: 0.0003837967524304986
    Epoch [740/1000], Training Loss: 0.0003779302351176739
    Epoch [750/1000], Training Loss: 0.0003720169479493052
    Epoch [760/1000], Training Loss: 0.0003669490397442132
    Epoch [770/1000], Training Loss: 0.0003623177472036332
    Epoch [780/1000], Training Loss: 0.0003564161597751081
    Epoch [790/1000], Training Loss: 0.00035119522362947464
    Epoch [800/1000], Training Loss: 0.00034722182317636907
    Epoch [810/1000], Training Loss: 0.00034295261139050126
    Epoch [820/1000], Training Loss: 0.0003381603746674955
    Epoch [830/1000], Training Loss: 0.0003343985590618104
    Epoch [840/1000], Training Loss: 0.000331773393554613
    Epoch [850/1000], Training Loss: 0.0003277690848335624
    Epoch [860/1000], Training Loss: 0.00032415398163720965
    Epoch [870/1000], Training Loss: 0.00032155803637579083
    Epoch [880/1000], Training Loss: 0.0003190614515915513
    Epoch [890/1000], Training Loss: 0.0003153412544634193
    Epoch [900/1000], Training Loss: 0.0003121505433227867
    Epoch [910/1000], Training Loss: 0.00031007430516183376
    Epoch [920/1000], Training Loss: 0.00030730810249224305
    Epoch [930/1000], Training Loss: 0.00030366345890797675
    Epoch [940/1000], Training Loss: 0.0003012649540323764
    Epoch [950/1000], Training Loss: 0.0002988525084219873
    Epoch [960/1000], Training Loss: 0.0002949452609755099
    Epoch [970/1000], Training Loss: 0.00029280706075951457
    Epoch [980/1000], Training Loss: 0.0002897852682508528
    Epoch [990/1000], Training Loss: 0.00028585526160895824
    Epoch [1000/1000], Training Loss: 0.000283512199530378



    
![png](output_51_1.png)
    


    Training complete! Total training time: 1133.52 seconds


# GIT Application


```python
# Determine if a GPU is available and move the computations to it; otherwise, use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming the model has been trained and hyperparameters (x_src, edge_src, x_tgt, edge_tgt) are set

# Move the trained GIT_model to the device (GPU or CPU)
GIT_model.to(device)

# Move the data tensors to the same device (GPU or CPU)
x_tgt = x_tgt.to(device)         # Target node embeddings
edge_tgt = edge_tgt.to(device)   # Target graph edges
x_src = x_src.to(device)         # Source node embeddings
edge_src = edge_src.to(device)   # Source graph edges

# Set the model to evaluation mode; this disables dropout and batch normalization
GIT_model.eval()

# Pass the source and target embeddings through the trained GNN model to update the embeddings
with torch.no_grad():  # Disable gradient computation (inference mode)
    embeddings_tgt = GIT_model(x_tgt, edge_tgt)  # Get updated embeddings for the target graph
    embeddings_src = GIT_model(x_src, edge_src)  # Get updated embeddings for the source graph

# Detach the embeddings from the computation graph and move them back to the CPU
# This step is useful if you need to use the embeddings for tasks outside PyTorch (e.g., saving to disk)
embeddings_tgt = embeddings_tgt.detach().cpu()  # Target graph embeddings
embeddings_src = embeddings_src.detach().cpu()  # Source graph embeddings

# At this point, embeddings_tgt and embeddings_src contain the updated embeddings, ready for downstream tasks
```

# Selecting embedding pairs to train the Gated Network


```python
# Read the training pairs from a CSV file into a pandas DataFrame
df_embeddings = pd.read_csv(train_file, index_col=0)

# Extract columns and convert to NumPy arrays
tensor_term1 = df_embeddings['SrcEntity'].values.astype(int)  # Source entity indices
tensor_term2 = df_embeddings['TgtEntity'].values.astype(int)  # Target entity indices
tensor_score = df_embeddings['Score'].values.astype(float)  # Matching scores

# Split data into training and validation sets
tensor_term1_train, tensor_term1_val, tensor_term2_train, tensor_term2_val, tensor_score_train, tensor_score_val = train_test_split(
    tensor_term1, tensor_term2, tensor_score, test_size=0.3, random_state=42
)

# Convert split data to PyTorch tensors
tensor_term1_train = torch.from_numpy(tensor_term1_train).type(torch.LongTensor)
tensor_term2_train = torch.from_numpy(tensor_term2_train).type(torch.LongTensor)
tensor_score_train = torch.from_numpy(tensor_score_train).type(torch.FloatTensor)

tensor_term1_val = torch.from_numpy(tensor_term1_val).type(torch.LongTensor)
tensor_term2_val = torch.from_numpy(tensor_term2_val).type(torch.LongTensor)
tensor_score_val = torch.from_numpy(tensor_score_val).type(torch.FloatTensor)

# Move the embeddings back to the CPU if not already there
x_tgt = x_tgt.cpu()  # Target node embeddings
x_src = x_src.cpu()  # Source node embeddings

# Select embeddings for the training set
X1_train = select_rows_by_index(embeddings_src, tensor_term1_train)
X2_train = select_rows_by_index(x_src, tensor_term1_train)
X3_train = select_rows_by_index(embeddings_tgt, tensor_term2_train)
X4_train = select_rows_by_index(x_tgt, tensor_term2_train)

# Select embeddings for the validation set
X1_val = select_rows_by_index(embeddings_src, tensor_term1_val)
X2_val = select_rows_by_index(x_src, tensor_term1_val)
X3_val = select_rows_by_index(embeddings_tgt, tensor_term2_val)
X4_val = select_rows_by_index(x_tgt, tensor_term2_val)

# Now you have:
# - Training tensors: X1_train, X2_train, X3_train, X4_train, tensor_score_train
# - Validation tensors: X1_val, X2_val, X3_val, X4_val, tensor_score_val
```

# Gated Network Training


```python
from sklearn.metrics import f1_score

def train_gated_combination_model(X1_t, X2_t, X3_t, X4_t, tensor_score_o,
                                  X1_val, X2_val, X3_val, X4_val, tensor_score_val,
                                  epochs=120, batch_size=32, learning_rate=0.001, weight_decay=1e-5):
    """
    Trains the GatedCombination model with training and validation data, using ReduceLROnPlateau scheduler.
    Also calculates and displays F1-score during training and validation.
    """

    # Create datasets and DataLoaders
    train_dataset = TensorDataset(X1_t, X2_t, X3_t, X4_t, tensor_score_o)
    val_dataset = TensorDataset(X1_val, X2_val, X3_val, X4_val, tensor_score_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedCombination(X1_t.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Use ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    train_losses, val_losses = [], []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        y_true_train, y_pred_train = [], []

        for batch_X1, batch_X2, batch_X3, batch_X4, batch_y in train_loader:
            batch_X1, batch_X2, batch_X3, batch_X4, batch_y = (
                batch_X1.to(device),
                batch_X2.to(device),
                batch_X3.to(device),
                batch_X4.to(device),
                batch_y.to(device),
            )
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X1, batch_X2, batch_X3, batch_X4)

            # Compute loss
            loss = F.binary_cross_entropy(outputs, batch_y.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # Store true labels and predictions for F1-score
            y_true_train.extend(batch_y.cpu().numpy())
            y_pred_train.extend((outputs > 0.5).float().cpu().numpy())

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Calculate F1-score for training
        train_f1 = f1_score(y_true_train, y_pred_train)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for batch_X1, batch_X2, batch_X3, batch_X4, batch_y in val_loader:
                batch_X1, batch_X2, batch_X3, batch_X4, batch_y = (
                    batch_X1.to(device),
                    batch_X2.to(device),
                    batch_X3.to(device),
                    batch_X4.to(device),
                    batch_y.to(device),
                )
                outputs = model(batch_X1, batch_X2, batch_X3, batch_X4)

                # Compute loss
                val_loss = F.binary_cross_entropy(outputs, batch_y.unsqueeze(1).float())
                total_val_loss += val_loss.item()

                # Store true labels and predictions for F1-score
                y_true_val.extend(batch_y.cpu().numpy())
                y_pred_val.extend((outputs > 0.5).float().cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate F1-score for validation
        val_f1 = f1_score(y_true_val, y_pred_val)

        # Step the scheduler with validation loss
        scheduler.step(avg_val_loss)

        # Print training and validation metrics
        print(f"Epoch [{epoch + 1}/{epochs}] Training Loss: {train_loss:.4f}, F1 Score: {train_f1:.4f} | "
              f"Validation Loss: {avg_val_loss:.4f}, F1 Score: {val_f1:.4f}")

    end_time = time.time()

    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print(f"Training complete! Total time: {end_time - start_time:.2f} seconds")
    return model


```


```python
# Train the GatedCombination model using training and validation data
trained_model = train_gated_combination_model(
    X1_train,          # Updated source embeddings (after applying the GNN model)
    X2_train,          # Original source embeddings (before applying the GNN model)
    X3_train,          # Updated target embeddings (after applying the GNN model)
    X4_train,          # Original target embeddings (before applying the GNN model)
    tensor_score_train, # Ground truth labels for the training set (1 for matched pairs, 0 for unmatched pairs)

    X1_val,            # Updated source embeddings for the validation set
    X2_val,            # Original source embeddings for the validation set
    X3_val,            # Updated target embeddings for the validation set
    X4_val,            # Original target embeddings for the validation set
    tensor_score_val,  # Ground truth labels for the validation set (1 for matched pairs, 0 for unmatched pairs)

    epochs=100,        # Number of epochs (iterations over the entire training dataset)
    batch_size=32,     # Number of training samples processed in one forward/backward pass
    learning_rate=0.001, # Learning rate for the optimizer (controls step size during optimization)
    weight_decay=1e-4 # Weight decay (L2 regularization) to prevent overfitting
)
```

    Epoch [1/100] Training Loss: 0.0532, F1 Score: 0.0099 | Validation Loss: 0.0506, F1 Score: 0.0969
    Epoch [2/100] Training Loss: 0.0480, F1 Score: 0.1398 | Validation Loss: 0.0458, F1 Score: 0.1942
    Epoch [3/100] Training Loss: 0.0439, F1 Score: 0.2626 | Validation Loss: 0.0418, F1 Score: 0.3228
    Epoch [4/100] Training Loss: 0.0406, F1 Score: 0.3343 | Validation Loss: 0.0388, F1 Score: 0.3792
    Epoch [5/100] Training Loss: 0.0377, F1 Score: 0.3690 | Validation Loss: 0.0359, F1 Score: 0.4132
    Epoch [6/100] Training Loss: 0.0351, F1 Score: 0.4136 | Validation Loss: 0.0335, F1 Score: 0.4273
    Epoch [7/100] Training Loss: 0.0331, F1 Score: 0.4442 | Validation Loss: 0.0318, F1 Score: 0.5091
    Epoch [8/100] Training Loss: 0.0311, F1 Score: 0.4694 | Validation Loss: 0.0296, F1 Score: 0.5196
    Epoch [9/100] Training Loss: 0.0294, F1 Score: 0.5101 | Validation Loss: 0.0277, F1 Score: 0.5734
    Epoch [10/100] Training Loss: 0.0279, F1 Score: 0.5426 | Validation Loss: 0.0267, F1 Score: 0.6198
    Epoch [11/100] Training Loss: 0.0265, F1 Score: 0.5726 | Validation Loss: 0.0253, F1 Score: 0.6565
    Epoch [12/100] Training Loss: 0.0253, F1 Score: 0.6034 | Validation Loss: 0.0241, F1 Score: 0.6633
    Epoch [13/100] Training Loss: 0.0243, F1 Score: 0.6352 | Validation Loss: 0.0239, F1 Score: 0.6305
    Epoch [14/100] Training Loss: 0.0233, F1 Score: 0.6578 | Validation Loss: 0.0220, F1 Score: 0.7143
    Epoch [15/100] Training Loss: 0.0225, F1 Score: 0.6804 | Validation Loss: 0.0222, F1 Score: 0.6800
    Epoch [16/100] Training Loss: 0.0217, F1 Score: 0.7067 | Validation Loss: 0.0213, F1 Score: 0.6962
    Epoch [17/100] Training Loss: 0.0210, F1 Score: 0.7216 | Validation Loss: 0.0200, F1 Score: 0.7482
    Epoch [18/100] Training Loss: 0.0203, F1 Score: 0.7391 | Validation Loss: 0.0197, F1 Score: 0.7728
    Epoch [19/100] Training Loss: 0.0198, F1 Score: 0.7524 | Validation Loss: 0.0194, F1 Score: 0.7482
    Epoch [20/100] Training Loss: 0.0193, F1 Score: 0.7656 | Validation Loss: 0.0188, F1 Score: 0.7937
    Epoch [21/100] Training Loss: 0.0188, F1 Score: 0.7722 | Validation Loss: 0.0184, F1 Score: 0.7608
    Epoch [22/100] Training Loss: 0.0183, F1 Score: 0.7789 | Validation Loss: 0.0177, F1 Score: 0.7826
    Epoch [23/100] Training Loss: 0.0180, F1 Score: 0.7816 | Validation Loss: 0.0171, F1 Score: 0.8165
    Epoch [24/100] Training Loss: 0.0176, F1 Score: 0.7982 | Validation Loss: 0.0171, F1 Score: 0.7991
    Epoch [25/100] Training Loss: 0.0172, F1 Score: 0.8012 | Validation Loss: 0.0169, F1 Score: 0.7854
    Epoch [26/100] Training Loss: 0.0170, F1 Score: 0.8107 | Validation Loss: 0.0181, F1 Score: 0.7608
    Epoch [27/100] Training Loss: 0.0168, F1 Score: 0.8155 | Validation Loss: 0.0160, F1 Score: 0.8151
    Epoch [28/100] Training Loss: 0.0164, F1 Score: 0.8217 | Validation Loss: 0.0160, F1 Score: 0.8460
    Epoch [29/100] Training Loss: 0.0162, F1 Score: 0.8186 | Validation Loss: 0.0152, F1 Score: 0.8460
    Epoch [30/100] Training Loss: 0.0159, F1 Score: 0.8313 | Validation Loss: 0.0155, F1 Score: 0.8308
    Epoch [31/100] Training Loss: 0.0158, F1 Score: 0.8342 | Validation Loss: 0.0150, F1 Score: 0.8435
    Epoch [32/100] Training Loss: 0.0155, F1 Score: 0.8363 | Validation Loss: 0.0147, F1 Score: 0.8333
    Epoch [33/100] Training Loss: 0.0153, F1 Score: 0.8386 | Validation Loss: 0.0151, F1 Score: 0.8256
    Epoch [34/100] Training Loss: 0.0153, F1 Score: 0.8407 | Validation Loss: 0.0148, F1 Score: 0.8269
    Epoch [35/100] Training Loss: 0.0151, F1 Score: 0.8424 | Validation Loss: 0.0144, F1 Score: 0.8501
    Epoch [36/100] Training Loss: 0.0149, F1 Score: 0.8485 | Validation Loss: 0.0147, F1 Score: 0.8256
    Epoch [37/100] Training Loss: 0.0148, F1 Score: 0.8518 | Validation Loss: 0.0142, F1 Score: 0.8510
    Epoch [38/100] Training Loss: 0.0147, F1 Score: 0.8505 | Validation Loss: 0.0139, F1 Score: 0.8838
    Epoch [39/100] Training Loss: 0.0145, F1 Score: 0.8528 | Validation Loss: 0.0144, F1 Score: 0.8999
    Epoch [40/100] Training Loss: 0.0145, F1 Score: 0.8507 | Validation Loss: 0.0140, F1 Score: 0.8753
    Epoch [41/100] Training Loss: 0.0144, F1 Score: 0.8577 | Validation Loss: 0.0135, F1 Score: 0.8732
    Epoch [42/100] Training Loss: 0.0144, F1 Score: 0.8582 | Validation Loss: 0.0139, F1 Score: 0.8596
    Epoch [43/100] Training Loss: 0.0143, F1 Score: 0.8507 | Validation Loss: 0.0137, F1 Score: 0.8534
    Epoch [44/100] Training Loss: 0.0140, F1 Score: 0.8589 | Validation Loss: 0.0135, F1 Score: 0.8696
    Epoch [45/100] Training Loss: 0.0140, F1 Score: 0.8605 | Validation Loss: 0.0136, F1 Score: 0.8510
    Epoch [46/100] Training Loss: 0.0139, F1 Score: 0.8567 | Validation Loss: 0.0136, F1 Score: 0.8946
    Epoch [47/100] Training Loss: 0.0137, F1 Score: 0.8618 | Validation Loss: 0.0143, F1 Score: 0.8426
    Epoch [48/100] Training Loss: 0.0138, F1 Score: 0.8630 | Validation Loss: 0.0134, F1 Score: 0.8812
    Epoch [49/100] Training Loss: 0.0137, F1 Score: 0.8614 | Validation Loss: 0.0139, F1 Score: 0.8967
    Epoch [50/100] Training Loss: 0.0137, F1 Score: 0.8605 | Validation Loss: 0.0131, F1 Score: 0.8672
    Epoch [51/100] Training Loss: 0.0137, F1 Score: 0.8646 | Validation Loss: 0.0135, F1 Score: 0.9057
    Epoch [52/100] Training Loss: 0.0136, F1 Score: 0.8652 | Validation Loss: 0.0134, F1 Score: 0.8460
    Epoch [53/100] Training Loss: 0.0135, F1 Score: 0.8667 | Validation Loss: 0.0129, F1 Score: 0.8705
    Epoch [54/100] Training Loss: 0.0136, F1 Score: 0.8666 | Validation Loss: 0.0128, F1 Score: 0.8684
    Epoch [55/100] Training Loss: 0.0135, F1 Score: 0.8656 | Validation Loss: 0.0151, F1 Score: 0.9225
    Epoch [56/100] Training Loss: 0.0134, F1 Score: 0.8677 | Validation Loss: 0.0127, F1 Score: 0.8861
    Epoch [57/100] Training Loss: 0.0134, F1 Score: 0.8698 | Validation Loss: 0.0133, F1 Score: 0.9033
    Epoch [58/100] Training Loss: 0.0134, F1 Score: 0.8730 | Validation Loss: 0.0132, F1 Score: 0.8930
    Epoch [59/100] Training Loss: 0.0133, F1 Score: 0.8747 | Validation Loss: 0.0131, F1 Score: 0.8447
    Epoch [60/100] Training Loss: 0.0134, F1 Score: 0.8730 | Validation Loss: 0.0133, F1 Score: 0.8497
    Epoch [61/100] Training Loss: 0.0133, F1 Score: 0.8739 | Validation Loss: 0.0129, F1 Score: 0.8741
    Epoch [62/100] Training Loss: 0.0132, F1 Score: 0.8705 | Validation Loss: 0.0128, F1 Score: 0.8632
    Epoch [63/100] Training Loss: 0.0132, F1 Score: 0.8731 | Validation Loss: 0.0129, F1 Score: 0.8896
    Epoch [64/100] Training Loss: 0.0132, F1 Score: 0.8752 | Validation Loss: 0.0125, F1 Score: 0.8803
    Epoch [65/100] Training Loss: 0.0132, F1 Score: 0.8709 | Validation Loss: 0.0127, F1 Score: 0.8791
    Epoch [66/100] Training Loss: 0.0131, F1 Score: 0.8688 | Validation Loss: 0.0127, F1 Score: 0.8657
    Epoch [67/100] Training Loss: 0.0132, F1 Score: 0.8704 | Validation Loss: 0.0128, F1 Score: 0.9022
    Epoch [68/100] Training Loss: 0.0130, F1 Score: 0.8731 | Validation Loss: 0.0128, F1 Score: 0.8584
    Epoch [69/100] Training Loss: 0.0130, F1 Score: 0.8771 | Validation Loss: 0.0125, F1 Score: 0.8672
    Epoch [70/100] Training Loss: 0.0130, F1 Score: 0.8730 | Validation Loss: 0.0126, F1 Score: 0.8910
    Epoch [71/100] Training Loss: 0.0131, F1 Score: 0.8755 | Validation Loss: 0.0128, F1 Score: 0.8547
    Epoch [72/100] Training Loss: 0.0130, F1 Score: 0.8767 | Validation Loss: 0.0123, F1 Score: 0.8755
    Epoch [73/100] Training Loss: 0.0130, F1 Score: 0.8766 | Validation Loss: 0.0128, F1 Score: 0.8729
    Epoch [74/100] Training Loss: 0.0131, F1 Score: 0.8776 | Validation Loss: 0.0124, F1 Score: 0.8800
    Epoch [75/100] Training Loss: 0.0130, F1 Score: 0.8756 | Validation Loss: 0.0123, F1 Score: 0.8770
    Epoch [76/100] Training Loss: 0.0130, F1 Score: 0.8792 | Validation Loss: 0.0136, F1 Score: 0.9212
    Epoch [77/100] Training Loss: 0.0129, F1 Score: 0.8753 | Validation Loss: 0.0125, F1 Score: 0.8611
    Epoch [78/100] Training Loss: 0.0129, F1 Score: 0.8787 | Validation Loss: 0.0129, F1 Score: 0.8550
    Epoch [79/100] Training Loss: 0.0129, F1 Score: 0.8798 | Validation Loss: 0.0131, F1 Score: 0.8447
    Epoch [80/100] Training Loss: 0.0130, F1 Score: 0.8807 | Validation Loss: 0.0121, F1 Score: 0.8838
    Epoch [81/100] Training Loss: 0.0129, F1 Score: 0.8731 | Validation Loss: 0.0147, F1 Score: 0.9364
    Epoch [82/100] Training Loss: 0.0129, F1 Score: 0.8782 | Validation Loss: 0.0122, F1 Score: 0.8884
    Epoch [83/100] Training Loss: 0.0128, F1 Score: 0.8838 | Validation Loss: 0.0129, F1 Score: 0.8510
    Epoch [84/100] Training Loss: 0.0128, F1 Score: 0.8762 | Validation Loss: 0.0122, F1 Score: 0.8791
    Epoch [85/100] Training Loss: 0.0129, F1 Score: 0.8788 | Validation Loss: 0.0125, F1 Score: 0.8584
    Epoch [86/100] Training Loss: 0.0130, F1 Score: 0.8807 | Validation Loss: 0.0127, F1 Score: 0.9070
    Epoch [87/100] Training Loss: 0.0129, F1 Score: 0.8776 | Validation Loss: 0.0124, F1 Score: 0.8635
    Epoch [88/100] Training Loss: 0.0129, F1 Score: 0.8745 | Validation Loss: 0.0124, F1 Score: 0.8660
    Epoch [89/100] Training Loss: 0.0128, F1 Score: 0.8751 | Validation Loss: 0.0124, F1 Score: 0.8944
    Epoch [90/100] Training Loss: 0.0129, F1 Score: 0.8761 | Validation Loss: 0.0125, F1 Score: 0.8999
    Epoch [91/100] Training Loss: 0.0127, F1 Score: 0.8792 | Validation Loss: 0.0135, F1 Score: 0.8384
    Epoch [92/100] Training Loss: 0.0121, F1 Score: 0.8875 | Validation Loss: 0.0117, F1 Score: 0.8988
    Epoch [93/100] Training Loss: 0.0119, F1 Score: 0.8895 | Validation Loss: 0.0118, F1 Score: 0.9044
    Epoch [94/100] Training Loss: 0.0119, F1 Score: 0.8871 | Validation Loss: 0.0118, F1 Score: 0.8965
    Epoch [95/100] Training Loss: 0.0119, F1 Score: 0.8905 | Validation Loss: 0.0118, F1 Score: 0.9033
    Epoch [96/100] Training Loss: 0.0119, F1 Score: 0.8895 | Validation Loss: 0.0118, F1 Score: 0.8965
    Epoch [97/100] Training Loss: 0.0119, F1 Score: 0.8889 | Validation Loss: 0.0118, F1 Score: 0.8930
    Epoch [98/100] Training Loss: 0.0120, F1 Score: 0.8884 | Validation Loss: 0.0118, F1 Score: 0.9033
    Epoch [99/100] Training Loss: 0.0119, F1 Score: 0.8914 | Validation Loss: 0.0118, F1 Score: 0.8976
    Epoch [100/100] Training Loss: 0.0119, F1 Score: 0.8889 | Validation Loss: 0.0119, F1 Score: 0.9022



    
![png](output_58_1.png)
    


    Training complete! Total time: 1030.71 seconds


# **Second Round Modifications**

# **Generate Embeddings**


```python
# Build an indexed dictionary for the source ontology classes
# src_class is the file path to the JSON file containing the source ontology classes
indexed_dict_src = build_indexed_dict(src_class)

# Build an indexed dictionary for the target ontology classes
# tgt_class is the file path to the JSON file containing the target ontology classes
indexed_dict_tgt = build_indexed_dict(tgt_class)
```


```python
# Define output file paths for final embeddings of source and target ontologies
output_file_src = f"{data_dir}/{src_ent}_final_embeddings_o_preflabel.tsv"
output_file_tgt = f"{data_dir}/{tgt_ent}_final_embeddings_o_preflabel.tsv"

# Save the final gated embeddings for all concepts in source and target ontologies
save_gated_embeddings(
    gated_model=trained_model,          # The trained GatedCombination model
    embeddings_src=embeddings_src,      # GNN-transformed embeddings for source entities
    x_src=x_src,                        # Initial semantic embeddings for source entities
    embeddings_tgt=embeddings_tgt,      # GNN-transformed embeddings for target entities
    x_tgt=x_tgt,                        # Initial semantic embeddings for target entities
    indexed_dict_src=indexed_dict_src,  # Index-to-URI mapping for source ontology
    indexed_dict_tgt=indexed_dict_tgt,  # Index-to-URI mapping for target ontology
    output_file_src=output_file_src,    # Destination file path for source embeddings
    output_file_tgt=output_file_tgt     # Destination file path for target embeddings
)

```

    ✅ Gated embeddings saved:
    - Source: /content/gdrive/My Drive/BioGITOM-VLDB//pharm/Data/snomed.pharm_final_embeddings_o_preflabel.tsv
    - Target: /content/gdrive/My Drive/BioGITOM-VLDB//pharm/Data/ncit.pharm_final_embeddings_o_preflabel.tsv
    ⏱️ Execution time: 39.26 seconds


# **Filter No Used Concepts**





```python
# 🔍 Filter ignored ontology concepts before alignment

# This function filters out concepts from the source and target embeddings
# that are marked with the annotation property "used in alignment = false"
# (e.g., owl:Thing, deprecated classes, or auxiliary terms not intended for alignment).

# Inputs:
# - src_emb_path: Path to the TSV file with embeddings for all source ontology concepts.
# - tgt_emb_path: Path to the TSV file with embeddings for all target ontology concepts.
# - src_onto / tgt_onto: DeepOnto OWLOntology objects representing the source and target ontologies.
#   These are used to access logical axioms and annotations, including the "used in alignment" flag.

# Behavior:
# - The function loads the embeddings and filters out all concepts where the ontology
#   contains an annotation: `usedInAlignment = false` (typically via an OWL annotation property).
# - These filtered concepts are ignored in the downstream matching process.

# Outputs:
# - src_file: Path to the cleaned TSV file of source embeddings (only concepts used in alignment).
# - tgt_file: Path to the cleaned TSV file of target embeddings.

src_file, tgt_file = filter_ignored_class(
    src_emb_path=f"{data_dir}/{src_ent}_final_embeddings_o_preflabel.tsv",
    tgt_emb_path=f"{data_dir}/{tgt_ent}_final_embeddings_o_preflabel.tsv",
    src_onto=src_onto,
    tgt_onto=tgt_onto
)
```

    🔍 Initial source file: 16179 rows
    🔍 Initial target file: 15465 rows
    ✅ Source after removing ignored classes: 16179 rows
    ✅ Target after removing ignored classes: 15465 rows
    📁 Cleaned source file saved to: /content/gdrive/My Drive/BioGITOM-VLDB//pharm/Data/snomed.pharm_final_embeddings_o_preflabel_cleaned.tsv
    📁 Cleaned target file saved to: /content/gdrive/My Drive/BioGITOM-VLDB//pharm/Data/ncit.pharm_final_embeddings_o_preflabel_cleaned.tsv


# **Mappings Generation**

# **Using faiss l2**


```python
# Compute the top-10 most similar mappings using l2 distance
# between ResMLP-encoded embeddings of the source and target ontologies.
# The input embeddings were previously encoded using the ResMLPEncoder,
# and the similarity score is computed as the inverse of the l2 distance.
# Results are saved in a TSV file with columns: SrcEntity, TgtEntity, Score.
topk_faiss_l2(
    src_emb_path=f"{data_dir}/{src_ent}_final_embeddings_o_preflabel_cleaned.tsv",
    tgt_emb_path=f"{data_dir}/{tgt_ent}_final_embeddings_o_preflabel_cleaned.tsv",
    top_k=3,
    output_file=f"{results_dir}/{task}_top_3_mappings_faiss_l2_o_preflabel.tsv"
)
```

    🔹 Using L2 (Euclidean) distance with FAISS
    Top-3 FAISS similarity results saved to: /content/gdrive/My Drive/BioGITOM-VLDB//pharm/Results/pharm_top_3_mappings_faiss_l2_o_preflabel.tsv
    ⏱️ Execution time: 14.12 seconds


# **Evaluation**

# **Global Metrics: Precision, Recall and F1 score**


```python
# Run the evaluation on the predicted top-1 mappings using a filtering and evaluation function.

output_file, metrics, correct = evaluate_predictions(
    pred_file=f"{results_dir}/{task}_top_3_mappings_faiss_l2_o_preflabel.tsv",
    # Path to the TSV file containing predicted mappings with scores (before filtering).

    train_file=f"{dataset_dir}/refs_equiv/train.tsv",
    # Path to the training reference file (used to exclude mappings involving train-only entities).

    test_file=f"{dataset_dir}/refs_equiv/test.tsv",
    # Path to the test reference file (used as the gold standard for evaluation).
)

# This function returns:
# - `output_file`: the path to the filtered and evaluated output file.
# - `metrics`: a tuple containing (Precision, Recall, F1-score).
# - `correct`: the number of correctly predicted mappings found in the gold standard.

```

    🏆 Selected candidates within 99.7% of best score per SrcEntity: 4009 rows
       ➤ Mappings file:   /content/gdrive/My Drive/BioGITOM-VLDB//pharm/Results/pharm_matching_results.tsv
    
    🎯 Evaluation Summary:
       - Correct mappings:     2804
       - Total predictions:    4009
       - Total references:     4062
    📊 Precision:              0.699
    📊 Recall:                 0.690
    📊 F1-score:               0.695
    


# **Metrics@1**


```python
# Compute the top-1 most similar mappings using l2 distance
# and the similarity score is computed as the inverse of the l2 distance.
# Results are saved in a TSV file with columns: SrcEntity, TgtEntity, Score.
topk_faiss_l2(
    src_emb_path=f"{data_dir}/{src_ent}_final_embeddings_o_preflabel_cleaned.tsv",
    tgt_emb_path=f"{data_dir}/{tgt_ent}_final_embeddings_o_preflabel_cleaned.tsv",
    top_k=1,
    output_file=f"{results_dir}/{task}_top_1_mappings_o_preflabel.tsv"
)
```

    🔹 Using L2 (Euclidean) distance with FAISS
    Top-1 FAISS similarity results saved to: /content/gdrive/My Drive/BioGITOM-VLDB//pharm/Results/pharm_top_1_mappings_o_preflabel.tsv
    ⏱️ Execution time: 7.91 seconds



```python
# === Evaluate Top-1 Mappings ===

results = evaluate_topk(
    topk_file=f"{results_dir}/{task}_top_1_mappings_o_preflabel.tsv",
    # Path to the file containing the predicted mappings with scores.
    # This file may include unfiltered predictions (e.g., over all candidates).

    train_file=f"{dataset_dir}/refs_equiv/train.tsv",
    # Path to the training reference mappings file.
    # Used to remove mappings that involve entities appearing only in training.

    test_file=f"{dataset_dir}/refs_equiv/test.tsv",
    # Path to the test reference mappings file.
    # Ground-truth correspondences are extracted from this file for evaluation.

    k=1  # Evaluate top-1 predictions per source entity.
)
```

    📊 Precision@1:            0.790
    📊 Recall@1:               0.776
    📊 F1@1:                   0.783
    


# **Local MRR and Hit@k**


```python
import pandas as pd

# === Step 1: Load input files ===

# Define paths to cleaned embedding files
src_emb_path = f"{data_dir}/{src_ent}_final_embeddings_o_preflabel_cleaned.tsv"
tgt_emb_path = f"{data_dir}/{tgt_ent}_final_embeddings_o_preflabel_cleaned.tsv"

# Load candidate mappings (SrcEntity, TgtEntity) and source/target embeddings
df_cands = pd.read_csv(cands_path)
src_emb_df = pd.read_csv(src_emb_path, sep="\t")
tgt_emb_df = pd.read_csv(tgt_emb_path, sep="\t")

# === Step 2: Extract unique source and target URIs from the candidate pairs ===

# Keep only distinct source and target entities (URIs) for which embeddings are needed
unique_src_df = pd.DataFrame(df_cands["SrcEntity"].unique(), columns=["Concept"])
unique_tgt_df = pd.DataFrame(df_cands["TgtEntity"].unique(), columns=["Concept"])

# === Step 3: Join embeddings for each concept based on the "Concept" URI ===

# Merge source entities with their corresponding embeddings (if available)
merged_src_df = pd.merge(unique_src_df, src_emb_df, on="Concept", how="left")

# Merge target entities with their corresponding embeddings (if available)
merged_tgt_df = pd.merge(unique_tgt_df, tgt_emb_df, on="Concept", how="left")

# === Step 4: Save the merged results to TSV files ===

# Save the source concepts and their embeddings to file
merged_src_df.to_csv(f"{data_dir}/{src_ent}_cands_with_embeddings_o_preflabel.tsv", sep="\t", index=False)

# Save the target concepts and their embeddings to file
merged_tgt_df.to_csv(f"{data_dir}/{tgt_ent}_cands_with_embeddings_o_preflabel.tsv", sep="\t", index=False)
```


```python
topk_faiss_l2(
    # Path to the source entity embeddings (already filtered and linearly encoded)
    src_emb_path=f"{data_dir}/{src_ent}_cands_with_embeddings_o_preflabel.tsv",

    # Path to the target entity embeddings (already filtered and linearly encoded)
    tgt_emb_path=f"{data_dir}/{tgt_ent}_cands_with_embeddings_o_preflabel.tsv",

    # Number of top matches to retrieve per source entity (Top-K candidates)
    top_k=200,

    # Path to save the resulting Top-K mappings sorted by FAISS L2 distance (converted to similarity)
    output_file=f"{results_dir}/{task}_top_200_mappings_mrr_hit_o_preflabel.tsv"
)

```

    🔹 Using L2 (Euclidean) distance with FAISS
    Top-200 FAISS similarity results saved to: /content/gdrive/My Drive/BioGITOM-VLDB//pharm/Results/pharm_top_200_mappings_mrr_hit_o_preflabel.tsv
    ⏱️ Execution time: 11.94 seconds



```python
# Format the prediction scores into ranked candidate lists per source entity,
# using the gold standard candidate file as reference. This prepares the output
# for MRR and Hits@k evaluation. The output is a TSV file with columns:
# SrcEntity, TgtEntity (ground truth), and TgtCandidates (ranked list with scores).
format_ranked_predictions_for_mrr(
    reference_file=f"{dataset_dir}/refs_equiv/test.cands.tsv",        # Gold reference with candidate sets
    predicted_file=f"{results_dir}/{task}_top_200_mappings_mrr_hit_o_preflabel.tsv",  # Flat prediction scores (Src, Tgt, Score)
    output_file=formatted_predictions_path                             # Output path for ranked candidate format
)

```

    ✅ Ranked predictions saved for evaluation: /content/gdrive/My Drive/BioGITOM-VLDB//pharm/Results/pharm_formatted_predictions.tsv





    '/content/gdrive/My Drive/BioGITOM-VLDB//pharm/Results/pharm_formatted_predictions.tsv'




```python
# Evaluate ranking performance using standard metrics like MRR and Hits@K
# 'formatted_predictions_path' should point to a TSV file with columns: SrcEntity, TgtEntity, TgtCandidates
# This function computes how well the true targets are ranked among the candidates
results = ranking_eval(formatted_predictions_path, Ks=[1, 5, 10])

# Print the evaluation results for Hits@1, Hits@5, and Hits@10
print("Ranking Evaluation Results at K=1, 5, and 10:")
print(results)
```

    Ranking Evaluation Results at K=1, 5, and 10:
    {'MRR': 0.8734283436834158, 'Hits@1': 0.7890201870999508, 'Hits@5': 0.9739044805514525, 'Hits@10': 0.9854751354012802}

