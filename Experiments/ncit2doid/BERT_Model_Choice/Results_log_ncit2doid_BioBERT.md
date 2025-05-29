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
      Downloading fsspec-2025.5.0-py3-none-any.whl.metadata (11 kB)
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
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m62.0/62.0 kB[0m [31m2.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting pillow!=8.3.*,>=5.3.0 (from torchvision==0.21.0)
      Downloading pillow-11.2.1-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (8.9 kB)
    Collecting mpmath<1.4,>=1.1.0 (from sympy==1.13.1->torch==2.6.0)
      Downloading mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
    Collecting MarkupSafe>=2.0 (from jinja2->torch==2.6.0)
      Downloading MarkupSafe-3.0.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)
    Downloading torch-2.6.0-cp311-cp311-manylinux1_x86_64.whl (766.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m766.7/766.7 MB[0m [31m1.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading torchvision-0.21.0-cp311-cp311-manylinux1_x86_64.whl (7.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.2/7.2 MB[0m [31m51.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl (363.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m363.4/363.4 MB[0m [31m1.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (13.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.8/13.8 MB[0m [31m121.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (24.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m24.6/24.6 MB[0m [31m72.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (883 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m883.7/883.7 kB[0m [31m43.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl (664.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m664.8/664.8 MB[0m [31m1.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl (211.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m211.5/211.5 MB[0m [31m1.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl (56.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.3/56.3 MB[0m [31m22.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl (127.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m127.9/127.9 MB[0m [31m9.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl (207.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m207.5/207.5 MB[0m [31m5.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cusparselt_cu12-0.6.2-py3-none-manylinux2014_x86_64.whl (150.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m150.1/150.1 MB[0m [31m7.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl (188.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m188.7/188.7 MB[0m [31m5.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.1/21.1 MB[0m [31m99.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_nvtx_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (99 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m99.1/99.1 kB[0m [31m6.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading sympy-1.13.1-py3-none-any.whl (6.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.2/6.2 MB[0m [31m108.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading triton-3.2.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (253.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m253.2/253.2 MB[0m [31m4.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pillow-11.2.1-cp311-cp311-manylinux_2_28_x86_64.whl (4.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.6/4.6 MB[0m [31m95.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading typing_extensions-4.13.2-py3-none-any.whl (45 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m45.8/45.8 kB[0m [31m3.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading filelock-3.18.0-py3-none-any.whl (16 kB)
    Downloading fsspec-2025.5.0-py3-none-any.whl (196 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m196.2/196.2 kB[0m [31m15.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading jinja2-3.1.6-py3-none-any.whl (134 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.9/134.9 kB[0m [31m10.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading networkx-3.4.2-py3-none-any.whl (1.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m60.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading numpy-2.2.6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m16.8/16.8 MB[0m [31m112.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading MarkupSafe-3.0.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (23 kB)
    Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m536.2/536.2 kB[0m [31m35.9 MB/s[0m eta [36m0:00:00[0m
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
        Found existing installation: fsspec 2025.3.2
        Uninstalling fsspec-2025.3.2:
          Successfully uninstalled fsspec-2025.3.2
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
    Successfully installed MarkupSafe-3.0.2 filelock-3.18.0 fsspec-2025.5.0 jinja2-3.1.6 mpmath-1.3.0 networkx-3.4.2 numpy-2.2.6 nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-cusparselt-cu12-0.6.2 nvidia-nccl-cu12-2.21.5 nvidia-nvjitlink-cu12-12.4.127 nvidia-nvtx-cu12-12.4.127 pillow-11.2.1 sympy-1.13.1 torch-2.6.0 torchvision-0.21.0 triton-3.2.0 typing-extensions-4.13.2





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
    Collecting optuna
      Downloading optuna-4.3.0-py3-none-any.whl.metadata (17 kB)
    Collecting alembic>=1.5.0 (from optuna)
      Downloading alembic-1.16.1-py3-none-any.whl.metadata (7.3 kB)
    Collecting colorlog (from optuna)
      Downloading colorlog-6.9.0-py3-none-any.whl.metadata (10 kB)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from optuna) (2.2.6)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from optuna) (25.0)
    Collecting sqlalchemy>=1.4.2 (from optuna)
      Downloading sqlalchemy-2.0.41-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.6 kB)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from optuna) (4.67.1)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.11/dist-packages (from optuna) (6.0.2)
    Requirement already satisfied: Mako in /usr/lib/python3/dist-packages (from alembic>=1.5.0->optuna) (1.1.3)
    Requirement already satisfied: typing-extensions>=4.12 in /usr/local/lib/python3.11/dist-packages (from alembic>=1.5.0->optuna) (4.13.2)
    Collecting greenlet>=1 (from sqlalchemy>=1.4.2->optuna)
      Downloading greenlet-3.2.2-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (4.1 kB)
    Downloading optuna-4.3.0-py3-none-any.whl (386 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m386.6/386.6 kB[0m [31m8.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading alembic-1.16.1-py3-none-any.whl (242 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m242.5/242.5 kB[0m [31m17.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading sqlalchemy-2.0.41-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.3/3.3 MB[0m [31m59.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading colorlog-6.9.0-py3-none-any.whl (11 kB)
    Downloading greenlet-3.2.2-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (583 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m583.9/583.9 kB[0m [31m34.8 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: greenlet, colorlog, sqlalchemy, alembic, optuna
    Successfully installed alembic-1.16.1 colorlog-6.9.0 greenlet-3.2.2 optuna-4.3.0 sqlalchemy-2.0.41
    Collecting faiss-cpu
      Downloading faiss_cpu-1.11.0-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (4.8 kB)
    Requirement already satisfied: numpy<3.0,>=1.25.0 in /usr/local/lib/python3.11/dist-packages (from faiss-cpu) (2.2.6)
    Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from faiss-cpu) (25.0)
    Downloading faiss_cpu-1.11.0-cp311-cp311-manylinux_2_28_x86_64.whl (31.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m31.3/31.3 MB[0m [31m53.3 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: faiss-cpu
    Successfully installed faiss-cpu-1.11.0
    Collecting torch-geometric==2.4.0
      Downloading torch_geometric-2.4.0-py3-none-any.whl.metadata (63 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m63.9/63.9 kB[0m [31m1.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from torch-geometric==2.4.0) (4.67.1)
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
    Downloading torch_geometric-2.4.0-py3-none-any.whl (1.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m16.3 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: torch-geometric
    Successfully installed torch-geometric-2.4.0
    Looking in links: https://data.pyg.org/whl/torch-2.0.0+cpu.html
    Collecting torch-scatter
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_scatter-2.1.2%2Bpt20cpu-cp311-cp311-linux_x86_64.whl (494 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m494.0/494.0 kB[0m [31m6.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting torch-sparse
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_sparse-0.6.18%2Bpt20cpu-cp311-cp311-linux_x86_64.whl (1.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m28.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting torch-cluster
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_cluster-1.6.3%2Bpt20cpu-cp311-cp311-linux_x86_64.whl (750 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m750.9/750.9 kB[0m [31m36.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting torch-spline-conv
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_spline_conv-1.2.2%2Bpt20cpu-cp311-cp311-linux_x86_64.whl (208 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m208.1/208.1 kB[0m [31m13.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: scipy in /usr/local/lib/python3.11/dist-packages (from torch-sparse) (1.15.3)
    Requirement already satisfied: numpy<2.5,>=1.23.5 in /usr/local/lib/python3.11/dist-packages (from scipy->torch-sparse) (2.2.6)
    Installing collected packages: torch-spline-conv, torch-scatter, torch-sparse, torch-cluster
    Successfully installed torch-cluster-1.6.3+pt20cpu torch-scatter-2.1.2+pt20cpu torch-sparse-0.6.18+pt20cpu torch-spline-conv-1.2.2+pt20cpu
      Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m194.8/194.8 kB[0m [31m3.7 MB/s[0m eta [36m0:00:00[0m
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
    Requirement already satisfied: click in /usr/local/lib/python3.11/dist-packages (from deeponto) (8.2.0)
    Collecting dill (from deeponto)
      Downloading dill-0.4.0-py3-none-any.whl.metadata (10 kB)
    Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (from deeponto) (2.2.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from deeponto) (2.2.6)
    Requirement already satisfied: scikit_learn in /usr/local/lib/python3.11/dist-packages (from deeponto) (1.6.1)
    Requirement already satisfied: transformers[torch] in /usr/local/lib/python3.11/dist-packages (from deeponto) (4.51.3)
    Collecting datasets (from deeponto)
      Downloading datasets-3.6.0-py3-none-any.whl.metadata (19 kB)
    Requirement already satisfied: spacy in /usr/local/lib/python3.11/dist-packages (from deeponto) (3.8.5)
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
    Requirement already satisfied: huggingface-hub>=0.24.0 in /usr/local/lib/python3.11/dist-packages (from datasets->deeponto) (0.31.2)
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
    Requirement already satisfied: accelerate>=0.26.0 in /usr/local/lib/python3.11/dist-packages (from transformers[torch]->deeponto) (1.6.0)
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
    Requirement already satisfied: typing-inspection>=0.4.0 in /usr/local/lib/python3.11/dist-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy->deeponto) (0.4.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas->deeponto) (1.17.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests>=2.32.2->datasets->deeponto) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests>=2.32.2->datasets->deeponto) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests>=2.32.2->datasets->deeponto) (2.4.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests>=2.32.2->datasets->deeponto) (2025.4.26)
    Requirement already satisfied: blis<1.4.0,>=1.3.0 in /usr/local/lib/python3.11/dist-packages (from thinc<8.4.0,>=8.3.4->spacy->deeponto) (1.3.0)
    Requirement already satisfied: confection<1.0.0,>=0.0.1 in /usr/local/lib/python3.11/dist-packages (from thinc<8.4.0,>=8.3.4->spacy->deeponto) (0.1.5)
    Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.11/dist-packages (from typer<1.0.0,>=0.3.0->spacy->deeponto) (1.5.4)
    Requirement already satisfied: rich>=10.11.0 in /usr/local/lib/python3.11/dist-packages (from typer<1.0.0,>=0.3.0->spacy->deeponto) (14.0.0)
    Requirement already satisfied: cloudpathlib<1.0.0,>=0.7.0 in /usr/local/lib/python3.11/dist-packages (from weasel<0.5.0,>=0.1.0->spacy->deeponto) (0.21.0)
    Requirement already satisfied: smart-open<8.0.0,>=5.2.1 in /usr/local/lib/python3.11/dist-packages (from weasel<0.5.0,>=0.1.0->spacy->deeponto) (7.1.0)
    Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.11/dist-packages (from widgetsnbextension~=3.6.0->ipywidgets->deeponto) (6.5.7)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->spacy->deeponto) (3.0.2)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets->deeponto) (2.6.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets->deeponto) (1.3.2)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets->deeponto) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets->deeponto) (1.6.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets->deeponto) (6.4.3)
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
    Requirement already satisfied: prometheus-client in /usr/local/lib/python3.11/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.21.1)
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
    Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.24.0)
    Requirement already satisfied: jupyter-server<3,>=1.8 in /usr/local/lib/python3.11/dist-packages (from notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.16.0)
    Requirement already satisfied: cffi>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.17.1)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.11/dist-packages (from beautifulsoup4->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (2.7)
    Requirement already satisfied: pycparser in /usr/local/lib/python3.11/dist-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (2.22)
    Requirement already satisfied: anyio>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (4.9.0)
    Requirement already satisfied: websocket-client in /usr/local/lib/python3.11/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.8.0)
    Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.11/dist-packages (from anyio>=3.1.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.3.1)
    Downloading deeponto-0.9.3-py3-none-any.whl (89.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m89.7/89.7 MB[0m [31m13.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading anytree-2.13.0-py3-none-any.whl (45 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m45.1/45.1 kB[0m [31m3.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading datasets-3.6.0-py3-none-any.whl (491 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m491.5/491.5 kB[0m [31m32.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading dill-0.3.8-py3-none-any.whl (116 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m116.3/116.3 kB[0m [31m10.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading enlighten-1.14.1-py2.py3-none-any.whl (42 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m42.3/42.3 kB[0m [31m3.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading jpype1-1.5.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (494 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m494.1/494.1 kB[0m [31m33.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading lxml-5.4.0-cp311-cp311-manylinux_2_28_x86_64.whl (4.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.9/4.9 MB[0m [31m111.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pprintpp-0.4.0-py2.py3-none-any.whl (16 kB)
    Downloading rdflib-7.1.4-py3-none-any.whl (565 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m565.1/565.1 kB[0m [31m34.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading textdistance-4.6.3-py3-none-any.whl (31 kB)
    Downloading yacs-0.1.8-py3-none-any.whl (14 kB)
    Downloading blessed-1.21.0-py2.py3-none-any.whl (84 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m84.7/84.7 kB[0m [31m7.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading fsspec-2025.3.0-py3-none-any.whl (193 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m193.6/193.6 kB[0m [31m15.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading multiprocess-0.70.16-py311-none-any.whl (143 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m143.5/143.5 kB[0m [31m12.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading prefixed-0.9.0-py2.py3-none-any.whl (13 kB)
    Downloading jedi-0.19.2-py2.py3-none-any.whl (1.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m62.5 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: prefixed, pprintpp, yacs, textdistance, rdflib, lxml, JPype1, jedi, fsspec, dill, blessed, anytree, multiprocess, enlighten, datasets, deeponto
      Attempting uninstall: fsspec
        Found existing installation: fsspec 2025.5.0
        Uninstalling fsspec-2025.5.0:
          Successfully uninstalled fsspec-2025.5.0
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
src_ent = "ncit"

# Define the target ontology name
tgt_ent = "doid"

# Define the task name for this ontology matching process
task = "ncit2doid"

# Define the similarity threshold for validating matches
thres = 0.50
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

# Define the file path for the Source embeddings CSV file
# Embeddings for the source ontology entities are stored in this file.
src_Emb = f"{data_dir}/{src_ent}_BioBERT_emb.csv"

# Define the file path for the Target embeddings CSV file
# Embeddings for the target ontology entities are stored in this file.
tgt_Emb = f"{data_dir}/{tgt_ent}_BioBERT_emb.csv"

# Define the file path for the Source adjacency matrix
# This file represents the relationships (edges) between entities in the source ontology.
src_Adjacence = f"{data_dir}/{src_ent}_adjacence.csv"

# Define the file path for the Target adjacency matrix
# This file represents the relationships (edges) between entities in the target ontology.
tgt_Adjacence = f"{data_dir}/{tgt_ent}_adjacence.csv"

# Define the file path for the JSON file containing the Source ontology class labels
# This file maps the source ontology entities to their labels or names.
src_class = f"{data_dir}/{src_ent}_classes.json"

# Define the file path for the JSON file containing the Target ontology class labels
# This file maps the target ontology entities to their labels or names.
tgt_class = f"{data_dir}/{tgt_ent}_classes.json"

# Define the file path for the train data
train_file = f"{data_dir}/{task}_train.csv"

# Define the file path for the test data
# The test file contains reference mappings (ground truth) between the source and target ontologies.
test_file = f"{dataset_dir}/refs_equiv/test.tsv"

# Define the file path for the candidate mappings used during testing
# This file includes the candidate pairs (source and target entities) for ranking based metrics.
test_cands = f"{dataset_dir}/refs_equiv/test.cands.tsv"
cands_path = f"{data_dir}/{task}_cands.csv"

# Define the path where the prediction results will be saved in TSV format
# This file will store the final predictions (mappings) between source and target entities.
prediction_path = f"{results_dir}/{task}_matching_results_BioBERT.tsv"

# Define the path where all prediction results will be saved in TSV format
# This file will store detailed prediction results, including all candidate scores.
all_predictions_path = f"{results_dir}/{task}_all_predictions_BioBERT.tsv"

# Define the path where formatted ranking predictions will be saved in TSV format
# This file will contain predictions formatted for evaluation using ranking-based metrics.
formatted_predictions_path = f"{results_dir}/{task}_formatted_predictions_BioBERT.tsv"
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
# Import required libraries
import torch
import torch.nn as nn
import faiss
import numpy as np

class GatedCombinationWithFaiss(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Linear layers to compute gating values for the source and target embeddings
        self.gate_A_fc = nn.Linear(input_dim, input_dim)
        self.gate_B_fc = nn.Linear(input_dim, input_dim)

        # Final linear layer to map similarity score to prediction (sigmoid output)
        self.fc = nn.Linear(1, 1)

    def faiss_l2(self, a, b):
        """
        Compute L2 distances using FAISS (non-differentiable).
        This function converts tensors to NumPy, builds a FAISS index, and performs a search.
        Only use this during inference or evaluation — not for training.

        Args:
            a (Tensor): Query vectors (batch_size x dim)
            b (Tensor): Database vectors (batch_size x dim)

        Returns:
            Tensor: L2 distances between aligned rows (one-to-one)
        """
        # Detach tensors from the computation graph and move to CPU
        a_np = a.detach().cpu().numpy().astype(np.float32)
        b_np = b.detach().cpu().numpy().astype(np.float32)

        # Create a FAISS index for L2 distance
        index = faiss.IndexFlatL2(a_np.shape[1])
        index.add(b_np)

        # Perform 1-NN search
        distances, _ = index.search(a_np, 1)  # shape: (batch_size, 1)

        # Convert back to PyTorch tensor on the original device
        return torch.tensor(distances[:, 0], dtype=torch.float32, device=a.device)

    def forward(self, x1, x2, x3, x4, return_embeddings=False):
        """
        Forward pass through the gated combination model.
        Combines original and transformed embeddings using learned gates.

        Args:
            x1, x2: original and GNN-transformed embeddings for source entities
            x3, x4: original and GNN-transformed embeddings for target entities
            return_embeddings (bool): if True, return gated embeddings instead of prediction

        Returns:
            Tensor: similarity score (if return_embeddings=False)
            OR
            Tuple[Tensor, Tensor]: gated source and target embeddings (if return_embeddings=True)
        """

        # Compute gate for source embeddings
        gate_values1 = torch.sigmoid(self.gate_A_fc(x1))
        a = x1 * gate_values1 + x2 * (1 - gate_values1)

        # Compute gate for target embeddings
        gate_values2 = torch.sigmoid(self.gate_B_fc(x3))
        b = x3 * gate_values2 + x4 * (1 - gate_values2)

        if return_embeddings:
            return a, b

        # Compute non-differentiable distance with FAISS (1-to-1)
        distance = self.faiss_l2(a, b)

        # Pass through a sigmoid layer for binary classification output
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
def compute_mrr_and_hits(reference_file, predicted_file, output_file, k_values=[1, 5, 10]):
    """
    Compute Mean Reciprocal Rank (MRR) and Hits@k metrics for ontology matching results.

    Args:
        reference_file (str): Path to the reference test candidate file (usually 'test.cands.tsv').
        predicted_file (str): Path to the prediction results (with columns: SrcEntity, TgtEntity, Score).
        output_file (str): Path to save ranked candidate predictions with scores.
        k_values (list): List of integers specifying which Hits@k metrics to compute.

    Returns:
        dict: A dictionary with MRR and Hits@k scores.
    """

    # Load reference candidate mappings: each row = (SrcEntity, CorrectTgtEntity, [CandidateTgtEntities])
    test_candidate_mappings = read_table(reference_file).values.tolist()

    # Load predictions and ensure Score is float
    predicted_data = pd.read_csv(predicted_file, sep="\t")
    predicted_data["Score"] = predicted_data["Score"].apply(
        lambda x: float(x.strip("[]")) if isinstance(x, str) else float(x)
    )

    # Create a dictionary mapping (SrcEntity, TgtEntity) -> predicted score
    score_lookup = {
        (row["SrcEntity"], row["TgtEntity"]): row["Score"]
        for _, row in predicted_data.iterrows()
    }

    ranking_results = []

    # Rank the candidates for each source entity
    for src_ref_class, tgt_ref_class, tgt_cands in test_candidate_mappings:
        # Safely parse the candidate list (tgt_cands is a stringified list)
        try:
            tgt_cands = eval(tgt_cands)
        except Exception:
            tgt_cands = []

        # Score each candidate (use a large negative default if not found)
        scored_cands = [
            (tgt_cand, score_lookup.get((src_ref_class, tgt_cand), -1e9))
            for tgt_cand in tgt_cands
        ]

        # Sort candidates by score descending
        scored_cands = sorted(scored_cands, key=lambda x: x[1], reverse=True)

        # Store the ranking result
        ranking_results.append((src_ref_class, tgt_ref_class, scored_cands))

    # Save ranked predictions for inspection/debugging
    pd.DataFrame(ranking_results, columns=["SrcEntity", "TgtEntity", "TgtCandidates"]).to_csv(
        output_file, sep="\t", index=False
    )

    # === Evaluation: compute MRR and Hits@k ===
    total_entities = len(ranking_results)
    reciprocal_ranks = []
    hits_at_k = {k: 0 for k in k_values}

    for src_entity, tgt_ref_class, tgt_cands in ranking_results:
        ranked_candidates = [cand[0] for cand in tgt_cands]  # candidate URIs only
        if tgt_ref_class in ranked_candidates:
            rank = ranked_candidates.index(tgt_ref_class) + 1
            reciprocal_ranks.append(1 / rank)
            for k in k_values:
                if rank <= k:
                    hits_at_k[k] += 1
        else:
            reciprocal_ranks.append(0)  # No correct match in candidate list

    # Compute final metrics
    mrr = sum(reciprocal_ranks) / total_entities
    hits_at_k = {k: hits / total_entities for k, hits in hits_at_k.items()}

    return {"MRR": mrr, "Hits@k": hits_at_k}
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
    df_src = pd.read_csv(src_emb_path, sep='\t')
    df_tgt = pd.read_csv(tgt_emb_path, sep='\t')
    uris_src = df_src["Concept"].values
    uris_tgt = df_tgt["Concept"].values
    src_vecs = df_src.drop(columns=["Concept"]).values.astype('float32')
    tgt_vecs = df_tgt.drop(columns=["Concept"]).values.astype('float32')
    return uris_src, uris_tgt, src_vecs, tgt_vecs

def save_results(uris_src, uris_tgt, indices, scores, output_file, top_k):
    rows = []
    for i, (ind_row, score_row) in enumerate(zip(indices, scores)):
        src_uri = uris_src[i]
        for j, tgt_idx in enumerate(ind_row):
            tgt_uri = uris_tgt[tgt_idx]
            score = score_row[j]
            rows.append((src_uri, tgt_uri, score))
    df_result = pd.DataFrame(rows, columns=["SrcEntity", "TgtEntity", "Score"])
    df_result.to_csv(output_file, sep='\t', index=False)
    print(f"Top-{top_k} FAISS similarity results saved to: {output_file}")

def topk_faiss_l2(src_emb_path, tgt_emb_path, top_k=15, output_file="topk_l2.tsv"):
    print("🔹 Using L2 (Euclidean) distance with FAISS")
    start = time.time()

    uris_src, uris_tgt, src_vecs, tgt_vecs = load_embeddings(src_emb_path, tgt_emb_path)
    dim = src_vecs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(tgt_vecs)
    distances, indices = index.search(src_vecs, top_k)
    similarity_scores = 1 / (1 + distances)

    save_results(uris_src, uris_tgt, indices, similarity_scores, output_file, top_k)

    print(f"⏱️ Execution time: {time.time() - start:.2f} seconds")

```

# **Mappings Evaluation Functions**

# **Precision, Recall, F1**

### Evaluation Strategy and Filtering Justification

### Filtering Justification

In the `evaluate_predictions` function, two important filtering steps are applied to ensure that the evaluation metrics (such as Precision, Recall, and F1-score) accurately reflect the model's performance:


#### 1. Filtering Out Training-Only Entities

We remove all predicted mappings involving source or target entities that are present **only in the training set** and not in the test set.

This step is critical because:

- In some datasets like **Bio-ML**, the same entity can appear in both training and test sets, although with **different correspondences**.
- If we don't remove training-only entities, it can lead to **label leakage** and **metric distortion**.

#### 2. Filtering on `SrcEntity` present in the test set

The second step keeps only the predictions where the `SrcEntity` is included in the test reference set.

- This eliminates **non-evaluable false positives**, i.e., predicted mappings for source entities that do not appear in the test set and therefore have no ground-truth correspondences. Including such predictions **unfairly penalizes precision and F1-score**, even though they are technically not verifiable errors.

- It focuses the evaluation on entities with defined ground-truth mappings, which is critical for computing metrics such as :

$P_{\text{test}} = \frac{|\mathcal{M}_{\text{out}} \cap \mathcal{M}_{\text{test}}|}{|\mathcal{M}_{\text{out}} \setminus (\mathcal{M}_{\text{ref}} \setminus \mathcal{M}_{\text{test}})|}$.

---



```python
import pandas as pd
from deeponto.align.mapping import EntityMapping, ReferenceMapping
from deeponto.align.evaluation import AlignmentEvaluator

def evaluate_predictions(
    topk_file,
    train_file,
    test_file,
    src_onto,
    tgt_onto,
    threshold=0.0
):
    # === Step 1: Load input files ===
    df = pd.read_csv(topk_file, sep='\t', dtype=str)
    train_df = pd.read_csv(train_file, sep="\t", dtype=str)
    test_df = pd.read_csv(test_file, sep="\t", dtype=str)

    # === Step 2: Remove URIs only present in training set ===
    train_uris = set(train_df['SrcEntity']) | set(train_df['TgtEntity'])
    test_uris = set(test_df['SrcEntity']) | set(test_df['TgtEntity'])
    uris_to_exclude = train_uris - test_uris
    df = df[~(df['SrcEntity'].isin(uris_to_exclude) | df['TgtEntity'].isin(uris_to_exclude))].reset_index(drop=True)

    # === Step 3: Keep only source entities from the test set ===
    src_entities_test = set(test_df['SrcEntity'])
    df = df[df['SrcEntity'].isin(src_entities_test)].reset_index(drop=True)

    # === Step 4: Save filtered Top-K predictions ===
    output_file1 = topk_file.replace(".tsv", "_filtered.tsv")
    df.to_csv(output_file1, sep='\t', index=False)

    # === Step 5: Convert score column to float and sort by descending score
    df['Score'] = df['Score'].apply(lambda x: float(x.strip("[]")) if isinstance(x, str) else float(x))
    df_sorted = df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    # === Step 6: Apply greedy 1-1 matching constraint with score threshold
    matched_sources = set()
    matched_targets = set()
    result = []
    for _, row in df_sorted.iterrows():
        src, tgt, score = row['SrcEntity'], row['TgtEntity'], row['Score']
        if src not in matched_sources and tgt not in matched_targets and score >= threshold:
            result.append((src, tgt, score))
            matched_sources.add(src)
            matched_targets.add(tgt)

    # === Step 7: Save final Top-1 predictions ===
    matching_results_df = pd.DataFrame(result, columns=['SrcEntity', 'TgtEntity', 'Score'])
    output_file2 = topk_file.replace(".tsv", f"_predictions.tsv")
    matching_results_df.to_csv(output_file2, sep='\t', index=False)

    print(f"   ➤ Mappings file:   {output_file2}")

    # === Step 8: Evaluate against reference mappings
    preds = EntityMapping.read_table_mappings(output_file2)
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

    return output_file2, results, correct
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

    Epoch [10/1000], Training Loss: 0.002684629987925291
    Epoch [20/1000], Training Loss: 0.002067181281745434
    Epoch [30/1000], Training Loss: 0.001738561550155282
    Epoch [40/1000], Training Loss: 0.0015298259677365422
    Epoch [50/1000], Training Loss: 0.0013840033207088709
    Epoch [60/1000], Training Loss: 0.001271839253604412
    Epoch [70/1000], Training Loss: 0.0011799265630543232
    Epoch [80/1000], Training Loss: 0.0011025272542610765
    Epoch [90/1000], Training Loss: 0.001035877619870007
    Epoch [100/1000], Training Loss: 0.0009779180400073528
    Epoch [110/1000], Training Loss: 0.0009266994893550873
    Epoch [120/1000], Training Loss: 0.0008811839506961405
    Epoch [130/1000], Training Loss: 0.0008406019769608974
    Epoch [140/1000], Training Loss: 0.0008042737026698887
    Epoch [150/1000], Training Loss: 0.0007711953949183226
    Epoch [160/1000], Training Loss: 0.0007409487152472138
    Epoch [170/1000], Training Loss: 0.0007135536870919168
    Epoch [180/1000], Training Loss: 0.0006884446484036744
    Epoch [190/1000], Training Loss: 0.0006654532044194639
    Epoch [200/1000], Training Loss: 0.0006441115983761847
    Epoch [210/1000], Training Loss: 0.0006245742551982403
    Epoch [220/1000], Training Loss: 0.0006064666667953134
    Epoch [230/1000], Training Loss: 0.0005896376096643507
    Epoch [240/1000], Training Loss: 0.0005739019252359867
    Epoch [250/1000], Training Loss: 0.0005592892412096262
    Epoch [260/1000], Training Loss: 0.0005456888466142118
    Epoch [270/1000], Training Loss: 0.0005328321130946279
    Epoch [280/1000], Training Loss: 0.0005208259099163115
    Epoch [290/1000], Training Loss: 0.0005094569642096758
    Epoch [300/1000], Training Loss: 0.0004988422151654959
    Epoch [310/1000], Training Loss: 0.0004889350384473801
    Epoch [320/1000], Training Loss: 0.00047968581202439964
    Epoch [330/1000], Training Loss: 0.000471043458674103
    Epoch [340/1000], Training Loss: 0.0004629384493455291
    Epoch [350/1000], Training Loss: 0.00045529502676799893
    Epoch [360/1000], Training Loss: 0.0004480613861232996
    Epoch [370/1000], Training Loss: 0.0004411337140481919
    Epoch [380/1000], Training Loss: 0.00043449067743495107
    Epoch [390/1000], Training Loss: 0.0004280460416339338
    Epoch [400/1000], Training Loss: 0.0004217299574520439
    Epoch [410/1000], Training Loss: 0.00041557010263204575
    Epoch [420/1000], Training Loss: 0.0004096103657502681
    Epoch [430/1000], Training Loss: 0.0004038472834508866
    Epoch [440/1000], Training Loss: 0.0003981485206168145
    Epoch [450/1000], Training Loss: 0.00039292729343287647
    Epoch [460/1000], Training Loss: 0.0003890166699420661
    Epoch [470/1000], Training Loss: 0.0003840193385258317
    Epoch [480/1000], Training Loss: 0.00037922311457805336
    Epoch [490/1000], Training Loss: 0.00037438218714669347
    Epoch [500/1000], Training Loss: 0.0003725844726432115
    Epoch [510/1000], Training Loss: 0.0003674498584587127
    Epoch [520/1000], Training Loss: 0.00036362247192300856
    Epoch [530/1000], Training Loss: 0.00036008929600939155
    Epoch [540/1000], Training Loss: 0.00035577957169152796
    Epoch [550/1000], Training Loss: 0.00035216909600421786
    Epoch [560/1000], Training Loss: 0.0003496955905575305
    Epoch [570/1000], Training Loss: 0.0003469448711257428
    Epoch [580/1000], Training Loss: 0.00034226395655423403
    Epoch [590/1000], Training Loss: 0.00033820525277405977
    Epoch [600/1000], Training Loss: 0.00033690265263430774
    Epoch [610/1000], Training Loss: 0.00033366994466632605
    Epoch [620/1000], Training Loss: 0.0003291306202299893
    Epoch [630/1000], Training Loss: 0.0003249499131925404
    Epoch [640/1000], Training Loss: 0.0003231216687709093
    Epoch [650/1000], Training Loss: 0.00032137183006852865
    Epoch [660/1000], Training Loss: 0.0003167146351188421
    Epoch [670/1000], Training Loss: 0.0003125527291558683
    Epoch [680/1000], Training Loss: 0.00031088932882994413
    Epoch [690/1000], Training Loss: 0.00030978862196207047
    Epoch [700/1000], Training Loss: 0.0003053078835364431
    Epoch [710/1000], Training Loss: 0.00030155404238030314
    Epoch [720/1000], Training Loss: 0.00030047757900319993
    Epoch [730/1000], Training Loss: 0.00029968537273816764
    Epoch [740/1000], Training Loss: 0.00029560760594904423
    Epoch [750/1000], Training Loss: 0.00029243886820040643
    Epoch [760/1000], Training Loss: 0.00029200632707215846
    Epoch [770/1000], Training Loss: 0.0002912233758252114
    Epoch [780/1000], Training Loss: 0.0002873375196941197
    Epoch [790/1000], Training Loss: 0.0002847195428330451
    Epoch [800/1000], Training Loss: 0.0002856355276890099
    Epoch [810/1000], Training Loss: 0.00028303122962825
    Epoch [820/1000], Training Loss: 0.00027942314045503736
    Epoch [830/1000], Training Loss: 0.00027905957540497184
    Epoch [840/1000], Training Loss: 0.0002785043034236878
    Epoch [850/1000], Training Loss: 0.0002746581449173391
    Epoch [860/1000], Training Loss: 0.00027371817850507796
    Epoch [870/1000], Training Loss: 0.000274040357908234
    Epoch [880/1000], Training Loss: 0.0002700167242437601
    Epoch [890/1000], Training Loss: 0.000267859228188172
    Epoch [900/1000], Training Loss: 0.0002694272843655199
    Epoch [910/1000], Training Loss: 0.00026661905576474965
    Epoch [920/1000], Training Loss: 0.000263075198745355
    Epoch [930/1000], Training Loss: 0.00026327522937208414
    Epoch [940/1000], Training Loss: 0.00026303346385248005
    Epoch [950/1000], Training Loss: 0.00025905208894982934
    Epoch [960/1000], Training Loss: 0.0002571341465227306
    Epoch [970/1000], Training Loss: 0.00025820601149462163
    Epoch [980/1000], Training Loss: 0.000255648948950693
    Epoch [990/1000], Training Loss: 0.0002519671688787639
    Epoch [1000/1000], Training Loss: 0.00025207188446074724



    
![png](output_50_1.png)
    


    Training complete! Total training time: 617.69 seconds


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
    model = GatedCombinationWithFaiss(X1_t.shape[1]).to(device)
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

    Epoch [1/100] Training Loss: 3.5693, F1 Score: 0.0594 | Validation Loss: 0.0570, F1 Score: 0.3042
    Epoch [2/100] Training Loss: 0.0526, F1 Score: 0.2972 | Validation Loss: 0.0532, F1 Score: 0.2890
    Epoch [3/100] Training Loss: 0.0511, F1 Score: 0.2979 | Validation Loss: 0.0528, F1 Score: 0.3010
    Epoch [4/100] Training Loss: 0.0509, F1 Score: 0.3196 | Validation Loss: 0.0521, F1 Score: 0.3227
    Epoch [5/100] Training Loss: 0.0500, F1 Score: 0.3414 | Validation Loss: 0.0512, F1 Score: 0.3469
    Epoch [6/100] Training Loss: 0.0484, F1 Score: 0.3779 | Validation Loss: 0.0504, F1 Score: 0.3893
    Epoch [7/100] Training Loss: 0.0478, F1 Score: 0.3994 | Validation Loss: 0.0494, F1 Score: 0.3972
    Epoch [8/100] Training Loss: 0.0472, F1 Score: 0.4187 | Validation Loss: 0.0489, F1 Score: 0.4139
    Epoch [9/100] Training Loss: 0.0474, F1 Score: 0.4384 | Validation Loss: 0.0487, F1 Score: 0.4241
    Epoch [10/100] Training Loss: 0.0469, F1 Score: 0.4510 | Validation Loss: 0.0483, F1 Score: 0.4354
    Epoch [11/100] Training Loss: 0.0466, F1 Score: 0.4674 | Validation Loss: 0.0481, F1 Score: 0.4433
    Epoch [12/100] Training Loss: 0.0463, F1 Score: 0.4790 | Validation Loss: 0.0483, F1 Score: 0.4415
    Epoch [13/100] Training Loss: 0.0469, F1 Score: 0.4799 | Validation Loss: 0.0480, F1 Score: 0.4610
    Epoch [14/100] Training Loss: 0.0461, F1 Score: 0.4924 | Validation Loss: 0.0479, F1 Score: 0.4525
    Epoch [15/100] Training Loss: 0.0468, F1 Score: 0.4911 | Validation Loss: 0.0477, F1 Score: 0.4645
    Epoch [16/100] Training Loss: 0.0468, F1 Score: 0.4962 | Validation Loss: 0.0477, F1 Score: 0.4645
    Epoch [17/100] Training Loss: 0.0462, F1 Score: 0.4908 | Validation Loss: 0.0477, F1 Score: 0.4628
    Epoch [18/100] Training Loss: 0.0459, F1 Score: 0.5037 | Validation Loss: 0.0477, F1 Score: 0.4645
    Epoch [19/100] Training Loss: 0.0464, F1 Score: 0.5034 | Validation Loss: 0.0476, F1 Score: 0.4689
    Epoch [20/100] Training Loss: 0.0464, F1 Score: 0.5064 | Validation Loss: 0.0476, F1 Score: 0.4689
    Epoch [21/100] Training Loss: 0.0460, F1 Score: 0.5054 | Validation Loss: 0.0479, F1 Score: 0.4585
    Epoch [22/100] Training Loss: 0.0473, F1 Score: 0.4973 | Validation Loss: 0.0477, F1 Score: 0.4640
    Epoch [23/100] Training Loss: 0.0459, F1 Score: 0.5125 | Validation Loss: 0.0476, F1 Score: 0.4689
    Epoch [24/100] Training Loss: 0.0457, F1 Score: 0.5152 | Validation Loss: 0.0477, F1 Score: 0.5039
    Epoch [25/100] Training Loss: 0.0465, F1 Score: 0.5141 | Validation Loss: 0.0476, F1 Score: 0.4803
    Epoch [26/100] Training Loss: 0.0458, F1 Score: 0.5129 | Validation Loss: 0.0475, F1 Score: 0.4803
    Epoch [27/100] Training Loss: 0.0463, F1 Score: 0.5074 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [28/100] Training Loss: 0.0461, F1 Score: 0.5111 | Validation Loss: 0.0477, F1 Score: 0.4689
    Epoch [29/100] Training Loss: 0.0461, F1 Score: 0.5097 | Validation Loss: 0.0481, F1 Score: 0.5174
    Epoch [30/100] Training Loss: 0.0460, F1 Score: 0.5203 | Validation Loss: 0.0480, F1 Score: 0.5174
    Epoch [31/100] Training Loss: 0.0468, F1 Score: 0.5033 | Validation Loss: 0.0476, F1 Score: 0.4738
    Epoch [32/100] Training Loss: 0.0469, F1 Score: 0.5050 | Validation Loss: 0.0476, F1 Score: 0.4803
    Epoch [33/100] Training Loss: 0.0466, F1 Score: 0.5110 | Validation Loss: 0.0477, F1 Score: 0.4665
    Epoch [34/100] Training Loss: 0.0461, F1 Score: 0.5060 | Validation Loss: 0.0476, F1 Score: 0.4738
    Epoch [35/100] Training Loss: 0.0458, F1 Score: 0.5135 | Validation Loss: 0.0475, F1 Score: 0.4890
    Epoch [36/100] Training Loss: 0.0457, F1 Score: 0.5157 | Validation Loss: 0.0476, F1 Score: 0.4803
    Epoch [37/100] Training Loss: 0.0455, F1 Score: 0.5138 | Validation Loss: 0.0475, F1 Score: 0.4890
    Epoch [38/100] Training Loss: 0.0468, F1 Score: 0.5069 | Validation Loss: 0.0476, F1 Score: 0.4803
    Epoch [39/100] Training Loss: 0.0459, F1 Score: 0.5178 | Validation Loss: 0.0476, F1 Score: 0.4803
    Epoch [40/100] Training Loss: 0.0455, F1 Score: 0.5197 | Validation Loss: 0.0475, F1 Score: 0.4890
    Epoch [41/100] Training Loss: 0.0464, F1 Score: 0.5151 | Validation Loss: 0.0476, F1 Score: 0.4803
    Epoch [42/100] Training Loss: 0.0466, F1 Score: 0.5087 | Validation Loss: 0.0475, F1 Score: 0.4890
    Epoch [43/100] Training Loss: 0.0459, F1 Score: 0.5175 | Validation Loss: 0.0475, F1 Score: 0.4803
    Epoch [44/100] Training Loss: 0.0458, F1 Score: 0.5154 | Validation Loss: 0.0475, F1 Score: 0.4890
    Epoch [45/100] Training Loss: 0.0458, F1 Score: 0.5181 | Validation Loss: 0.0475, F1 Score: 0.4890
    Epoch [46/100] Training Loss: 0.0461, F1 Score: 0.5154 | Validation Loss: 0.0476, F1 Score: 0.4803
    Epoch [47/100] Training Loss: 0.0458, F1 Score: 0.5165 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [48/100] Training Loss: 0.0464, F1 Score: 0.5113 | Validation Loss: 0.0475, F1 Score: 0.4826
    Epoch [49/100] Training Loss: 0.0462, F1 Score: 0.5136 | Validation Loss: 0.0475, F1 Score: 0.4867
    Epoch [50/100] Training Loss: 0.0455, F1 Score: 0.5157 | Validation Loss: 0.0475, F1 Score: 0.4850
    Epoch [51/100] Training Loss: 0.0457, F1 Score: 0.5221 | Validation Loss: 0.0475, F1 Score: 0.4867
    Epoch [52/100] Training Loss: 0.0460, F1 Score: 0.5116 | Validation Loss: 0.0475, F1 Score: 0.4906
    Epoch [53/100] Training Loss: 0.0463, F1 Score: 0.5109 | Validation Loss: 0.0476, F1 Score: 0.4803
    Epoch [54/100] Training Loss: 0.0455, F1 Score: 0.5245 | Validation Loss: 0.0475, F1 Score: 0.4867
    Epoch [55/100] Training Loss: 0.0462, F1 Score: 0.5175 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [56/100] Training Loss: 0.0456, F1 Score: 0.5126 | Validation Loss: 0.0475, F1 Score: 0.4906
    Epoch [57/100] Training Loss: 0.0454, F1 Score: 0.5210 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [58/100] Training Loss: 0.0457, F1 Score: 0.5209 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [59/100] Training Loss: 0.0456, F1 Score: 0.5202 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [60/100] Training Loss: 0.0455, F1 Score: 0.5230 | Validation Loss: 0.0475, F1 Score: 0.4826
    Epoch [61/100] Training Loss: 0.0462, F1 Score: 0.5154 | Validation Loss: 0.0475, F1 Score: 0.4890
    Epoch [62/100] Training Loss: 0.0459, F1 Score: 0.5151 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [63/100] Training Loss: 0.0460, F1 Score: 0.5133 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [64/100] Training Loss: 0.0467, F1 Score: 0.5118 | Validation Loss: 0.0475, F1 Score: 0.4890
    Epoch [65/100] Training Loss: 0.0462, F1 Score: 0.5126 | Validation Loss: 0.0475, F1 Score: 0.4906
    Epoch [66/100] Training Loss: 0.0459, F1 Score: 0.5113 | Validation Loss: 0.0476, F1 Score: 0.4803
    Epoch [67/100] Training Loss: 0.0455, F1 Score: 0.5206 | Validation Loss: 0.0475, F1 Score: 0.4890
    Epoch [68/100] Training Loss: 0.0451, F1 Score: 0.5255 | Validation Loss: 0.0475, F1 Score: 0.4826
    Epoch [69/100] Training Loss: 0.0459, F1 Score: 0.5164 | Validation Loss: 0.0475, F1 Score: 0.4890
    Epoch [70/100] Training Loss: 0.0461, F1 Score: 0.5133 | Validation Loss: 0.0476, F1 Score: 0.4803
    Epoch [71/100] Training Loss: 0.0458, F1 Score: 0.5220 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [72/100] Training Loss: 0.0458, F1 Score: 0.5240 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [73/100] Training Loss: 0.0464, F1 Score: 0.5146 | Validation Loss: 0.0475, F1 Score: 0.4890
    Epoch [74/100] Training Loss: 0.0454, F1 Score: 0.5195 | Validation Loss: 0.0475, F1 Score: 0.4906
    Epoch [75/100] Training Loss: 0.0455, F1 Score: 0.5212 | Validation Loss: 0.0475, F1 Score: 0.4906
    Epoch [76/100] Training Loss: 0.0457, F1 Score: 0.5219 | Validation Loss: 0.0475, F1 Score: 0.4906
    Epoch [77/100] Training Loss: 0.0453, F1 Score: 0.5213 | Validation Loss: 0.0475, F1 Score: 0.4906
    Epoch [78/100] Training Loss: 0.0468, F1 Score: 0.5082 | Validation Loss: 0.0475, F1 Score: 0.4906
    Epoch [79/100] Training Loss: 0.0462, F1 Score: 0.5167 | Validation Loss: 0.0475, F1 Score: 0.4906
    Epoch [80/100] Training Loss: 0.0457, F1 Score: 0.5188 | Validation Loss: 0.0475, F1 Score: 0.4906
    Epoch [81/100] Training Loss: 0.0463, F1 Score: 0.5160 | Validation Loss: 0.0475, F1 Score: 0.4906
    Epoch [82/100] Training Loss: 0.0461, F1 Score: 0.5171 | Validation Loss: 0.0475, F1 Score: 0.4906
    Epoch [83/100] Training Loss: 0.0454, F1 Score: 0.5241 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [84/100] Training Loss: 0.0457, F1 Score: 0.5171 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [85/100] Training Loss: 0.0462, F1 Score: 0.5188 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [86/100] Training Loss: 0.0457, F1 Score: 0.5188 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [87/100] Training Loss: 0.0458, F1 Score: 0.5185 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [88/100] Training Loss: 0.0464, F1 Score: 0.5126 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [89/100] Training Loss: 0.0458, F1 Score: 0.5191 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [90/100] Training Loss: 0.0457, F1 Score: 0.5216 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [91/100] Training Loss: 0.0456, F1 Score: 0.5154 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [92/100] Training Loss: 0.0467, F1 Score: 0.5072 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [93/100] Training Loss: 0.0460, F1 Score: 0.5150 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [94/100] Training Loss: 0.0456, F1 Score: 0.5223 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [95/100] Training Loss: 0.0464, F1 Score: 0.5116 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [96/100] Training Loss: 0.0461, F1 Score: 0.5160 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [97/100] Training Loss: 0.0464, F1 Score: 0.5123 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [98/100] Training Loss: 0.0455, F1 Score: 0.5178 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [99/100] Training Loss: 0.0456, F1 Score: 0.5192 | Validation Loss: 0.0475, F1 Score: 0.4930
    Epoch [100/100] Training Loss: 0.0461, F1 Score: 0.5167 | Validation Loss: 0.0475, F1 Score: 0.4930



    
![png](output_57_1.png)
    


    Training complete! Total time: 541.56 seconds


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
output_file_src = f"{data_dir}/{src_ent}_final_embeddings_BioBERT.tsv"
output_file_tgt = f"{data_dir}/{tgt_ent}_final_embeddings_BioBERT.tsv"

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
    - Source: /content/gdrive/My Drive/BioGITOM-VLDB//ncit2doid/Data/ncit_final_embeddings_BioBERT.tsv
    - Target: /content/gdrive/My Drive/BioGITOM-VLDB//ncit2doid/Data/doid_final_embeddings_BioBERT.tsv
    ⏱️ Execution time: 22.94 seconds


# **Filter No Used Concepts**





```python
# Call the function to filter out ignored concepts (e.g., owl:Thing, deprecated, etc.)
# from the source and target ontology embeddings.

# Input:
# - src_emb_path: Path to the TSV file containing embeddings for the source ontology
# - tgt_emb_path: Path to the TSV file containing embeddings for the target ontology
# - src_onto / tgt_onto: DeepOnto ontology objects used to identify ignored concepts

# Output:
# - src_file: Path to the cleaned source embeddings (with ignored concepts removed)
# - tgt_file: Path to the cleaned target embeddings (with ignored concepts removed)

src_file, tgt_file = filter_ignored_class(
    src_emb_path=f"{data_dir}/{src_ent}_final_embeddings_BioBERT.tsv",
    tgt_emb_path=f"{data_dir}/{tgt_ent}_final_embeddings_BioBERT.tsv",
    src_onto=src_onto,
    tgt_onto=tgt_onto

)

```

    🔍 Initial source file: 15992 rows
    🔍 Initial target file: 8480 rows
    ✅ Source after removing ignored classes: 7065 rows
    ✅ Target after removing ignored classes: 8463 rows
    📁 Cleaned source file saved to: /content/gdrive/My Drive/BioGITOM-VLDB//ncit2doid/Data/ncit_final_embeddings_BioBERT_cleaned.tsv
    📁 Cleaned target file saved to: /content/gdrive/My Drive/BioGITOM-VLDB//ncit2doid/Data/doid_final_embeddings_BioBERT_cleaned.tsv


# **Mappings Generation**

# **Using faiss l2**


```python
# Compute the top-10 most similar mappings using l2 distance
# between ResMLP-encoded embeddings of the source and target ontologies.
# The input embeddings were previously encoded using the ResMLPEncoder,
# and the similarity score is computed as the inverse of the l2 distance.
# Results are saved in a TSV file with columns: SrcEntity, TgtEntity, Score.
topk_faiss_l2(
    src_emb_path=f"{data_dir}/{src_ent}_final_embeddings_BioBERT_cleaned.tsv",
    tgt_emb_path=f"{data_dir}/{tgt_ent}_final_embeddings_BioBERT_cleaned.tsv",
    top_k=10,
    output_file=f"{results_dir}/{task}_top_10_mappings_BioBERT.tsv"
)
```

    🔹 Using L2 (Euclidean) distance with FAISS
    Top-10 FAISS similarity results saved to: /content/gdrive/My Drive/BioGITOM-VLDB//ncit2doid/Results/ncit2doid_top_10_mappings_BioBERT.tsv
    ⏱️ Execution time: 9.15 seconds


# **Evaluation**

# **Global Metrics: Precision, Recall and F1 score**


```python
# Run the evaluation on the predicted mappings using a filtering and evaluation function.

output_file, metrics, correct = evaluate_predictions(
    topk_file=f"{results_dir}/{task}_top_10_mappings_BioBERT.tsv",
    # Path to the TSV file containing predicted mappings with scores (before filtering).

    train_file=f"{dataset_dir}/refs_equiv/train.tsv",
    # Path to the training reference file (used to exclude mappings involving train-only entities).

    test_file=f"{dataset_dir}/refs_equiv/test.tsv",
    # Path to the test reference file (used as the gold standard for evaluation).

    src_onto=src_onto,
    # The source ontology object, used to detect ignored classes or perform additional filtering.

    tgt_onto=tgt_onto,
    # The target ontology object, used similarly for filtering ignored or irrelevant classes.
)

# This function returns:
# - `output_file`: the path to the filtered and evaluated output file.
# - `metrics`: a tuple containing (Precision, Recall, F1-score).
# - `correct`: the number of correctly predicted mappings found in the gold standard.

```

       ➤ Mappings file:   /content/gdrive/My Drive/BioGITOM-VLDB//ncit2doid/Results/ncit2doid_top_10_mappings_BioBERT_predictions.tsv
    
    🎯 Evaluation Summary:
       - Correct mappings:     2462
       - Total predictions:    3060
       - Total references:     3280
    📊 Precision:              0.805
    📊 Recall:                 0.751
    📊 F1-score:               0.777
    


# **Metrics@1**


```python
# Compute the top-1 most similar mappings using l2 distance
# and the similarity score is computed as the inverse of the l2 distance.
# Results are saved in a TSV file with columns: SrcEntity, TgtEntity, Score.
topk_faiss_l2(
    src_emb_path=f"{data_dir}/{src_ent}_final_embeddings_BioBERT_cleaned.tsv",
    tgt_emb_path=f"{data_dir}/{tgt_ent}_final_embeddings_BioBERT_cleaned.tsv",
    top_k=1,
    output_file=f"{results_dir}/{task}_top_1_mappings_BioBERT.tsv"
)
```

    🔹 Using L2 (Euclidean) distance with FAISS
    Top-1 FAISS similarity results saved to: /content/gdrive/My Drive/BioGITOM-VLDB//ncit2doid/Results/ncit2doid_top_1_mappings_BioBERT.tsv
    ⏱️ Execution time: 4.89 seconds



```python
# === Evaluate Top-1 Mappings ===

results = evaluate_topk(
    topk_file=f"{results_dir}/{task}_top_1_mappings_BioBERT.tsv",
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

# === Display evaluation results ===
print(results)
# Outputs a dictionary with Precision@1, Recall@1, and F1@1

```

    📁 Top-1 file saved: /content/gdrive/My Drive/BioGITOM-VLDB//ncit2doid/Results/ncit2doid_top_1_mappings_BioBERT_predictions.tsv
    📌 Number of predictions in output: 2351
    {'Precision@1': 0.8707, 'Recall@1': 0.8623, 'F1@1': 0.8665}


# **Local MRR and Hit@k**


```python
import pandas as pd

# === Step 1: Load input files ===

# Define paths to cleaned embedding files
src_emb_path = f"{data_dir}/{src_ent}_final_embeddings_BioBERT_cleaned.tsv"
tgt_emb_path = f"{data_dir}/{tgt_ent}_final_embeddings_BioBERT_cleaned.tsv"

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
merged_src_df.to_csv(f"{data_dir}/{src_ent}_cands_with_embeddings_BioBERT.tsv", sep="\t", index=False)

# Save the target concepts and their embeddings to file
merged_tgt_df.to_csv(f"{data_dir}/{tgt_ent}_cands_with_embeddings_BioBERT.tsv", sep="\t", index=False)
```


```python
topk_faiss_l2(
    # Path to the source entity embeddings (already filtered and linearly encoded)
    src_emb_path=f"{data_dir}/{src_ent}_cands_with_embeddings_BioBERT.tsv",

    # Path to the target entity embeddings (already filtered and linearly encoded)
    tgt_emb_path=f"{data_dir}/{tgt_ent}_cands_with_embeddings_BioBERT.tsv",

    # Number of top matches to retrieve per source entity (Top-K candidates)
    top_k=200,

    # Path to save the resulting Top-K mappings sorted by FAISS L2 distance (converted to similarity)
    output_file=f"{results_dir}/{task}_top_200_mappings_mrr_hit_BioBERT.tsv"
)

```

    🔹 Using L2 (Euclidean) distance with FAISS
    Top-200 FAISS similarity results saved to: /content/gdrive/My Drive/BioGITOM-VLDB//ncit2doid/Results/ncit2doid_top_200_mappings_mrr_hit_BioBERT.tsv
    ⏱️ Execution time: 7.18 seconds



```python
test_cands = f"{dataset_dir}/refs_equiv/test.cands.tsv"
# Compute MRR and Hits@k metrics
# This function evaluates the predicted rankings against the reference mappings
results = compute_mrr_and_hits(
    reference_file=test_cands,             # Reference file with true ranks
    predicted_file=f"{results_dir}/{task}_top_200_mappings_mrr_hit_BioBERT.tsv",             # File containing predicted rankings
    output_file=formatted_predictions_path,    # File path to save formatted predictions
    k_values=[1, 5, 10]                        # Evaluate Hits@1, Hits@5, and Hits@10
)
```


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
    {'MRR': 0.7768680451932772, 'Hits@1': 0.686890243902439, 'Hits@5': 0.8890243902439025, 'Hits@10': 0.9384146341463414}



