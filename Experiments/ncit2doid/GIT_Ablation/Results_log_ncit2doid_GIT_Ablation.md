# **Package Installation**


```python
# We assume that PyTorch is already installed in the environment.
# If not, this command installs it.
!pip install torch==2.0.0

# Install PyTorch Geometric, a library for creating graph neural networks using PyTorch.
!pip install torch-geometric==2.4.0

# Import PyTorch to access its functionalities.
import torch

# Install additional PyTorch Geometric dependencies for graph processing (scatter, sparse, cluster, spline-conv).
# These packages enable operations like sparse tensors and convolutions on graphs.
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Reinstall PyTorch Geometric to ensure all dependencies are correctly loaded.
!pip install torch-geometric

# Retrieve the installed version of PyTorch to ensure compatibility with other packages.
torchversion = torch.__version__

# Install the latest version of PyTorch Geometric directly from the GitHub repository.
# This allows access to the most recent updates and features for graph-based neural networks.
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

# Install DeepOnto, a package specifically designed for ontology matching, particularly useful in biomedical applications.
!pip install deeponto

# Install a custom version of DeepOnto from a GitHub repository.
# The '<username>' part should be replaced with the actual GitHub username of the repository maintainer.
!pip install git+https://github.com/<username>/deeponto.git

```

    Collecting torch==2.0.0
      Using cached torch-2.0.0-cp310-cp310-manylinux1_x86_64.whl.metadata (24 kB)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch==2.0.0) (3.16.1)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch==2.0.0) (4.12.2)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch==2.0.0) (1.13.1)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch==2.0.0) (3.4.2)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch==2.0.0) (3.1.4)
    Collecting nvidia-cuda-nvrtc-cu11==11.7.99 (from torch==2.0.0)
      Using cached nvidia_cuda_nvrtc_cu11-11.7.99-2-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cuda-runtime-cu11==11.7.99 (from torch==2.0.0)
      Using cached nvidia_cuda_runtime_cu11-11.7.99-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cuda-cupti-cu11==11.7.101 (from torch==2.0.0)
      Using cached nvidia_cuda_cupti_cu11-11.7.101-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cudnn-cu11==8.5.0.96 (from torch==2.0.0)
      Using cached nvidia_cudnn_cu11-8.5.0.96-2-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cublas-cu11==11.10.3.66 (from torch==2.0.0)
      Using cached nvidia_cublas_cu11-11.10.3.66-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cufft-cu11==10.9.0.58 (from torch==2.0.0)
      Using cached nvidia_cufft_cu11-10.9.0.58-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-curand-cu11==10.2.10.91 (from torch==2.0.0)
      Using cached nvidia_curand_cu11-10.2.10.91-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cusolver-cu11==11.4.0.1 (from torch==2.0.0)
      Using cached nvidia_cusolver_cu11-11.4.0.1-2-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cusparse-cu11==11.7.4.91 (from torch==2.0.0)
      Using cached nvidia_cusparse_cu11-11.7.4.91-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-nccl-cu11==2.14.3 (from torch==2.0.0)
      Using cached nvidia_nccl_cu11-2.14.3-py3-none-manylinux1_x86_64.whl.metadata (1.8 kB)
    Collecting nvidia-nvtx-cu11==11.7.91 (from torch==2.0.0)
      Using cached nvidia_nvtx_cu11-11.7.91-py3-none-manylinux1_x86_64.whl.metadata (1.7 kB)
    Collecting triton==2.0.0 (from torch==2.0.0)
      Using cached triton-2.0.0-1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.0 kB)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from nvidia-cublas-cu11==11.10.3.66->torch==2.0.0) (75.1.0)
    Requirement already satisfied: wheel in /usr/local/lib/python3.10/dist-packages (from nvidia-cublas-cu11==11.10.3.66->torch==2.0.0) (0.45.0)
    Collecting cmake (from triton==2.0.0->torch==2.0.0)
      Using cached cmake-3.31.0.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.4 kB)
    Collecting lit (from triton==2.0.0->torch==2.0.0)
      Using cached lit-18.1.8-py3-none-any.whl.metadata (2.5 kB)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch==2.0.0) (3.0.2)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy->torch==2.0.0) (1.3.0)
    Downloading torch-2.0.0-cp310-cp310-manylinux1_x86_64.whl (619.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m619.9/619.9 MB[0m [31m1.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cublas_cu11-11.10.3.66-py3-none-manylinux1_x86_64.whl (317.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m317.1/317.1 MB[0m [31m3.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_cupti_cu11-11.7.101-py3-none-manylinux1_x86_64.whl (11.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.8/11.8 MB[0m [31m94.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_nvrtc_cu11-11.7.99-2-py3-none-manylinux1_x86_64.whl (21.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.0/21.0 MB[0m [31m79.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_runtime_cu11-11.7.99-py3-none-manylinux1_x86_64.whl (849 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m849.3/849.3 kB[0m [31m33.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cudnn_cu11-8.5.0.96-2-py3-none-manylinux1_x86_64.whl (557.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m557.1/557.1 MB[0m [31m952.1 kB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cufft_cu11-10.9.0.58-py3-none-manylinux2014_x86_64.whl (168.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m168.4/168.4 MB[0m [31m767.9 kB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_curand_cu11-10.2.10.91-py3-none-manylinux1_x86_64.whl (54.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m54.6/54.6 MB[0m [31m21.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cusolver_cu11-11.4.0.1-2-py3-none-manylinux1_x86_64.whl (102.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m102.6/102.6 MB[0m [31m10.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cusparse_cu11-11.7.4.91-py3-none-manylinux1_x86_64.whl (173.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m173.2/173.2 MB[0m [31m5.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_nccl_cu11-2.14.3-py3-none-manylinux1_x86_64.whl (177.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m177.1/177.1 MB[0m [31m5.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_nvtx_cu11-11.7.91-py3-none-manylinux1_x86_64.whl (98 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m98.6/98.6 kB[0m [31m5.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading triton-2.0.0-1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (63.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m63.3/63.3 MB[0m [31m18.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading cmake-3.31.0.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (27.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m27.8/27.8 MB[0m [31m58.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading lit-18.1.8-py3-none-any.whl (96 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m96.4/96.4 kB[0m [31m4.1 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: lit, nvidia-nvtx-cu11, nvidia-nccl-cu11, nvidia-cusparse-cu11, nvidia-curand-cu11, nvidia-cufft-cu11, nvidia-cuda-runtime-cu11, nvidia-cuda-nvrtc-cu11, nvidia-cuda-cupti-cu11, nvidia-cublas-cu11, cmake, nvidia-cusolver-cu11, nvidia-cudnn-cu11, triton, torch
      Attempting uninstall: torch
        Found existing installation: torch 2.5.1+cpu
        Uninstalling torch-2.5.1+cpu:
          Successfully uninstalled torch-2.5.1+cpu
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    torchaudio 2.5.1+cpu requires torch==2.5.1, but you have torch 2.0.0 which is incompatible.
    torchvision 0.20.1+cpu requires torch==2.5.1, but you have torch 2.0.0 which is incompatible.[0m[31m
    [0mSuccessfully installed cmake-3.31.0.1 lit-18.1.8 nvidia-cublas-cu11-11.10.3.66 nvidia-cuda-cupti-cu11-11.7.101 nvidia-cuda-nvrtc-cu11-11.7.99 nvidia-cuda-runtime-cu11-11.7.99 nvidia-cudnn-cu11-8.5.0.96 nvidia-cufft-cu11-10.9.0.58 nvidia-curand-cu11-10.2.10.91 nvidia-cusolver-cu11-11.4.0.1 nvidia-cusparse-cu11-11.7.4.91 nvidia-nccl-cu11-2.14.3 nvidia-nvtx-cu11-11.7.91 torch-2.0.0 triton-2.0.0
    Requirement already satisfied: torch-geometric==2.4.0 in /usr/local/lib/python3.10/dist-packages (2.4.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from torch-geometric==2.4.0) (4.66.6)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torch-geometric==2.4.0) (1.26.4)
    Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from torch-geometric==2.4.0) (1.13.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch-geometric==2.4.0) (3.1.4)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from torch-geometric==2.4.0) (2.32.3)
    Requirement already satisfied: pyparsing in /usr/local/lib/python3.10/dist-packages (from torch-geometric==2.4.0) (3.2.0)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (from torch-geometric==2.4.0) (1.5.2)
    Requirement already satisfied: psutil>=5.8.0 in /usr/local/lib/python3.10/dist-packages (from torch-geometric==2.4.0) (5.9.5)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch-geometric==2.4.0) (3.0.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->torch-geometric==2.4.0) (3.4.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->torch-geometric==2.4.0) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->torch-geometric==2.4.0) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->torch-geometric==2.4.0) (2024.8.30)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->torch-geometric==2.4.0) (1.4.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->torch-geometric==2.4.0) (3.5.0)
    Looking in links: https://data.pyg.org/whl/torch-2.0.0+cpu.html
    Collecting torch-scatter
      Using cached https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_scatter-2.1.2%2Bpt20cpu-cp310-cp310-linux_x86_64.whl (494 kB)
    Collecting torch-sparse
      Using cached https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_sparse-0.6.18%2Bpt20cpu-cp310-cp310-linux_x86_64.whl (1.2 MB)
    Collecting torch-cluster
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_cluster-1.6.3%2Bpt20cpu-cp310-cp310-linux_x86_64.whl (751 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m751.3/751.3 kB[0m [31m8.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting torch-spline-conv
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_spline_conv-1.2.2%2Bpt20cpu-cp310-cp310-linux_x86_64.whl (208 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m208.1/208.1 kB[0m [31m8.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from torch-sparse) (1.13.1)
    Requirement already satisfied: numpy<2.3,>=1.22.4 in /usr/local/lib/python3.10/dist-packages (from scipy->torch-sparse) (1.26.4)
    Installing collected packages: torch-spline-conv, torch-scatter, torch-sparse, torch-cluster
    Successfully installed torch-cluster-1.6.3+pt20cpu torch-scatter-2.1.2+pt20cpu torch-sparse-0.6.18+pt20cpu torch-spline-conv-1.2.2+pt20cpu
    Requirement already satisfied: torch-geometric in /usr/local/lib/python3.10/dist-packages (2.4.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from torch-geometric) (4.66.6)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torch-geometric) (1.26.4)
    Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from torch-geometric) (1.13.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch-geometric) (3.1.4)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from torch-geometric) (2.32.3)
    Requirement already satisfied: pyparsing in /usr/local/lib/python3.10/dist-packages (from torch-geometric) (3.2.0)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (from torch-geometric) (1.5.2)
    Requirement already satisfied: psutil>=5.8.0 in /usr/local/lib/python3.10/dist-packages (from torch-geometric) (5.9.5)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch-geometric) (3.0.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->torch-geometric) (3.4.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->torch-geometric) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->torch-geometric) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->torch-geometric) (2024.8.30)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->torch-geometric) (1.4.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->torch-geometric) (3.5.0)
      Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m67.7/67.7 kB[0m [31m1.6 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m22.0 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m241.9/241.9 kB[0m [31m16.5 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m124.6/124.6 kB[0m [31m7.4 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m208.9/208.9 kB[0m [31m10.3 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m319.4/319.4 kB[0m [31m16.5 MB/s[0m eta [36m0:00:00[0m
    [?25h  Building wheel for torch-geometric (pyproject.toml) ... [?25l[?25hdone
    Collecting deeponto
      Using cached deeponto-0.9.2-py3-none-any.whl.metadata (15 kB)
    Collecting JPype1 (from deeponto)
      Using cached jpype1-1.5.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)
    Collecting yacs (from deeponto)
      Using cached yacs-0.1.8-py3-none-any.whl.metadata (639 bytes)
    Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (from deeponto) (2.0.0)
    Collecting anytree (from deeponto)
      Using cached anytree-2.12.1-py3-none-any.whl.metadata (8.1 kB)
    Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from deeponto) (8.1.7)
    Collecting dill (from deeponto)
      Using cached dill-0.3.9-py3-none-any.whl.metadata (10 kB)
    Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from deeponto) (2.2.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from deeponto) (1.26.4)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (from deeponto) (1.5.2)
    Requirement already satisfied: transformers[torch] in /usr/local/lib/python3.10/dist-packages (from deeponto) (4.46.2)
    Collecting datasets (from deeponto)
      Using cached datasets-3.1.0-py3-none-any.whl.metadata (20 kB)
    Requirement already satisfied: spacy in /usr/local/lib/python3.10/dist-packages (from deeponto) (3.7.5)
    Collecting pprintpp (from deeponto)
      Using cached pprintpp-0.4.0-py2.py3-none-any.whl.metadata (7.9 kB)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from deeponto) (3.4.2)
    Collecting lxml (from deeponto)
      Using cached lxml-5.3.0-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (3.8 kB)
    Collecting textdistance (from deeponto)
      Using cached textdistance-4.6.3-py3-none-any.whl.metadata (18 kB)
    Requirement already satisfied: ipywidgets in /usr/local/lib/python3.10/dist-packages (from deeponto) (7.7.1)
    Requirement already satisfied: ipykernel in /usr/local/lib/python3.10/dist-packages (from deeponto) (5.5.6)
    Collecting enlighten (from deeponto)
      Using cached enlighten-1.12.4-py2.py3-none-any.whl.metadata (18 kB)
    Collecting rdflib (from deeponto)
      Using cached rdflib-7.1.1-py3-none-any.whl.metadata (11 kB)
    Requirement already satisfied: nltk in /usr/local/lib/python3.10/dist-packages (from deeponto) (3.9.1)
    Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from anytree->deeponto) (1.16.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from datasets->deeponto) (3.16.1)
    Requirement already satisfied: pyarrow>=15.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets->deeponto) (18.0.0)
    Collecting dill (from deeponto)
      Using cached dill-0.3.8-py3-none-any.whl.metadata (10 kB)
    Requirement already satisfied: requests>=2.32.2 in /usr/local/lib/python3.10/dist-packages (from datasets->deeponto) (2.32.3)
    Requirement already satisfied: tqdm>=4.66.3 in /usr/local/lib/python3.10/dist-packages (from datasets->deeponto) (4.66.6)
    Collecting xxhash (from datasets->deeponto)
      Using cached xxhash-3.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (12 kB)
    Collecting multiprocess<0.70.17 (from datasets->deeponto)
      Downloading multiprocess-0.70.16-py310-none-any.whl.metadata (7.2 kB)
    Collecting fsspec<=2024.9.0,>=2023.1.0 (from fsspec[http]<=2024.9.0,>=2023.1.0->datasets->deeponto)
      Downloading fsspec-2024.9.0-py3-none-any.whl.metadata (11 kB)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets->deeponto) (3.11.7)
    Requirement already satisfied: huggingface-hub>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from datasets->deeponto) (0.26.2)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from datasets->deeponto) (24.2)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from datasets->deeponto) (6.0.2)
    Collecting blessed>=1.17.7 (from enlighten->deeponto)
      Downloading blessed-1.20.0-py2.py3-none-any.whl.metadata (13 kB)
    Collecting prefixed>=0.3.2 (from enlighten->deeponto)
      Downloading prefixed-0.9.0-py2.py3-none-any.whl.metadata (11 kB)
    Requirement already satisfied: ipython-genutils in /usr/local/lib/python3.10/dist-packages (from ipykernel->deeponto) (0.2.0)
    Requirement already satisfied: ipython>=5.0.0 in /usr/local/lib/python3.10/dist-packages (from ipykernel->deeponto) (7.34.0)
    Requirement already satisfied: traitlets>=4.1.0 in /usr/local/lib/python3.10/dist-packages (from ipykernel->deeponto) (5.7.1)
    Requirement already satisfied: jupyter-client in /usr/local/lib/python3.10/dist-packages (from ipykernel->deeponto) (6.1.12)
    Requirement already satisfied: tornado>=4.2 in /usr/local/lib/python3.10/dist-packages (from ipykernel->deeponto) (6.3.3)
    Requirement already satisfied: widgetsnbextension~=3.6.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets->deeponto) (3.6.10)
    Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets->deeponto) (3.0.13)
    Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk->deeponto) (1.4.2)
    Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.10/dist-packages (from nltk->deeponto) (2024.11.6)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->deeponto) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->deeponto) (2024.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas->deeponto) (2024.2)
    Collecting isodate<1.0.0,>=0.7.2 (from rdflib->deeponto)
      Downloading isodate-0.7.2-py3-none-any.whl.metadata (11 kB)
    Requirement already satisfied: pyparsing<4,>=2.1.0 in /usr/local/lib/python3.10/dist-packages (from rdflib->deeponto) (3.2.0)
    Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->deeponto) (1.13.1)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->deeponto) (3.5.0)
    Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.11 in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (3.0.12)
    Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (1.0.5)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (1.0.10)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (2.0.8)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (3.0.9)
    Requirement already satisfied: thinc<8.3.0,>=8.2.2 in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (8.2.5)
    Requirement already satisfied: wasabi<1.2.0,>=0.9.1 in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (1.1.3)
    Requirement already satisfied: srsly<3.0.0,>=2.4.3 in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (2.4.8)
    Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (2.0.10)
    Requirement already satisfied: weasel<0.5.0,>=0.1.0 in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (0.4.1)
    Requirement already satisfied: typer<1.0.0,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (0.13.0)
    Requirement already satisfied: pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4 in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (2.9.2)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (3.1.4)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (75.1.0)
    Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in /usr/local/lib/python3.10/dist-packages (from spacy->deeponto) (3.4.1)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch->deeponto) (4.12.2)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch->deeponto) (1.13.1)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu11==11.7.99 in /usr/local/lib/python3.10/dist-packages (from torch->deeponto) (11.7.99)
    Requirement already satisfied: nvidia-cuda-runtime-cu11==11.7.99 in /usr/local/lib/python3.10/dist-packages (from torch->deeponto) (11.7.99)
    Requirement already satisfied: nvidia-cuda-cupti-cu11==11.7.101 in /usr/local/lib/python3.10/dist-packages (from torch->deeponto) (11.7.101)
    Requirement already satisfied: nvidia-cudnn-cu11==8.5.0.96 in /usr/local/lib/python3.10/dist-packages (from torch->deeponto) (8.5.0.96)
    Requirement already satisfied: nvidia-cublas-cu11==11.10.3.66 in /usr/local/lib/python3.10/dist-packages (from torch->deeponto) (11.10.3.66)
    Requirement already satisfied: nvidia-cufft-cu11==10.9.0.58 in /usr/local/lib/python3.10/dist-packages (from torch->deeponto) (10.9.0.58)
    Requirement already satisfied: nvidia-curand-cu11==10.2.10.91 in /usr/local/lib/python3.10/dist-packages (from torch->deeponto) (10.2.10.91)
    Requirement already satisfied: nvidia-cusolver-cu11==11.4.0.1 in /usr/local/lib/python3.10/dist-packages (from torch->deeponto) (11.4.0.1)
    Requirement already satisfied: nvidia-cusparse-cu11==11.7.4.91 in /usr/local/lib/python3.10/dist-packages (from torch->deeponto) (11.7.4.91)
    Requirement already satisfied: nvidia-nccl-cu11==2.14.3 in /usr/local/lib/python3.10/dist-packages (from torch->deeponto) (2.14.3)
    Requirement already satisfied: nvidia-nvtx-cu11==11.7.91 in /usr/local/lib/python3.10/dist-packages (from torch->deeponto) (11.7.91)
    Requirement already satisfied: triton==2.0.0 in /usr/local/lib/python3.10/dist-packages (from torch->deeponto) (2.0.0)
    Requirement already satisfied: wheel in /usr/local/lib/python3.10/dist-packages (from nvidia-cublas-cu11==11.10.3.66->torch->deeponto) (0.45.0)
    Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch->deeponto) (3.31.0.1)
    Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch->deeponto) (18.1.8)
    Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]->deeponto) (0.4.5)
    Requirement already satisfied: tokenizers<0.21,>=0.20 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]->deeponto) (0.20.3)
    Requirement already satisfied: accelerate>=0.26.0 in /usr/local/lib/python3.10/dist-packages (from transformers[torch]->deeponto) (1.1.1)
    Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from accelerate>=0.26.0->transformers[torch]->deeponto) (5.9.5)
    Requirement already satisfied: wcwidth>=0.1.4 in /usr/local/lib/python3.10/dist-packages (from blessed>=1.17.7->enlighten->deeponto) (0.2.13)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->deeponto) (2.4.3)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->deeponto) (1.3.1)
    Requirement already satisfied: async-timeout<6.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->deeponto) (5.0.1)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->deeponto) (24.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->deeponto) (1.5.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->deeponto) (6.1.0)
    Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->deeponto) (0.2.0)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->deeponto) (1.18.0)
    Collecting jedi>=0.16 (from ipython>=5.0.0->ipykernel->deeponto)
      Downloading jedi-0.19.2-py2.py3-none-any.whl.metadata (22 kB)
    Requirement already satisfied: decorator in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->deeponto) (5.1.1)
    Requirement already satisfied: pickleshare in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->deeponto) (0.7.5)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->deeponto) (3.0.48)
    Requirement already satisfied: pygments in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->deeponto) (2.18.0)
    Requirement already satisfied: backcall in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->deeponto) (0.2.0)
    Requirement already satisfied: matplotlib-inline in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->deeponto) (0.1.7)
    Requirement already satisfied: pexpect>4.3 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->deeponto) (4.9.0)
    Requirement already satisfied: language-data>=1.2 in /usr/local/lib/python3.10/dist-packages (from langcodes<4.0.0,>=3.2.0->spacy->deeponto) (1.2.0)
    Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.10/dist-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy->deeponto) (0.7.0)
    Requirement already satisfied: pydantic-core==2.23.4 in /usr/local/lib/python3.10/dist-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy->deeponto) (2.23.4)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.32.2->datasets->deeponto) (3.4.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.32.2->datasets->deeponto) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.32.2->datasets->deeponto) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.32.2->datasets->deeponto) (2024.8.30)
    Requirement already satisfied: blis<0.8.0,>=0.7.8 in /usr/local/lib/python3.10/dist-packages (from thinc<8.3.0,>=8.2.2->spacy->deeponto) (0.7.11)
    Requirement already satisfied: confection<1.0.0,>=0.0.1 in /usr/local/lib/python3.10/dist-packages (from thinc<8.3.0,>=8.2.2->spacy->deeponto) (0.1.5)
    Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from typer<1.0.0,>=0.3.0->spacy->deeponto) (1.5.4)
    Requirement already satisfied: rich>=10.11.0 in /usr/local/lib/python3.10/dist-packages (from typer<1.0.0,>=0.3.0->spacy->deeponto) (13.9.4)
    Requirement already satisfied: cloudpathlib<1.0.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from weasel<0.5.0,>=0.1.0->spacy->deeponto) (0.20.0)
    Requirement already satisfied: smart-open<8.0.0,>=5.2.1 in /usr/local/lib/python3.10/dist-packages (from weasel<0.5.0,>=0.1.0->spacy->deeponto) (7.0.5)
    Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.10/dist-packages (from widgetsnbextension~=3.6.0->ipywidgets->deeponto) (6.5.5)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->spacy->deeponto) (3.0.2)
    Requirement already satisfied: jupyter-core>=4.6.0 in /usr/local/lib/python3.10/dist-packages (from jupyter-client->ipykernel->deeponto) (5.7.2)
    Requirement already satisfied: pyzmq>=13 in /usr/local/lib/python3.10/dist-packages (from jupyter-client->ipykernel->deeponto) (24.0.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy->torch->deeponto) (1.3.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.4 in /usr/local/lib/python3.10/dist-packages (from jedi>=0.16->ipython>=5.0.0->ipykernel->deeponto) (0.8.4)
    Requirement already satisfied: platformdirs>=2.5 in /usr/local/lib/python3.10/dist-packages (from jupyter-core>=4.6.0->jupyter-client->ipykernel->deeponto) (4.3.6)
    Requirement already satisfied: marisa-trie>=0.7.7 in /usr/local/lib/python3.10/dist-packages (from language-data>=1.2->langcodes<4.0.0,>=3.2.0->spacy->deeponto) (1.2.1)
    Requirement already satisfied: argon2-cffi in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (23.1.0)
    Requirement already satisfied: nbformat in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (5.10.4)
    Requirement already satisfied: nbconvert>=5 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (7.16.4)
    Requirement already satisfied: nest-asyncio>=1.5 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.6.0)
    Requirement already satisfied: Send2Trash>=1.8.0 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.8.3)
    Requirement already satisfied: terminado>=0.8.3 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.18.1)
    Requirement already satisfied: prometheus-client in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.21.0)
    Requirement already satisfied: nbclassic>=0.4.7 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.1.0)
    Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.10/dist-packages (from pexpect>4.3->ipython>=5.0.0->ipykernel->deeponto) (0.7.0)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy->deeponto) (3.0.0)
    Requirement already satisfied: wrapt in /usr/local/lib/python3.10/dist-packages (from smart-open<8.0.0,>=5.2.1->weasel<0.5.0,>=0.1.0->spacy->deeponto) (1.14.1)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy->deeponto) (0.1.2)
    Requirement already satisfied: notebook-shim>=0.2.3 in /usr/local/lib/python3.10/dist-packages (from nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.2.4)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (4.12.3)
    Requirement already satisfied: bleach!=5.0.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (6.2.0)
    Requirement already satisfied: defusedxml in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.7.1)
    Requirement already satisfied: jupyterlab-pygments in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.3.0)
    Requirement already satisfied: mistune<4,>=2.0.3 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (3.0.2)
    Requirement already satisfied: nbclient>=0.5.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.10.0)
    Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.5.1)
    Requirement already satisfied: tinycss2 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.4.0)
    Requirement already satisfied: fastjsonschema>=2.15 in /usr/local/lib/python3.10/dist-packages (from nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (2.20.0)
    Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.10/dist-packages (from nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (4.23.0)
    Requirement already satisfied: argon2-cffi-bindings in /usr/local/lib/python3.10/dist-packages (from argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (21.2.0)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.10/dist-packages (from bleach!=5.0.0->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.5.1)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (2024.10.1)
    Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.35.1)
    Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (0.21.0)
    Requirement already satisfied: jupyter-server<3,>=1.8 in /usr/local/lib/python3.10/dist-packages (from notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.24.0)
    Requirement already satisfied: cffi>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.17.1)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (2.6)
    Requirement already satisfied: pycparser in /usr/local/lib/python3.10/dist-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (2.22)
    Requirement already satisfied: anyio<4,>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (3.7.1)
    Requirement already satisfied: websocket-client in /usr/local/lib/python3.10/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.8.0)
    Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.10/dist-packages (from anyio<4,>=3.1.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.3.1)
    Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<4,>=3.1.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->deeponto) (1.2.2)
    Downloading deeponto-0.9.2-py3-none-any.whl (89.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m89.7/89.7 MB[0m [31m12.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading anytree-2.12.1-py3-none-any.whl (44 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m44.9/44.9 kB[0m [31m2.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading datasets-3.1.0-py3-none-any.whl (480 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m480.6/480.6 kB[0m [31m26.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading dill-0.3.8-py3-none-any.whl (116 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m116.3/116.3 kB[0m [31m6.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading enlighten-1.12.4-py2.py3-none-any.whl (41 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m41.9/41.9 kB[0m [31m2.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading jpype1-1.5.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (493 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m493.8/493.8 kB[0m [31m29.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading lxml-5.3.0-cp310-cp310-manylinux_2_28_x86_64.whl (5.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.0/5.0 MB[0m [31m92.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pprintpp-0.4.0-py2.py3-none-any.whl (16 kB)
    Downloading rdflib-7.1.1-py3-none-any.whl (562 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m562.4/562.4 kB[0m [31m24.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading textdistance-4.6.3-py3-none-any.whl (31 kB)
    Downloading yacs-0.1.8-py3-none-any.whl (14 kB)
    Downloading blessed-1.20.0-py2.py3-none-any.whl (58 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.4/58.4 kB[0m [31m3.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading fsspec-2024.9.0-py3-none-any.whl (179 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m179.3/179.3 kB[0m [31m10.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading isodate-0.7.2-py3-none-any.whl (22 kB)
    Downloading multiprocess-0.70.16-py310-none-any.whl (134 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.8/134.8 kB[0m [31m7.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading prefixed-0.9.0-py2.py3-none-any.whl (13 kB)
    Downloading xxhash-3.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (194 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m194.1/194.1 kB[0m [31m10.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading jedi-0.19.2-py2.py3-none-any.whl (1.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m43.4 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: prefixed, pprintpp, yacs, xxhash, textdistance, lxml, JPype1, jedi, isodate, fsspec, dill, blessed, anytree, rdflib, multiprocess, enlighten, datasets, deeponto
      Attempting uninstall: fsspec
        Found existing installation: fsspec 2024.10.0
        Uninstalling fsspec-2024.10.0:
          Successfully uninstalled fsspec-2024.10.0
    Successfully installed JPype1-1.5.1 anytree-2.12.1 blessed-1.20.0 datasets-3.1.0 deeponto-0.9.2 dill-0.3.8 enlighten-1.12.4 fsspec-2024.9.0 isodate-0.7.2 jedi-0.19.2 lxml-5.3.0 multiprocess-0.70.16 pprintpp-0.4.0 prefixed-0.9.0 rdflib-7.1.1 textdistance-4.6.3 xxhash-3.5.0 yacs-0.1.8
    /bin/bash: line 1: username: No such file or directory



```python
# Import pandas for data manipulation and analysis, such as loading, processing, and saving tabular data.
import pandas as pd

# Import pickle for saving and loading serialized objects (e.g., trained models or preprocessed data).
import pickle

# Import function to convert a directed graph to an undirected one, useful for certain graph algorithms.
from torch_geometric.utils import to_undirected

# Import optimizer module from PyTorch for training models using gradient-based optimization techniques.
import torch.optim as optim

# Import PyTorch's modules for defining neural network architectures and operations:
from torch.nn import (
    Linear,       # For linear transformations (dense layers).
    Sequential,   # For stacking layers sequentially.
    BatchNorm1d,  # For normalizing input within mini-batches.
    PReLU,        # Parametric ReLU activation function.
    Dropout       # For regularization by randomly dropping connections during training.
)

# Import functional API from PyTorch for operations like activations and loss functions.
import torch.nn.functional as F

# Import Matplotlib for visualizations, such as plotting training loss curves.
import matplotlib.pyplot as plt

# Import PyTorch Geometric's graph convolutional layers:
from torch_geometric.nn import GCNConv, GINConv

# Import pooling operations for aggregating node embeddings to graph-level representations:
from torch_geometric.nn import global_mean_pool, global_add_pool

# Import NumPy for numerical operations, such as working with arrays and matrices.
import numpy as np

# Import time module for measuring execution time of code blocks.
import time

# Import typing module for specifying types in function arguments and return values.
from typing import Optional, Tuple, Union, Callable

# Import PyTorch's DataLoader and TensorDataset for handling data batching and loading during training.
from torch.utils.data import DataLoader, TensorDataset

# Import PyTorch's Parameter class for defining learnable parameters in custom models.
from torch.nn import Parameter

# Import math module for performing mathematical computations.
import math

# Import Tensor type from PyTorch for defining and manipulating tensors.
from torch import Tensor

# Import PyTorch's nn module for defining and building neural network architectures.
import torch.nn as nn

# Import initialization utilities from PyTorch Geometric for resetting weights and biases in layers.
from torch_geometric.nn.inits import reset

# Import the base class for defining message-passing layers in graph neural networks (GNNs).
from torch_geometric.nn.conv import MessagePassing

# Import linear transformation utilities for creating dense representations in graph models.
from torch_geometric.nn.dense.linear import Linear

# Import typing utilities for defining adjacency matrices and tensor types specific to PyTorch Geometric.
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor

# Import softmax function for normalizing attention scores in GNNs.
from torch_geometric.utils import softmax

# Import initialization utilities for weight initialization (e.g., Glorot initialization).
from torch_geometric.nn.inits import glorot, zeros

# Import F1 score metric from scikit-learn for evaluating model performance in binary/multi-class tasks.
from sklearn.metrics import f1_score

# Import JSON module for reading and writing JSON files, useful for storing configuration or ontology data.
import json

# Import Ontology class from DeepOnto for representing and manipulating ontologies in the pipeline.
from deeponto.onto import Ontology

# Import tools from DeepOnto for handling Ontology Alignment Evaluation Initiative (OAEI) tasks.
from deeponto.align.oaei import *

# Import evaluation tools from DeepOnto for assessing alignment results using metrics like precision, recall, and F1.
from deeponto.align.evaluation import AlignmentEvaluator

# Import mapping utilities from DeepOnto for working with reference mappings and entity pairs.
from deeponto.align.mapping import ReferenceMapping, EntityMapping

# Import utility function for reading tables (e.g., TSV, CSV) from DeepOnto.
from deeponto.utils import read_table

# Importing the train_test_split function from sklearn's model_selection module.
from sklearn.model_selection import train_test_split
```

    Please enter the maximum memory located to JVM [8g]: 8g
    


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

# Define the weight for the training data
# This weight is likely used to balance the training process, giving more emphasis to certain examples.
# For instance, a weight of 10.0 could be applied to penalize errors in certain types of predictions more heavily.
weight_train = 10.0

# Define the similarity threshold for validating matches
thres = 0.15
```


```python
dir = f"/content/gdrive/My Drive/BioGITOM-VLDB/Experiments/{task}"

dataset="/content/gdrive/My Drive/BioGITOM-VLDB/"

# Define the directory for the dataset containing source and target ontologies
dataset_dir = f"{dataset}/Datasets/{task}"

# Define the data directory for storing embeddings, adjacency matrices, and related files
data_dir = f"{dir}/Data"

# Define the directory for storing the results
results_dir = f"{dir}/GIT_Ablation/Results"
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
src_Emb = f"{data_dir}/{src_ent}_emb.csv"

# Define the file path for the Target embeddings CSV file
# Embeddings for the target ontology entities are stored in this file.
tgt_Emb = f"{data_dir}/{tgt_ent}_emb.csv"

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
train_file = f"{data_dir}/{task}_train_100.csv"

# Define the file path for the test data
# The test file contains reference mappings (ground truth) between the source and target ontologies.
test_file = f"{dataset_dir}/refs_equiv/test.tsv"

# Define the file path for the candidate mappings used during testing
# This file includes the candidate pairs (source and target entities) for ranking and evaluation.
test_cands = f"{dataset_dir}/refs_equiv/test.cands.tsv"

# Define the file path for the candidate mappings between Source to Target entities
# This file contains cleaned, combined, and encoded candidates used for predictions.
candidates_Prediction = f"{data_dir}/{task}_candidates_prediction.csv"

# Define the file path for the candidate mappings between Source to Target entities for ranking-based metrics
# This file is used to compute ranking-based metrics like MRR and Hits@k.
candidates_Rank = f"{data_dir}/{task}_candidates.csv"

# Define the path where the prediction results will be saved in TSV format
# This file will store the final predictions (mappings) between source and target entities.
prediction_path = f"{results_dir}/{task}_matching_results.tsv"

# Define the path where all prediction results will be saved in TSV format
# This file will store detailed prediction results, including all candidate scores.
all_predictions_path = f"{results_dir}/{task}_all_predictions.tsv"

# Define the path where all ranking prediction results will be saved in TSV format
# This file will store predictions sorted by rank based on their scores.
all_predictions_path_ranked = f"{results_dir}/{task}_all_predictions_ranked.tsv"

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
# Define the GatedCombination class for combining two pairs of embeddings using a gating mechanism
class GatedCombination(nn.Module):
    def __init__(self, input_dim):
        """
        Initialize the GatedCombination model.

        Args:
            input_dim (int): The dimensionality of the input embeddings (x1, x2, x3, x4).
        """
        super(GatedCombination, self).__init__()

        # Define a linear layer (gate) for combining embeddings x1 and x2 (first pair)
        self.gate_A_fc = nn.Linear(input_dim, input_dim)

        # Define a linear layer (gate) for combining embeddings x3 and x4 (second pair)
        self.gate_B_fc = nn.Linear(input_dim, input_dim)

        # A final fully connected layer that outputs a single neuron (binary classification)
        self.fc = nn.Linear(1, 1)

    def forward(self, x1, x2, x3, x4):
        """
        Forward pass through the gating mechanism and cosine similarity.

        Args:
            x1 (torch.Tensor): First set of embeddings (source embeddings after update).
            x2 (torch.Tensor): Second set of embeddings (original source embeddings).
            x3 (torch.Tensor): Third set of embeddings (target embeddings after update).
            x4 (torch.Tensor): Fourth set of embeddings (original target embeddings).

        Returns:
            torch.Tensor: Output of the model (probability score for binary classification).
        """
        # Compute gate values for the first pair (x1 and x2) using a sigmoid activation
        gate_values1 = torch.sigmoid(self.gate_A_fc(x1))

        # Combine x1 and x2 using the gate values
        # The result is a weighted combination of x1 and x2
        a = x1 * gate_values1 + x2 * (1 - gate_values1)

        # Compute gate values for the second pair (x3 and x4) using a sigmoid activation
        gate_values2 = torch.sigmoid(self.gate_B_fc(x3))

        # Combine x3 and x4 using the gate values
        # The result is a weighted combination of x3 and x4
        b = x3 * gate_values2 + x4 * (1 - gate_values2)

        # Compute cosine similarity between the combined vectors a and b
        x = torch.cosine_similarity(a, b, dim=1)

        # Pass the cosine similarity result through a fully connected layer (fc) for classification
        # Use a sigmoid activation to output a probability for binary classification
        out = torch.sigmoid(self.fc(x.unsqueeze(1)))  # unsqueeze(1) to match the input shape for the fc layer
        return out


```


```python
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        """
        Weighted Binary Cross-Entropy Loss.

        Args:
            pos_weight (float): Weight for the positive class.
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, outputs, targets):
        """
        Args:
            outputs (torch.Tensor): Predicted probabilities from the model (after sigmoid).
            targets (torch.Tensor): Ground truth labels (0 or 1).

        Returns:
            torch.Tensor: Computed weighted binary cross-entropy loss.
        """
        # Compute weighted BCE loss
        loss = - (self.pos_weight * targets * torch.log(outputs + 1e-8) +
                  (1 - targets) * torch.log(1 - outputs + 1e-8))
        return loss.mean()
```


```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        """
        Focal Loss for binary classification.

        Args:
            alpha (float): Balancing factor for positive/negative classes.
            gamma (float): Focusing parameter for hard examples.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        """
        Args:
            outputs (torch.Tensor): Predicted probabilities from the model (after sigmoid).
            targets (torch.Tensor): Ground truth labels (0 or 1).

        Returns:
            torch.Tensor: Computed focal loss.
        """
        # Compute binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(outputs, targets, reduction='none')

        # Compute modulating factor (1 - p_t)^gamma
        pt = torch.where(targets == 1, outputs, 1 - outputs)  # pt = p if y==1 else 1-p
        modulating_factor = (1 - pt) ** self.gamma

        # Apply alpha and modulating factor
        focal_loss = self.alpha * modulating_factor * bce_loss
        return focal_loss.mean()
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
def Prediction_with_candidates(model, X1_tt, X2_tt, X3_tt, X4_tt, src_entity_tensor_o, tgt_entity_tensor_o,
                                   indexed_dict_src, indexed_dict_tgt, all_predictions_path):
    """
    Evaluates the GatedCombination model using the given embeddings and candidate entity pairs.
    Saves the predictions and evaluation results to a file.

    Args:
        model: Trained GatedCombination model.
        X1_tt, X2_tt, X3_tt, X4_tt (torch.Tensor): Tensors of source and target entity embeddings (updated and original).
        src_entity_tensor_o, tgt_entity_tensor_o (torch.Tensor): Tensors of source and target entity indices.
        indexed_dict_src, indexed_dict_tgt (dict): Dictionaries mapping entity indices to URIs for source and target.
        output_file (str): Path to save the predictions and results.
        hits_at_k_values (list): List of k-values for which hits@k is evaluated.

    Returns:
        None
    """
    # Move the model to CPU and set it to evaluation mode
    model = model.to("cpu")
    model.eval()

    # Set batch size for evaluation
    batch_size_test = 32

    # Create a DataLoader for the evaluation data
    test_dataset = TensorDataset(X1_tt, X2_tt, X3_tt, X4_tt)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test)

    # Prepare for collecting predictions and results
    predictions = []
    results = []
    count_predictions = 0  # Counter for predictions above threshold (0.5)

    # Measure prediction time
    start_time = time.time()

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over batches and compute model predictions
        for batch_X1, batch_X2, batch_X3, batch_X4 in test_dataloader:
            outputs = model(batch_X1, batch_X2, batch_X3, batch_X4)
            predictions.extend(outputs.cpu().numpy())  # Collect predictions in CPU memory

    end_time = time.time()
    predicting_time = end_time - start_time
    print(f"Predicting time: {predicting_time:.2f} seconds")

    # Convert tensors to lists for easier iteration
    src_indices = src_entity_tensor_o.tolist()
    tgt_indices = tgt_entity_tensor_o.tolist()

    # Prepare results
    for i in range(len(predictions)):
        if predictions[i] >= 0.00:  # Consider only predictions greater than 0.5
            count_predictions += 1  # Increment the counter

            # Map the source and target entity indices to their URIs
            src_code = src_indices[i]
            tgt_code = tgt_indices[i]

            src_uri = indexed_dict_src.get(int(src_code), "Unknown URI")
            tgt_uri = indexed_dict_tgt.get(int(tgt_code), "Unknown URI")

            # Get the model's predicted score for the current pair
            score = predictions[i]

            # Append the results (with URIs instead of entity indices)
            results.append({
                'SrcEntity': src_uri,
                'TgtEntity': tgt_uri,
                'Score': score
            })

    # Convert the results into a pandas DataFrame
    df_results = pd.DataFrame(results)

    # Save the results to a TSV file
    df_results.to_csv(all_predictions_path, sep='\t', index=False)

    print(f"Predictions saved to {all_predictions_path}")
```


```python
def filter_highest_predictions(input_file_path, output_file_path, threshold=thres):
    # Load the all predictions file
    df = pd.read_csv(input_file_path, sep='\t')

    # Extract the similarity score from the list in the 'Score' column
    df['Score'] = df['Score'].apply(lambda x: float(x.strip('[]')))

    # Sorting the dataframe by similarity score in descending order
    df_sorted = df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    # Initialize variables with threshold value
    source_concepts = set(df_sorted['SrcEntity'])
    target_concepts = set(df_sorted['TgtEntity'])
    matched_sources = set()
    matched_targets = set()
    result = []

    # Iterate through the sorted dataframe and find highest correspondences
    for _, row in df_sorted.iterrows():
        source, target, similarity = row['SrcEntity'], row['TgtEntity'], row['Score']

        # Check if the source or target has already been matched and if the similarity is above the threshold
        if source not in matched_sources and target not in matched_targets and similarity >= threshold:
            # Add the match to the result list
            result.append((source, target, similarity))
            # Mark the source and target as matched
            matched_sources.add(source)
            matched_targets.add(target)

    # Create a dataframe for the matching results with threshold applied
    matching_results_df_threshold = pd.DataFrame(result, columns=['SrcEntity', 'TgtEntity', 'Score'])

    # Save the matching results with the updated column names to a new TSV file
    matching_results_df_threshold.to_csv(output_file_path, sep='\t', index=False)

    # Print the number of predictions saved
    print(f"Number of Positive predictions: {len(matching_results_df_threshold)}")

    return matching_results_df_threshold, len(matching_results_df_threshold)
```


```python
def compute_mrr_and_hits(reference_file, predicted_file, output_file, k_values=[1, 5, 10]):
    """
    Compute MRR and Hits@k for ontology matching predictions based on a reference file.

    Args:
        reference_file (str): Path to the reference file (test.cands.tsv format).
        predicted_file (str): Path to the predictions file with scores.
        output_file (str): Path to save the scored results.
        k_values (list): List of k values for Hits@k.

    Returns:
        dict: A dictionary containing MRR and Hits@k metrics.
    """
    # Read the reference mappings
    test_candidate_mappings = read_table(reference_file).values.tolist()
    ranking_results = []

    # Read the predicted scores
    predicted_data = pd.read_csv(predicted_file, sep="\t")
    predicted_data["Score"] = predicted_data["Score"].apply(lambda x: float(x.strip("[]")))

    # Create a lookup dictionary for predicted scores
    score_lookup = {}
    for _, row in predicted_data.iterrows():
        score_lookup[(row["SrcEntity"], row["TgtEntity"])] = row["Score"]

    for src_ref_class, tgt_ref_class, tgt_cands in test_candidate_mappings:
        tgt_cands = eval(tgt_cands)  # Convert string to list of candidates
        scored_cands = []
        for tgt_cand in tgt_cands:
            # Retrieve score for each candidate, defaulting to a very low score if not found
            matching_score = score_lookup.get((src_ref_class, tgt_cand), -1e9)
            scored_cands.append((tgt_cand, matching_score))

        # Sort candidates by score in descending order
        scored_cands = sorted(scored_cands, key=lambda x: x[1], reverse=True)
        ranking_results.append((src_ref_class, tgt_ref_class, scored_cands))

    # Save the ranked results to a file
    pd.DataFrame(ranking_results, columns=["SrcEntity", "TgtEntity", "TgtCandidates"]).to_csv(output_file, sep="\t", index=False)

    # Compute MRR and Hits@k
    total_entities = len(ranking_results)
    reciprocal_ranks = []
    hits_at_k = {k: 0 for k in k_values}

    for src_entity, tgt_ref_class, tgt_cands in ranking_results:
        ranked_candidates = [candidate[0] for candidate in tgt_cands]
        if tgt_ref_class in ranked_candidates:
            rank = ranked_candidates.index(tgt_ref_class) + 1
            reciprocal_ranks.append(1 / rank)
            for k in k_values:
                if rank <= k:
                    hits_at_k[k] += 1
        else:
            reciprocal_ranks.append(0)

    mrr = sum(reciprocal_ranks) / total_entities
    hits_at_k = {k: hits / total_entities for k, hits in hits_at_k.items()}

    return {"MRR": mrr, "Hits@k": hits_at_k}
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

    Epoch [10/1000], Training Loss: 0.0018639207119122148
    Epoch [20/1000], Training Loss: 0.0015663406811654568
    Epoch [30/1000], Training Loss: 0.0013771239900961518
    Epoch [40/1000], Training Loss: 0.001246417872607708
    Epoch [50/1000], Training Loss: 0.0011453819461166859
    Epoch [60/1000], Training Loss: 0.001067296601831913
    Epoch [70/1000], Training Loss: 0.0010031838901340961
    Epoch [80/1000], Training Loss: 0.0009486632770858705
    Epoch [90/1000], Training Loss: 0.0009017452830448747
    Epoch [100/1000], Training Loss: 0.0008612964884378016
    Epoch [110/1000], Training Loss: 0.0008256665314547718
    Epoch [120/1000], Training Loss: 0.0007943543023429811
    Epoch [130/1000], Training Loss: 0.0007669503684155643
    Epoch [140/1000], Training Loss: 0.0007421737536787987
    Epoch [150/1000], Training Loss: 0.0007202554261311889
    Epoch [160/1000], Training Loss: 0.0007009028922766447
    Epoch [170/1000], Training Loss: 0.0006834416999481618
    Epoch [180/1000], Training Loss: 0.0006678910576738417
    Epoch [190/1000], Training Loss: 0.0006536765722557902
    Epoch [200/1000], Training Loss: 0.0006411538342945278
    Epoch [210/1000], Training Loss: 0.0006300000241026282
    Epoch [220/1000], Training Loss: 0.0006199842900969088
    Epoch [230/1000], Training Loss: 0.0006108981906436384
    Epoch [240/1000], Training Loss: 0.000602675077971071
    Epoch [250/1000], Training Loss: 0.0005952027277089655
    Epoch [260/1000], Training Loss: 0.0005884325946681201
    Epoch [270/1000], Training Loss: 0.000582334294449538
    Epoch [280/1000], Training Loss: 0.0005767731345258653
    Epoch [290/1000], Training Loss: 0.0005716997548006475
    Epoch [300/1000], Training Loss: 0.0005670347018167377
    Epoch [310/1000], Training Loss: 0.0005626484635286033
    Epoch [320/1000], Training Loss: 0.0005586721817962825
    Epoch [330/1000], Training Loss: 0.0005549676716327667
    Epoch [340/1000], Training Loss: 0.000551374803762883
    Epoch [350/1000], Training Loss: 0.0005480060935951769
    Epoch [360/1000], Training Loss: 0.0005447924486361444
    Epoch [370/1000], Training Loss: 0.0005416946369223297
    Epoch [380/1000], Training Loss: 0.0005387563141994178
    Epoch [390/1000], Training Loss: 0.000535941100679338
    Epoch [400/1000], Training Loss: 0.0005332588916644454
    Epoch [410/1000], Training Loss: 0.0005306496750563383
    Epoch [420/1000], Training Loss: 0.0005281417397782207
    Epoch [430/1000], Training Loss: 0.0005257632001303136
    Epoch [440/1000], Training Loss: 0.0005234282580204308
    Epoch [450/1000], Training Loss: 0.0005211543757468462
    Epoch [460/1000], Training Loss: 0.0005189166404306889
    Epoch [470/1000], Training Loss: 0.0005167184281162918
    Epoch [480/1000], Training Loss: 0.0005145713221281767
    Epoch [490/1000], Training Loss: 0.000512409140355885
    Epoch [500/1000], Training Loss: 0.0005102347931824625
    Epoch [510/1000], Training Loss: 0.0005081003182567656
    Epoch [520/1000], Training Loss: 0.0005059898248873651
    Epoch [530/1000], Training Loss: 0.0005038987728767097
    Epoch [540/1000], Training Loss: 0.0005018521915189922
    Epoch [550/1000], Training Loss: 0.0004998291842639446
    Epoch [560/1000], Training Loss: 0.0004978120559826493
    Epoch [570/1000], Training Loss: 0.0004958126228302717
    Epoch [580/1000], Training Loss: 0.0004938197671435773
    Epoch [590/1000], Training Loss: 0.0004918361082673073
    Epoch [600/1000], Training Loss: 0.0004898050101473927
    Epoch [610/1000], Training Loss: 0.00048777993652038276
    Epoch [620/1000], Training Loss: 0.00048572689411230385
    Epoch [630/1000], Training Loss: 0.00048371002776548266
    Epoch [640/1000], Training Loss: 0.00048170567606575787
    Epoch [650/1000], Training Loss: 0.00047970289597287774
    Epoch [660/1000], Training Loss: 0.0004776912392117083
    Epoch [670/1000], Training Loss: 0.0004757391579914838
    Epoch [680/1000], Training Loss: 0.000473849184345454
    Epoch [690/1000], Training Loss: 0.00047199506661854684
    Epoch [700/1000], Training Loss: 0.0004701538709923625
    Epoch [710/1000], Training Loss: 0.000468330952571705
    Epoch [720/1000], Training Loss: 0.00046656548511236906
    Epoch [730/1000], Training Loss: 0.00046485886559821665
    Epoch [740/1000], Training Loss: 0.0004631704359780997
    Epoch [750/1000], Training Loss: 0.00046149277477525175
    Epoch [760/1000], Training Loss: 0.00045982131268829107
    Epoch [770/1000], Training Loss: 0.0004581693501677364
    Epoch [780/1000], Training Loss: 0.0004565674753393978
    Epoch [790/1000], Training Loss: 0.0004549552104435861
    Epoch [800/1000], Training Loss: 0.00045339635107666254
    Epoch [810/1000], Training Loss: 0.00045187247451394796
    Epoch [820/1000], Training Loss: 0.0004503539530560374
    Epoch [830/1000], Training Loss: 0.00044883450027555227
    Epoch [840/1000], Training Loss: 0.0004473602748475969
    Epoch [850/1000], Training Loss: 0.00044587315642274916
    Epoch [860/1000], Training Loss: 0.00044439840712584555
    Epoch [870/1000], Training Loss: 0.00044293279643170536
    Epoch [880/1000], Training Loss: 0.00044147332664579153
    Epoch [890/1000], Training Loss: 0.0004400189209263772
    Epoch [900/1000], Training Loss: 0.00043856515549123287
    Epoch [910/1000], Training Loss: 0.00043710885802283883
    Epoch [920/1000], Training Loss: 0.00043564828229136765
    Epoch [930/1000], Training Loss: 0.00043421160080470145
    Epoch [940/1000], Training Loss: 0.0004327642382122576
    Epoch [950/1000], Training Loss: 0.0004313233948778361
    Epoch [960/1000], Training Loss: 0.0004298958519939333
    Epoch [970/1000], Training Loss: 0.00042844729614444077
    Epoch [980/1000], Training Loss: 0.00042698316974565387
    Epoch [990/1000], Training Loss: 0.0004255363892298192
    Epoch [1000/1000], Training Loss: 0.0004240930429659784



    
![png](output_36_1.png)
    


    Training complete! Total training time: 1080.83 seconds


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


# x_src: Original source embeddings
# x_tgt: Original target embeddings
X1_train = select_rows_by_index(x_src, tensor_term1_train)  # Use original source embeddings
X2_train = select_rows_by_index(x_src, tensor_term1_train)
X3_train = select_rows_by_index(x_tgt, tensor_term2_train)  # Use original target embeddings
X4_train = select_rows_by_index(x_tgt, tensor_term2_train)

X1_val = select_rows_by_index(x_src, tensor_term1_val)
X2_val = select_rows_by_index(x_src, tensor_term1_val)
X3_val = select_rows_by_index(x_tgt, tensor_term2_val)
X4_val = select_rows_by_index(x_tgt, tensor_term2_val)


```


```python
positive_weight = len(tensor_score_train) / (weight_train * tensor_score_train.sum())
```

# Gated Network Training


```python
def train_gated_combination_model(X1_t, X2_t, X3_t, X4_t, tensor_score_o,
                                  X1_val, X2_val, X3_val, X4_val, tensor_score_val,
                                  epochs=120, batch_size=32, learning_rate=0.001, weight_decay=1e-5):
    """
    Trains the GatedCombination model with training and validation data, using ReduceLROnPlateau scheduler.
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

    criterion = WeightedBCELoss(pos_weight=positive_weight).to(device)

    train_losses, val_losses = [], []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_train_loss, y_true_train, y_pred_train = 0.0, [], []

        for batch_X1, batch_X2, batch_X3, batch_X4, batch_y in train_loader:
            batch_X1, batch_X2, batch_X3, batch_X4, batch_y = (batch_X1.to(device), batch_X2.to(device),
                                                               batch_X3.to(device), batch_X4.to(device), batch_y.to(device))
            optimizer.zero_grad()
            outputs = model(batch_X1, batch_X2, batch_X3, batch_X4)
            loss = criterion(outputs, batch_y.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            y_true_train.extend(batch_y.cpu().numpy())
            y_pred_train.extend((outputs > 0.2).float().cpu().numpy())

        train_f1 = f1_score(y_true_train, y_pred_train)
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        total_val_loss, y_true_val, y_pred_val = 0.0, [], []
        with torch.no_grad():
            for batch_X1, batch_X2, batch_X3, batch_X4, batch_y in val_loader:
                batch_X1, batch_X2, batch_X3, batch_X4, batch_y = (batch_X1.to(device), batch_X2.to(device),
                                                                   batch_X3.to(device), batch_X4.to(device), batch_y.to(device))
                outputs = model(batch_X1, batch_X2, batch_X3, batch_X4)
                val_loss = criterion(outputs, batch_y.unsqueeze(1).float())
                total_val_loss += val_loss.item()
                y_true_val.extend(batch_y.cpu().numpy())
                y_pred_val.extend((outputs > 0.4).float().cpu().numpy())

        val_f1 = f1_score(y_true_val, y_pred_val)
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Step the scheduler with validation loss
        scheduler.step(avg_val_loss)

        # Print training and validation metrics
        print(f"Epoch [{epoch + 1}/{epochs}] Training Loss: {train_loss:.4f}, F1 Score: {train_f1:.4f} | "
              f"Validation Loss: {avg_val_loss:.4f}, F1 Score: {val_f1:.4f}")

    end_time = time.time()

    # Plotting
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
    weight_decay=1e-5  # Weight decay (L2 regularization) to prevent overfitting
)
```

    Epoch [1/100] Training Loss: 0.4159, F1 Score: 0.0184 | Validation Loss: 0.3547, F1 Score: 0.0000
    Epoch [2/100] Training Loss: 0.3273, F1 Score: 0.0000 | Validation Loss: 0.3266, F1 Score: 0.0000
    Epoch [3/100] Training Loss: 0.3021, F1 Score: 0.0000 | Validation Loss: 0.3022, F1 Score: 0.0000
    Epoch [4/100] Training Loss: 0.2799, F1 Score: 0.2146 | Validation Loss: 0.2794, F1 Score: 0.0000
    Epoch [5/100] Training Loss: 0.2594, F1 Score: 0.7028 | Validation Loss: 0.2587, F1 Score: 0.0000
    Epoch [6/100] Training Loss: 0.2410, F1 Score: 0.8280 | Validation Loss: 0.2401, F1 Score: 0.0000
    Epoch [7/100] Training Loss: 0.2240, F1 Score: 0.8538 | Validation Loss: 0.2227, F1 Score: 0.1432
    Epoch [8/100] Training Loss: 0.2083, F1 Score: 0.8386 | Validation Loss: 0.2068, F1 Score: 0.4065
    Epoch [9/100] Training Loss: 0.1940, F1 Score: 0.7868 | Validation Loss: 0.1925, F1 Score: 0.5329
    Epoch [10/100] Training Loss: 0.1811, F1 Score: 0.7537 | Validation Loss: 0.1795, F1 Score: 0.6566
    Epoch [11/100] Training Loss: 0.1695, F1 Score: 0.7169 | Validation Loss: 0.1679, F1 Score: 0.7165
    Epoch [12/100] Training Loss: 0.1591, F1 Score: 0.6844 | Validation Loss: 0.1575, F1 Score: 0.7738
    Epoch [13/100] Training Loss: 0.1498, F1 Score: 0.6507 | Validation Loss: 0.1481, F1 Score: 0.8063
    Epoch [14/100] Training Loss: 0.1413, F1 Score: 0.6432 | Validation Loss: 0.1396, F1 Score: 0.8372
    Epoch [15/100] Training Loss: 0.1337, F1 Score: 0.6029 | Validation Loss: 0.1321, F1 Score: 0.8372
    Epoch [16/100] Training Loss: 0.1269, F1 Score: 0.6069 | Validation Loss: 0.1252, F1 Score: 0.8561
    Epoch [17/100] Training Loss: 0.1207, F1 Score: 0.5946 | Validation Loss: 0.1192, F1 Score: 0.8557
    Epoch [18/100] Training Loss: 0.1152, F1 Score: 0.5906 | Validation Loss: 0.1135, F1 Score: 0.8638
    Epoch [19/100] Training Loss: 0.1102, F1 Score: 0.5958 | Validation Loss: 0.1084, F1 Score: 0.8765
    Epoch [20/100] Training Loss: 0.1057, F1 Score: 0.5690 | Validation Loss: 0.1041, F1 Score: 0.8783
    Epoch [21/100] Training Loss: 0.1016, F1 Score: 0.5843 | Validation Loss: 0.0999, F1 Score: 0.8825
    Epoch [22/100] Training Loss: 0.0980, F1 Score: 0.5706 | Validation Loss: 0.0962, F1 Score: 0.8815
    Epoch [23/100] Training Loss: 0.0947, F1 Score: 0.5679 | Validation Loss: 0.0928, F1 Score: 0.8815
    Epoch [24/100] Training Loss: 0.0916, F1 Score: 0.5671 | Validation Loss: 0.0897, F1 Score: 0.8816
    Epoch [25/100] Training Loss: 0.0888, F1 Score: 0.5609 | Validation Loss: 0.0868, F1 Score: 0.8829
    Epoch [26/100] Training Loss: 0.0862, F1 Score: 0.5692 | Validation Loss: 0.0842, F1 Score: 0.8746
    Epoch [27/100] Training Loss: 0.0839, F1 Score: 0.5599 | Validation Loss: 0.0818, F1 Score: 0.8749
    Epoch [28/100] Training Loss: 0.0817, F1 Score: 0.5665 | Validation Loss: 0.0796, F1 Score: 0.8667
    Epoch [29/100] Training Loss: 0.0798, F1 Score: 0.5562 | Validation Loss: 0.0776, F1 Score: 0.8674
    Epoch [30/100] Training Loss: 0.0779, F1 Score: 0.5626 | Validation Loss: 0.0757, F1 Score: 0.8654
    Epoch [31/100] Training Loss: 0.0762, F1 Score: 0.5576 | Validation Loss: 0.0740, F1 Score: 0.8667
    Epoch [32/100] Training Loss: 0.0746, F1 Score: 0.5568 | Validation Loss: 0.0724, F1 Score: 0.8654
    Epoch [33/100] Training Loss: 0.0732, F1 Score: 0.5629 | Validation Loss: 0.0709, F1 Score: 0.8639
    Epoch [34/100] Training Loss: 0.0718, F1 Score: 0.5594 | Validation Loss: 0.0694, F1 Score: 0.8596
    Epoch [35/100] Training Loss: 0.0705, F1 Score: 0.5565 | Validation Loss: 0.0681, F1 Score: 0.8623
    Epoch [36/100] Training Loss: 0.0692, F1 Score: 0.5677 | Validation Loss: 0.0668, F1 Score: 0.8548
    Epoch [37/100] Training Loss: 0.0681, F1 Score: 0.5580 | Validation Loss: 0.0657, F1 Score: 0.8548
    Epoch [38/100] Training Loss: 0.0670, F1 Score: 0.5495 | Validation Loss: 0.0646, F1 Score: 0.8562
    Epoch [39/100] Training Loss: 0.0661, F1 Score: 0.5664 | Validation Loss: 0.0635, F1 Score: 0.8512
    Epoch [40/100] Training Loss: 0.0651, F1 Score: 0.5618 | Validation Loss: 0.0626, F1 Score: 0.8454
    Epoch [41/100] Training Loss: 0.0642, F1 Score: 0.5563 | Validation Loss: 0.0617, F1 Score: 0.8454
    Epoch [42/100] Training Loss: 0.0634, F1 Score: 0.5548 | Validation Loss: 0.0608, F1 Score: 0.8480
    Epoch [43/100] Training Loss: 0.0626, F1 Score: 0.5683 | Validation Loss: 0.0600, F1 Score: 0.8405
    Epoch [44/100] Training Loss: 0.0619, F1 Score: 0.5493 | Validation Loss: 0.0592, F1 Score: 0.8480
    Epoch [45/100] Training Loss: 0.0612, F1 Score: 0.5631 | Validation Loss: 0.0584, F1 Score: 0.8445
    Epoch [46/100] Training Loss: 0.0605, F1 Score: 0.5621 | Validation Loss: 0.0578, F1 Score: 0.8405
    Epoch [47/100] Training Loss: 0.0599, F1 Score: 0.5570 | Validation Loss: 0.0571, F1 Score: 0.8428
    Epoch [48/100] Training Loss: 0.0592, F1 Score: 0.5664 | Validation Loss: 0.0565, F1 Score: 0.8354
    Epoch [49/100] Training Loss: 0.0587, F1 Score: 0.5607 | Validation Loss: 0.0558, F1 Score: 0.8354
    Epoch [50/100] Training Loss: 0.0581, F1 Score: 0.5597 | Validation Loss: 0.0553, F1 Score: 0.8354
    Epoch [51/100] Training Loss: 0.0576, F1 Score: 0.5580 | Validation Loss: 0.0547, F1 Score: 0.8362
    Epoch [52/100] Training Loss: 0.0571, F1 Score: 0.5597 | Validation Loss: 0.0542, F1 Score: 0.8362
    Epoch [53/100] Training Loss: 0.0566, F1 Score: 0.5614 | Validation Loss: 0.0537, F1 Score: 0.8354
    Epoch [54/100] Training Loss: 0.0562, F1 Score: 0.5633 | Validation Loss: 0.0532, F1 Score: 0.8323
    Epoch [55/100] Training Loss: 0.0558, F1 Score: 0.5566 | Validation Loss: 0.0528, F1 Score: 0.8354
    Epoch [56/100] Training Loss: 0.0553, F1 Score: 0.5633 | Validation Loss: 0.0523, F1 Score: 0.8315
    Epoch [57/100] Training Loss: 0.0549, F1 Score: 0.5600 | Validation Loss: 0.0519, F1 Score: 0.8315
    Epoch [58/100] Training Loss: 0.0545, F1 Score: 0.5654 | Validation Loss: 0.0515, F1 Score: 0.8290
    Epoch [59/100] Training Loss: 0.0542, F1 Score: 0.5587 | Validation Loss: 0.0511, F1 Score: 0.8332
    Epoch [60/100] Training Loss: 0.0538, F1 Score: 0.5656 | Validation Loss: 0.0507, F1 Score: 0.8293
    Epoch [61/100] Training Loss: 0.0535, F1 Score: 0.5626 | Validation Loss: 0.0504, F1 Score: 0.8285
    Epoch [62/100] Training Loss: 0.0531, F1 Score: 0.5692 | Validation Loss: 0.0500, F1 Score: 0.8162
    Epoch [63/100] Training Loss: 0.0529, F1 Score: 0.5573 | Validation Loss: 0.0497, F1 Score: 0.8219
    Epoch [64/100] Training Loss: 0.0526, F1 Score: 0.5657 | Validation Loss: 0.0494, F1 Score: 0.8162
    Epoch [65/100] Training Loss: 0.0524, F1 Score: 0.5664 | Validation Loss: 0.0491, F1 Score: 0.8150
    Epoch [66/100] Training Loss: 0.0519, F1 Score: 0.5594 | Validation Loss: 0.0488, F1 Score: 0.8166
    Epoch [67/100] Training Loss: 0.0517, F1 Score: 0.5614 | Validation Loss: 0.0485, F1 Score: 0.8162
    Epoch [68/100] Training Loss: 0.0514, F1 Score: 0.5676 | Validation Loss: 0.0482, F1 Score: 0.8110
    Epoch [69/100] Training Loss: 0.0512, F1 Score: 0.5615 | Validation Loss: 0.0479, F1 Score: 0.8174
    Epoch [70/100] Training Loss: 0.0509, F1 Score: 0.5656 | Validation Loss: 0.0476, F1 Score: 0.8166
    Epoch [71/100] Training Loss: 0.0507, F1 Score: 0.5616 | Validation Loss: 0.0474, F1 Score: 0.8174
    Epoch [72/100] Training Loss: 0.0505, F1 Score: 0.5678 | Validation Loss: 0.0472, F1 Score: 0.8142
    Epoch [73/100] Training Loss: 0.0502, F1 Score: 0.5679 | Validation Loss: 0.0469, F1 Score: 0.8102
    Epoch [74/100] Training Loss: 0.0500, F1 Score: 0.5600 | Validation Loss: 0.0467, F1 Score: 0.8166
    Epoch [75/100] Training Loss: 0.0498, F1 Score: 0.5685 | Validation Loss: 0.0465, F1 Score: 0.8110
    Epoch [76/100] Training Loss: 0.0496, F1 Score: 0.5683 | Validation Loss: 0.0463, F1 Score: 0.8062
    Epoch [77/100] Training Loss: 0.0494, F1 Score: 0.5609 | Validation Loss: 0.0460, F1 Score: 0.8134
    Epoch [78/100] Training Loss: 0.0492, F1 Score: 0.5720 | Validation Loss: 0.0459, F1 Score: 0.8039
    Epoch [79/100] Training Loss: 0.0490, F1 Score: 0.5624 | Validation Loss: 0.0456, F1 Score: 0.8062
    Epoch [80/100] Training Loss: 0.0489, F1 Score: 0.5653 | Validation Loss: 0.0454, F1 Score: 0.8094
    Epoch [81/100] Training Loss: 0.0487, F1 Score: 0.5714 | Validation Loss: 0.0453, F1 Score: 0.8031
    Epoch [82/100] Training Loss: 0.0485, F1 Score: 0.5657 | Validation Loss: 0.0451, F1 Score: 0.8039
    Epoch [83/100] Training Loss: 0.0484, F1 Score: 0.5711 | Validation Loss: 0.0450, F1 Score: 0.8046
    Epoch [84/100] Training Loss: 0.0482, F1 Score: 0.5634 | Validation Loss: 0.0447, F1 Score: 0.8039
    Epoch [85/100] Training Loss: 0.0481, F1 Score: 0.5674 | Validation Loss: 0.0446, F1 Score: 0.8039
    Epoch [86/100] Training Loss: 0.0479, F1 Score: 0.5672 | Validation Loss: 0.0444, F1 Score: 0.8031
    Epoch [87/100] Training Loss: 0.0478, F1 Score: 0.5665 | Validation Loss: 0.0442, F1 Score: 0.8039
    Epoch [88/100] Training Loss: 0.0476, F1 Score: 0.5762 | Validation Loss: 0.0442, F1 Score: 0.8027
    Epoch [89/100] Training Loss: 0.0475, F1 Score: 0.5623 | Validation Loss: 0.0440, F1 Score: 0.8023
    Epoch [90/100] Training Loss: 0.0474, F1 Score: 0.5690 | Validation Loss: 0.0438, F1 Score: 0.8023
    Epoch [91/100] Training Loss: 0.0472, F1 Score: 0.5653 | Validation Loss: 0.0437, F1 Score: 0.8039
    Epoch [92/100] Training Loss: 0.0471, F1 Score: 0.5686 | Validation Loss: 0.0435, F1 Score: 0.8039
    Epoch [93/100] Training Loss: 0.0470, F1 Score: 0.5697 | Validation Loss: 0.0434, F1 Score: 0.8039
    Epoch [94/100] Training Loss: 0.0469, F1 Score: 0.5698 | Validation Loss: 0.0433, F1 Score: 0.8023
    Epoch [95/100] Training Loss: 0.0467, F1 Score: 0.5730 | Validation Loss: 0.0432, F1 Score: 0.8027
    Epoch [96/100] Training Loss: 0.0466, F1 Score: 0.5674 | Validation Loss: 0.0430, F1 Score: 0.8046
    Epoch [97/100] Training Loss: 0.0465, F1 Score: 0.5676 | Validation Loss: 0.0429, F1 Score: 0.8046
    Epoch [98/100] Training Loss: 0.0464, F1 Score: 0.5760 | Validation Loss: 0.0428, F1 Score: 0.7996
    Epoch [99/100] Training Loss: 0.0463, F1 Score: 0.5646 | Validation Loss: 0.0426, F1 Score: 0.8046
    Epoch [100/100] Training Loss: 0.0462, F1 Score: 0.5748 | Validation Loss: 0.0426, F1 Score: 0.7996



    
![png](output_44_1.png)
    


    Training complete! Total time: 4462.33 seconds


# **Mappings Selector**


```python
# Build an indexed dictionary for the source ontology classes
# src_class is the file path to the JSON file containing the source ontology classes
indexed_dict_src = build_indexed_dict(src_class)

# Build an indexed dictionary for the target ontology classes
# tgt_class is the file path to the JSON file containing the target ontology classes
indexed_dict_tgt = build_indexed_dict(tgt_class)
```


```python
# Read the candidate pairs from a Candidates CSV file into a pandas DataFrame
df_embbedings = pd.read_csv(candidates_Prediction, index_col=0)

# Extract the 'SrcEntity' column (source entity indices) and convert it to a NumPy array of integers
tensor_term1 = df_embbedings['SrcEntity'].values.astype(int)

# Extract the 'TgtEntity' column (target entity indices) and convert it to a NumPy array of integers
tensor_term2 = df_embbedings['TgtEntity'].values.astype(int)

# Convert the source entity indices to a PyTorch LongTensor
src_entity_tensor_o = torch.from_numpy(tensor_term1).type(torch.LongTensor)

# Convert the target entity indices to a PyTorch LongTensor
tgt_entity_tenso_or = torch.from_numpy(tensor_term2).type(torch.LongTensor)
```


```python
# Select rows from the updated source embeddings based on the indices in src_entity_tensor_o
X1_tt = select_rows_by_index(embeddings_src, src_entity_tensor_o)

# Select rows from the original source embeddings based on the indices in src_entity_tensor_o
X2_tt = select_rows_by_index(x_src, src_entity_tensor_o)

# Select rows from the updated target embeddings based on the indices in tgt_entity_tenso_or
X3_tt = select_rows_by_index(embeddings_tgt, tgt_entity_tenso_or)

# Select rows from the original target embeddings based on the indices in tgt_entity_tenso_or
X4_tt = select_rows_by_index(x_tgt, tgt_entity_tenso_or)
```


```python
# Generate predictions for candidate mappings using the trained GatedCombination model
Prediction_with_candidates(
    model=trained_model,             # The trained GatedCombination model used to evaluate similarity
    X1_tt=X1_tt,                     # Updated source embeddings (after applying the GIT model)
    X2_tt=X2_tt,                     # Original source embeddings (before applying the GIT model)
    X3_tt=X3_tt,                     # Updated target embeddings (after applying the GIT model)
    X4_tt=X4_tt,                     # Original target embeddings (before applying the GIT model)
    src_entity_tensor_o=src_entity_tensor_o,  # Tensor of source entity indices used for evaluation
    tgt_entity_tensor_o=tgt_entity_tenso_or,  # Tensor of target entity indices used for evaluation
    indexed_dict_src=indexed_dict_src,        # Dictionary mapping source entity indices to their URIs
    indexed_dict_tgt=indexed_dict_tgt,        # Dictionary mapping target entity indices to their URIs
    all_predictions_path=all_predictions_path # Path to save all predictions with similarity scores in TSV format
)
```

    Predicting time: 66.99 seconds
    Predictions saved to /content/gdrive/My Drive/BioGITOM-VLDB/Experiments/ncit2doid/GIT_Ablation/Results/ncit2doid_all_predictions.tsv



```python
# Filter the highest scoring predictions from the predictions file and save the results to a new file
matching_results_df = filter_highest_predictions(
    all_predictions_path,  # Path to the file containing all predictions with scores for all candidate pairs
    prediction_path        # Path where the filtered predictions with highest scores will be saved
)
```

    Number of Positive predictions: 3213


# **Evaluation**

# Global metrics calculation


```python
# Retrieve the indices of the ignored classes (from source and target ontologies)
ignored_class_index = get_ignored_class_index(src_onto)  # Get ignored class indices from source ontology
ignored_class_index.update(get_ignored_class_index(tgt_onto))  # Update with ignored class indices from target ontology

# Read the predicted mappings from the prediction results file
preds = EntityMapping.read_table_mappings(prediction_path)

# Read the reference mappings from the ground truth test file
refs = ReferenceMapping.read_table_mappings(f"{dataset_dir}/refs_equiv/test.tsv")

# Filter the predicted mappings to remove any mappings that involve ignored classes
preds = remove_ignored_mappings(preds, ignored_class_index)

# Compute the precision, recall, and F1-score by comparing predictions with the reference mappings
results = AlignmentEvaluator.f1(preds, refs)

preds2 = [p.to_tuple() for p in preds]
refs2 = [r.to_tuple() for r in refs]

correct= len(set(preds2).intersection(set(refs2)))

print(f"Number of Correct Predictions: {correct}")

# Print the computed precision, recall, and F1-score metrics
print(results)
```

    Number of Correct Predictions: 2666
    {'P': 0.83, 'R': 0.813, 'F1': 0.821}


# Ranked-based metrics calculation


```python
# Read the candidate pairs from a Candidates CSV file into a pandas DataFrame
df_embbedings = pd.read_csv(candidates_Rank, index_col=0)

# Extract the 'SrcEntity' column (source entity indices) and convert it to a NumPy array of integers
tensor_term1 = df_embbedings['SrcEntity'].values.astype(int)

# Extract the 'TgtEntity' column (target entity indices) and convert it to a NumPy array of integers
tensor_term2 = df_embbedings['TgtEntity'].values.astype(int)

# Convert the source entity indices to a PyTorch LongTensor
src_entity_tensor_o = torch.from_numpy(tensor_term1).type(torch.LongTensor)

# Convert the target entity indices to a PyTorch LongTensor
tgt_entity_tenso_or = torch.from_numpy(tensor_term2).type(torch.LongTensor)
```


```python
# Select rows from the updated source embeddings based on the indices in src_entity_tensor_o
X1_tt = select_rows_by_index(embeddings_src, src_entity_tensor_o)

# Select rows from the original source embeddings based on the indices in src_entity_tensor_o
X2_tt = select_rows_by_index(x_src, src_entity_tensor_o)

# Select rows from the updated target embeddings based on the indices in tgt_entity_tenso_or
X3_tt = select_rows_by_index(embeddings_tgt, tgt_entity_tenso_or)

# Select rows from the original target embeddings based on the indices in tgt_entity_tenso_or
X4_tt = select_rows_by_index(x_tgt, tgt_entity_tenso_or)
```


```python
# Perform ranking-based predictions using the trained GatedCombination model
# Generate predictions for candidate mappings using the trained GatedCombination model
Prediction_with_candidates(
    model=trained_model,             # The trained GatedCombination model used to evaluate similarity
    X1_tt=X1_tt,                     # Updated source embeddings (after applying the GIT model)
    X2_tt=X2_tt,                     # Original source embeddings (before applying the GIT model)
    X3_tt=X3_tt,                     # Updated target embeddings (after applying the GIT model)
    X4_tt=X4_tt,                     # Original target embeddings (before applying the GIT model)
    src_entity_tensor_o=src_entity_tensor_o,  # Tensor of source entity indices used for evaluation
    tgt_entity_tensor_o=tgt_entity_tenso_or,  # Tensor of target entity indices used for evaluation
    indexed_dict_src=indexed_dict_src,        # Dictionary mapping source entity indices to their URIs
    indexed_dict_tgt=indexed_dict_tgt,        # Dictionary mapping target entity indices to their URIs
    all_predictions_path=all_predictions_path_ranked, # Path where the ranked predictions will be saved in TSV format
)
```

    Predicting time: 63.62 seconds
    Predictions saved to /content/gdrive/My Drive/BioGITOM-VLDB/Experiments/ncit2doid/GIT_Ablation/Results/ncit2doid_all_predictions_ranked.tsv



```python
# Compute MRR and Hits@k metrics
# This function evaluates the predicted rankings against the reference mappings
results = compute_mrr_and_hits(
    reference_file=test_cands,             # Reference file with true ranks
    predicted_file=all_predictions_path_ranked,             # File containing predicted rankings
    output_file=formatted_predictions_path,    # File path to save formatted predictions
    k_values=[1, 5, 10]                        # Evaluate Hits@1, Hits@5, and Hits@10
)

# Display the computed metrics
print("MRR and Hits@k Results:")
print(results)  # Output the Mean Reciprocal Rank (MRR) and Hits@k metrics
```

    MRR and Hits@k Results:
    {'MRR': 0.8334413964653022, 'Hits@k': {1: 0.7670731707317073, 5: 0.9173780487804878, 10: 0.9567073170731707}}



```python
# Call the ranking evaluation function, passing the path to the formatted predictions file.
# Ks specifies the evaluation levels, checking if the correct target is within the top K candidates.
results = ranking_eval(formatted_predictions_path, Ks=[1, 5, 10])
print("Ranking Evaluation Results at K=1, 5, and 10:")
print(results)
```

    Ranking Evaluation Results at K=1, 5, and 10:
    {'MRR': 0.8334413964653022, 'Hits@1': 0.7670731707317073, 'Hits@5': 0.9173780487804878, 'Hits@10': 0.9567073170731707}

