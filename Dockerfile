# Use NVIDIA CUDA base image with PyTorch
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric and related packages
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
RUN pip install torch-geometric

# Install scientific computing packages
RUN pip install numpy>=1.21.0 scipy>=1.9.0 scikit-learn>=1.1.0

# Install data processing and visualization packages
RUN pip install matplotlib>=3.5.0 pyvista>=0.38.0 h5py>=3.7.0 tqdm>=4.64.0

# Install configuration and utility packages
RUN pip install PyYAML>=6.0 einops>=0.6.0 jaxtyping>=0.2.0

# Install NVIDIA Modulus (for Grid models)
RUN pip install modulus>=23.0.0

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/res/trained_model
RUN mkdir -p /workspace/res/visualizations

# Set default command
CMD ["/bin/bash"] 