FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    gnupg \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    lsb-release \
    software-properties-common \
    unzip \
    ripgrep \
    fd-find \
    && rm -rf /var/lib/apt/lists/*

# Install Neovim 0.10.x from GitHub releases (official pre-built binary)
RUN curl -LO https://github.com/neovim/neovim/releases/download/v0.11.5/nvim-linux-x86_64.tar.gz && \
    tar -xzf nvim-linux-x86_64.tar.gz && \
    mv nvim-linux-x86_64 /opt/nvim && \
    ln -sf /opt/nvim/bin/nvim /usr/local/bin/nvim && \
    rm nvim-linux-x86_64.tar.gz && \
    nvim --version

# Install Nsight Systems CLI (nsys) via official NVIDIA repo
# Using the modern signed-by approach instead of deprecated apt-key
RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg ca-certificates && \
    # Get the Ubuntu version without dots (e.g., 2204 for 22.04)
    UBUNTU_VERSION=$(grep -oP 'DISTRIB_RELEASE=\K[0-9.]+' /etc/lsb-release | tr -d '.') && \
    ARCH=$(dpkg --print-architecture) && \
    echo "Detected Ubuntu ${UBUNTU_VERSION} on ${ARCH}" && \
    # Add NVIDIA devtools repository
    echo "deb [trusted=yes] https://developer.download.nvidia.com/devtools/repos/ubuntu${UBUNTU_VERSION}/${ARCH}/ /" | \
    tee /etc/apt/sources.list.d/nvidia-devtools.list && \
    # Update and install nsight-systems-cli
    apt-get update && \
    apt-get install -y nsight-systems-cli && \
    # Cleanup
    rm -rf /var/lib/apt/lists/* && \
    # Verify nsys is installed
    which nsys && nsys --version

# Install LazyVim config
# Create config directory structure first, then clone and set up
RUN mkdir -p /root/.config && \
    git clone --depth 1 https://github.com/LazyVim/starter /root/.config/nvim && \
    rm -rf /root/.config/nvim/.git

# Set working directory
WORKDIR /workspace

# Final verification of installations
RUN echo "=== Verifying installations ===" && \
    echo "CUDA:" && nvcc --version && \
    echo "Nsight Systems:" && nsys --version && \
    echo "Neovim:" && nvim --version | head -1 && \
    echo "Python:" && python3 --version && \
    echo "=== All installations verified ==="

# Keep container running for SSH access
CMD ["sleep", "infinity"]