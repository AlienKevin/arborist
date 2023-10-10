# Use Rust as the base image
FROM rust:1.73.0-bookworm AS build-env

ARG TARGETARCH

# Install basic utilities and Python
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends make python3 python3-dev python3-venv curl gcc g++ libclang-dev pigz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment and activate it
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install pip in the virtual environment
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install Python dependencies
RUN pip install numpy pandas matplotlib

# Install Go version 1.20
RUN curl -O https://dl.google.com/go/go1.20.linux-${TARGETARCH}.tar.gz && \
    tar -xvf go1.20.linux-${TARGETARCH}.tar.gz && \
    mv go /usr/local && \
    rm go1.20.linux-${TARGETARCH}.tar.gz

# Set Go Environment Variables
ENV GOROOT=/usr/local/go
ENV GOPATH=$HOME/go
ENV PATH=$GOPATH/bin:$GOROOT/bin:$PATH

# Install rustfmt and bindgen
RUN rustup component add rustfmt && cargo install bindgen-cli

# Copy everything from the current directory to the workspace
COPY . /workspace

# Make the workspace directory
WORKDIR /workspace

# CMD instruction to keep the container running
CMD ["/bin/bash"]
