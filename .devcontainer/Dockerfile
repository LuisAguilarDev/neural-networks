FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

ENV PYTHONUNBUFFERED 1

WORKDIR /workspace

# Copy your requirements file (if you have one)
COPY requirements.txt .

# Install dependencies from the requirements file
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    sudo apt-get update && sudo apt-get install graphviz xdg-utils