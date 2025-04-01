FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential
RUN apt-get install -y ffmpeg git wget curl

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Add conda to PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Create conda environment with Python 3.10
RUN conda create -y -n tts python=3.10 && \
    conda clean -ya

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "tts", "/bin/bash", "-c"]

# Set environment activation on interactive shells
RUN echo "conda activate tts" >> ~/.bashrc

COPY . .

# Install Python dependencies within conda environment
RUN conda run -n tts pip install --upgrade pip && \
    conda run -n tts pip install -U diffusers[torch] torch torchvision torchaudio \
    tokenizers moshi torchtune torchao transformers \
    flask protobuf --upgrade accelerate huggingface_hub

RUN conda run -n tts pip install "silentcipher @ git+https://github.com/SesameAILabs/silentcipher@master"

ENV NO_TORCH_COMPILE=1

# Create directories
RUN mkdir -p inputs results

EXPOSE 8383

# Use conda environment when running the application
CMD ["conda", "run", "--no-capture-output", "-n", "tts", "python", "app.py"]