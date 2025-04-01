FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

RUN apt-get install ffmpeg git -y

COPY . .

RUN pip install --upgrade pip

RUN pip install --upgrade pip && \
    pip install -U diffusers[torch] torch torchvision torchaudio \
    tokenizers moshi torchtune torchao transformers \
    flask protobuf --upgrade accelerate huggingface_hub 

RUN pip install "silentcipher @ git+https://github.com/SesameAILabs/silentcipher@master"

ENV NO_TORCH_COMPILE=1

# Create directories
RUN mkdir -p inputs results

EXPOSE 8383

CMD ["python", "app.py"]