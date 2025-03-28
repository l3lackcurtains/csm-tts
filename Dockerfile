FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

COPY local_models/ /app/local_models/
COPY segments/ /app/segments/

COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

ENV NO_TORCH_COMPILE=1

# Create directories
RUN mkdir -p inputs results

EXPOSE 8383

CMD ["python", "app.py"]