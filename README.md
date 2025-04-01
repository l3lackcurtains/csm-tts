### Download the model

```bash
huggingface-cli download sesame/csm-1b --local-dir ./local_models/sesame/csm-1b
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir ./local_models/meta-llama/Llama-3.2-1B
```

#### Build the docker image and push to gcr

```bash
gcloud auth configure-docker
docker login gcr.io

docker build -t silk-csm .
docker tag silk-csm gcr.io/citric-lead-450721-v2/silk-csm:1.0.0
docker push gcr.io/citric-lead-450721-v2/silk-csm:1.0.0


```
