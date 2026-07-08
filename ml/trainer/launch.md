# Comandos para ejecutar el entrenamiento

#not ejecuted yet...

  PROJECT_ID="vertexai-ml-pipelines"
  REGION="europe-west1"
  JOB_NAME="house-prices-model-train2"
  IMAGE_URI="europe-west1-docker.pkg.dev/${PROJECT_ID}/train-ml/lgbm-model:latest"
  BUCKET_URI="gs://models-ml-testing-files/model-output"

  gcloud ai custom-jobs create \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --display-name="${JOB_NAME}" \
    --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri="${IMAGE_URI}" \
    --model_dir="${BUCKET_URI}/artifacts/${JOB_NAME}"