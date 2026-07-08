# ml-training-vertex

A machine-learning training project that predicts house sale prices (the classic
[Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
dataset) with a **LightGBM** regressor. The same pipeline can be run locally or
packaged as a container and executed on **Google Cloud Vertex AI** as a custom
training job, reading data from **BigQuery** and writing the trained model to
**Google Cloud Storage**.

## What it does

1. **Feature engineering** – loads the train/test tables, imputes missing values
   (neighbourhood-aware means and modes), drops low-variance / highly-correlated
   columns, maps ordinal categoricals to numeric scales, builds aggregate
   features (`TotalSF`, `TotalBath`, `TotalPorch`, …), one-hot encodes the rest
   and scales numeric columns with a `RobustScaler`.
2. **Model training** – tunes a `LGBMRegressor` with `RandomizedSearchCV`
   (5-fold CV, RMSE objective) over depth, learning rate, estimators,
   regularisation and sampling parameters. The target `SalePrice` is
   log-transformed.
3. **Evaluation** – reports MAE, RMSE, R² and cross-validated score on a held-out
   validation split.
4. **Model export** – serialises the fitted model with `joblib` and uploads it to
   a GCS bucket (`models-ml-testing-files/model-output`).

## Repository layout

| Path | Description |
| --- | --- |
| `full-local-pipeline.py` | End-to-end pipeline that runs entirely locally from the CSVs in `data/`. |
| `ml/trainer/task.py` | Vertex AI training entrypoint: reads from BigQuery, trains, and uploads the model to GCS. |
| `ml/Dockerfile` | Container image (based on the Vertex AI XGBoost training image) that packages the trainer. |
| `ml/trainer/launch.md` | `gcloud ai custom-jobs create` command to launch the training job on Vertex AI. |
| `data/train.csv`, `data/test.csv` | Local copies of the House Prices dataset. |
| `*.ipynb` | Exploratory / notebook versions of the local and GCP training workflows. |
| `pyproject.toml`, `uv.lock` | Project metadata and pinned dependencies (managed with [uv](https://github.com/astral-sh/uv)). |

## Tech stack

Python, pandas, scikit-learn, LightGBM (with XGBoost/CatBoost available),
BigQuery (`bigframes`), Google Cloud Storage, Docker and Vertex AI custom jobs.

## Running locally

```bash
uv sync
uv run python full-local-pipeline.py
```

## Running on Vertex AI

Build and push the training image, then launch a custom job:

```bash
# Build & push
gcloud auth configure-docker europe-west1-docker.pkg.dev
IMAGE_URI=europe-west1-docker.pkg.dev/vertexai-ml-pipelines/train-ml/lgbm-model:latest
docker build -t $IMAGE_URI ml/
docker push $IMAGE_URI

# Launch the job (see ml/trainer/launch.md for the full command)
gcloud ai custom-jobs create \
  --project=vertexai-ml-pipelines \
  --region=europe-west1 \
  --display-name=house-prices-model-train \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=$IMAGE_URI
```



## Commands to the jupyterlab terminal


    #jupyter nbconvert final-complete-training.ipynb --to python
    #mkdir ml
    #mkdir ml/trainer
    #cp final-complete-training.py ml/trainer/
    #gcloud auth configure-docker europe-west1-docker.pkg.dev
    #IMAGE_URI=europe-west1-docker.pkg.dev/vertexai-ml-pipelines/train-ml/lgbm-model:latest
    #docker push $IMAGE_URI