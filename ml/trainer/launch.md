# Intro de ejecución

Hay dos formas de ejecutar el contenedor propio y ambas opciones ejecutan tu mismo contenedor sobre la infraestructura gestionada. La diferencia no está en cómo corre el entrenamiento, sino en qué pasa alrededor de él.

- *Trabajo personalizado (CustomJob)*
Es la unidad más básica de entrenamiento personalizado. Le dices: "ejecuta esta imagen de Docker en este hardware, una vez". Configuras el worker pool spec (tipo de máquina, GPUs/aceleradores, número de réplicas, la imagen del contenedor y sus argumentos) y nada más.

No registra el modelo automáticamente en el Model Registry.
No tiene integración nativa con datasets gestionados de Vertex.
Tú te encargas de dónde quedan los artefactos (normalmente los escribes tú a Cloud Storage desde el propio contenedor).

Es la opción más directa y transparente. Ideal para experimentación, o cuando la orquestación la manejas tú por fuera.

- *Canalización de entrenamiento (TrainingPipeline)*
Es una orquestación de nivel superior que envuelve un trabajo personalizado (o uno de ajuste de hiperparámetros) y le añade pasos automáticos alrededor:

Puede inyectar un dataset gestionado de Vertex en tu entrenamiento (hace la división train/test y te lo pasa al contenedor).
Puede subir y registrar automáticamente el modelo resultante en el Vertex AI Model Registry (parámetro modelToUpload) al terminar.

Resumen:  

Simplicando mucho, una canalización de entrenamiento (TrainingPipeline) es un una ampliación del otro caso: envuelve un nuevo trabajo de entrenamiento (CustomJob o de ajuste de hiperparámetros) con dataset + registro de modelo.
Los dos añadidos (dataset y registro) son opcionales. Si creas una TrainingPipeline sin ninguno de los dos, obtienes algo casi idéntico a lanzar un CustomJob directamente; solo cambia el tipo de recurso y el punto de entrada.

## Comandos para ejecutar el entrenamiento

De momento solo existe un comando para el entrenamiento con custom job, no hay comando para canalización (no existe _gcloud ai training-pipelines create_). Se crea por API REST o por el SDK de Python.
Esos son los comandos para la ejecución del custom Job:


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
