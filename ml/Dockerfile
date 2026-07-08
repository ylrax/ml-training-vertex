FROM europe-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1:latest

WORKDIR /

COPY trainer /trainer

RUN pip install lightgbm bigframes joblib


ENTRYPOINT ["python", "-m", "trainer.task"]
