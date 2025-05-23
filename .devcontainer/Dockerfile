FROM mcr.microsoft.com/devcontainers/python:3.9

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install \
    kedro==0.19.12 \
    kedro-mlflow \
    mlflow \
    ipykernel \
    jupyterlab \
    xgboost \
    catboost \
    pandas \
    matplotlib \
    seaborn

WORKDIR /workspaces/startupdelay-horizon-app