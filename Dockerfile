# using python 3.9.13-slim image from dockerhub to create dev container
FROM python:3.10

# creating our workdir Modern-Data-Analytics - named /app by convention for workdir in docker container. 
# few lines down - all files from our workdir Modern-Data-Analytics will be copied to this workdir. 
WORKDIR /app

# system installments, especially needed for packages like numpy, pandas which have native extensions
RUN apt-get update && apt-get install -y build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# copying all dependencies and installing them in dev container
COPY requirements.txt .
RUN pip install -r requirements.txt

# Upgrade MLflow to 2.22.0
RUN pip install --upgrade mlflow==2.22.0 

# copyinig all files from Modern-Data-Analytics to the workdir app/ in docker dev container
COPY . .

# setting port where docker container will listen - to run a jypter notebooks server which we can access via browser: 
# http://localhost:8888
# for windows via docker desktop: http://127.0.0.1:8888
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
 