FROM tensorflow/tensorflow:latest-gpu-py3-jupyter
ADD requirements.txt .
RUN pip install -r requirements.txt