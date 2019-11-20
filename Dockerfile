FROM tensorflow/tensorflow:latest-gpu-py3-jupyter
ADD requirements.txt .
RUN pip install -r requirements.txt
RUN apt update
RUN apt install -y libsm6 libxext6
RUN apt-get install libxrender1