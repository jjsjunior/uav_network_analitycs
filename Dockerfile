FROM tensorflow/tensorflow:2.6.0-gpu

# install dependencies
RUN add-apt-repository ppa:deadsnakes/ppa -y ; exit 0
#RUN apt-get update                             # Commented to skip Orbit Testbed bug
RUN apt install python3.7 libsm6 libxext6 libxrender-dev -y			
RUN python3.7 -m pip install --upgrade pip
#RUN apt-get clean                              # Commented to skip Orbit Testbed bug

WORKDIR /app

# add files
COPY ./ .

# Install requirements
RUN pip3.7 install -r requirements.txt

#CMD [ "python3.7", "/app/detecao_pessoas.py", "--help", ]
