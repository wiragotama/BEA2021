FROM pytorch/pytorch
FROM nvidia/cuda:10.2-base
FROM allennlp/allennlp:v0.9.0

COPY requirements.txt /script/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /script/requirements.txt

RUN apt-get update
RUN apt-get install vim -y

WORKDIR /

EXPOSE 8080
ENTRYPOINT ["/usr/bin/env"]
CMD ["/bin/bash"]
