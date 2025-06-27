FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /ethos
ENV TZ=America/New_York
ARG DEBIAN_FRONTEND=noninteractive

# Add a /.local/bin and /.local/lib directories to allow editable python
# installs by any user
RUN mkdir -p -m 777 /.local/bin /.local/lib

# Add the deadsnakes repository to get python 3.11
RUN apt update && apt install -y \
        software-properties-common && \
        add-apt-repository ppa:deadsnakes/ppa

# Install python and other useful programs
RUN apt update && apt install -y \
        git \
        vim \
        curl \
        python-is-python3 \
        python3-pip \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-venv \
    && \
    apt clean && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Add the bashrc to start up the container correctly for local development
COPY container_bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

RUN python -m ensurepip -U && \
    pip install -U glances pip setuptools wheel

# Install Python requirements using the compiled version of the requirements
COPY requirements.txt requirements.txt
RUN pip install --ignore-installed -r requirements.txt && \
    rm requirements.txt
