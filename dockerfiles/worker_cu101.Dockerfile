#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

FROM nvidia/cuda:10.1-base-ubuntu16.04

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# `tensorflow-gpu` dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cuda-command-line-tools-10-1 \
      cuda-cufft-10-1 \
      cuda-curand-10-1 \
      cuda-cusolver-10-1 \
      cuda-cusparse-10-1 \      
      libcublas10=10.2.3.254-1 \
      libcublas-dev=10.2.3.254-1 \
      libcudnn7=7.6.4.38-1+cuda10.1 \
      libcudnn7-dev=7.6.4.38-1+cuda10.1  \
      libfreetype6-dev \
      libhdf5-serial-dev \
      libpng-dev \
      libgl1-mesa-glx \
      libsm6 \
      libxrender1 \
      libzmq3-dev \
      pkg-config \
      software-properties-common \
      unzip \
      lsb-core \
      && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# cuda-10.1 package install cublas in cuda-10.2
# call ldconfig to link them
RUN cp -r /usr/local/cuda-10.2/* /usr/local/cuda-10.1/ && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/ && \
    ldconfig /etc/ld.so.conf.d

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libnvinfer5=5.1.5-1+cuda10.1 \
      libnvinfer6=6.0.1-1+cuda10.1 \
      libnvinfer-dev=5.1.5-1+cuda10.1 \
      libnvinfer-dev=6.0.1-1+cuda10.1 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# install cuda/bin
# RUN mkdir -p /usr/local/cuda-10.1/bin
# COPY /usr/local/cuda-10.1/bin/ /usr/local/cuda-10.1/bin/

# Install conda with pip and python 3.6
ARG CONDA_ENVIORNMENT
RUN apt-get update --fix-missing && apt-get -y upgrade && \
    apt-get install -y curl bzip2 && \
    curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh && \
    conda create -y --name $CONDA_ENVIORNMENT python=3.6 && \
    conda clean --all --yes
ENV PATH /usr/local/envs/$CONDA_ENVIORNMENT/bin:$PATH

RUN pip install --upgrade pip
ENV PYTHONUNBUFFERED 1

ARG DOCKER_WORKDIR_PATH
RUN mkdir -p $DOCKER_WORKDIR_PATH
WORKDIR $DOCKER_WORKDIR_PATH
ENV PYTHONPATH $DOCKER_WORKDIR_PATH

# Install python dependencies
COPY singa_auto/ singa_auto/

RUN mkdir -p /root/.config/pip/
COPY ./.config/pip/pip.conf /root/.config/pip/pip.conf

COPY ./backup_lib/torch-1.6.0-cp36-cp36m-manylinux1_x86_64.whl /root/torch-1.6.0-cp36-cp36m-manylinux1_x86_64.whl
RUN pip install /root/torch-1.6.0-cp36-cp36m-manylinux1_x86_64.whl
COPY ./backup_lib/opencv_python-4.4.0.46-cp36-cp36m-manylinux2014_x86_64.whl /root/opencv_python-4.4.0.46-cp36-cp36m-manylinux2014_x86_64.whl
RUN pip install /root/opencv_python-4.4.0.46-cp36-cp36m-manylinux2014_x86_64.whl

RUN pip install -r singa_auto/requirements.txt
RUN pip install -r singa_auto/utils/requirements.txt
RUN pip install -r singa_auto/meta_store/requirements.txt
RUN pip install -r singa_auto/redis/requirements.txt
RUN pip install -r singa_auto/kafka/requirements.txt
RUN pip install -r singa_auto/advisor/requirements.txt
RUN pip install -r singa_auto/worker/requirements.txt

COPY scripts/ scripts/
RUN mkdir data/

CMD ["python", "scripts/start_worker.py"]
