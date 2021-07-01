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

FROM nvidia/cuda:10.0-base-ubuntu16.04

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# `tensorflow-gpu` dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends  --allow-unauthenticated\
      build-essential \
      cuda-command-line-tools-10-0 \
      cuda-cublas-dev-10-0 \
      cuda-cudart-dev-10-0 \
      cuda-cufft-dev-10-0 \
      cuda-curand-dev-10-0 \
      cuda-cusolver-dev-10-0 \
      cuda-cusparse-dev-10-0 \
      libcudnn7=7.5.1.10-1+cuda10.0 \
      libfreetype6-dev \
      libhdf5-serial-dev \
      libnccl-dev=2.4.7-1+cuda10.0 \
      libnccl2=2.4.7-1+cuda10.0 \
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

RUN apt-get update && \
    apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda10.0 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      libnvinfer5=5.1.5-1+cuda10.0 \
      libnvinfer6=6.0.1-1+cuda10.0 \
      libnvinfer7=7.0.0-1+cuda10.0 \
      libnvinfer-dev=5.1.5-1+cuda10.0 \
      libnvinfer-dev=6.0.1-1+cuda10.0 \
      libnvinfer-dev=7.0.0-1+cuda10.0 \
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

RUN pip install -r singa_auto/requirements.txt
RUN pip install -r singa_auto/utils/requirements.txt
RUN pip install -r singa_auto/meta_store/requirements.txt
RUN pip install -r singa_auto/redis/requirements.txt
RUN pip install -r singa_auto/kafka/requirements.txt
RUN pip install -r singa_auto/advisor/requirements.txt

COPY scripts/ scripts/
RUN mkdir data/

CMD ["python", "scripts/start_worker.py"]
