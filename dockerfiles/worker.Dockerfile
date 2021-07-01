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

FROM ubuntu:16.04

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# update and install dependencies
RUN apt-get update &&  \
    apt-get install -y \
      software-properties-common \
      wget \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y \
        make \
        git \
        curl \
        vim \
        vim-gnome \
    && apt-get install -y cmake=3.5.1-1ubuntu3 \
    && apt-get install -y \
        gcc-4.9 g++-4.9 gcc-4.9-base \
        gcc-4.8 g++-4.8 gcc-4.8-base \
        gcc-4.7 g++-4.7 gcc-4.7-base \
        gcc-4.6 g++-4.6 gcc-4.6-base \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 100

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
