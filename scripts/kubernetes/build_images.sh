#!/usr/bin/env bash
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

# Read from shell configuration file

source ./scripts/kubernetes/.env.sh

source ./scripts/kubernetes/utils.sh

title "Using K8S"
# Build SINGA-Auto's images

# Docker build -t <label-of-docker-image>
# Docker build -f <path-to-dockerfile>

echo "using $1 docker files"

if [[ $1 = "dev" ]]
then
  title "Building SINGA-Auto Admin's image..."
  docker build -t $SINGA_AUTO_IMAGE_ADMIN:$SINGA_AUTO_VERSION -f ./dockerfiles/dev_dockerfiles/admin.Dockerfile \
      --build-arg DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
      --build-arg CONDA_ENVIORNMENT=$CONDA_ENVIORNMENT $PWD || exit 1
  title "Building SINGA-Auto Worker's image..."
  docker build -t $SINGA_AUTO_IMAGE_WORKER:$SINGA_AUTO_VERSION -f ./dockerfiles/dev_dockerfiles/worker.Dockerfile \
      --build-arg DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
      --build-arg CONDA_ENVIORNMENT=$CONDA_ENVIORNMENT $PWD || exit 1
  title "Building SINGA-Auto Predictor's image..."
  docker build -t $SINGA_AUTO_IMAGE_PREDICTOR:$SINGA_AUTO_VERSION -f ./dockerfiles/dev_dockerfiles/predictor.Dockerfile \
      --build-arg DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
      --build-arg CONDA_ENVIORNMENT=$CONDA_ENVIORNMENT $PWD || exit 1

else
  title "Building SINGA-Auto Admin's image..."
  docker build -t $SINGA_AUTO_IMAGE_ADMIN:$SINGA_AUTO_VERSION -f ./dockerfiles/admin.Dockerfile \
      --build-arg DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
      --build-arg CONDA_ENVIORNMENT=$CONDA_ENVIORNMENT $PWD || exit 1
  title "Building SINGA-Auto Worker's image..."
  docker build -t $SINGA_AUTO_IMAGE_WORKER:$SINGA_AUTO_VERSION -f ./dockerfiles/worker.Dockerfile \
      --build-arg DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
      --build-arg CONDA_ENVIORNMENT=$CONDA_ENVIORNMENT $PWD || exit 1
  title "Building SINGA-Auto Predictor's image..."
  docker build -t $SINGA_AUTO_IMAGE_PREDICTOR:$SINGA_AUTO_VERSION -f ./dockerfiles/predictor.Dockerfile \
      --build-arg DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
      --build-arg CONDA_ENVIORNMENT=$CONDA_ENVIORNMENT $PWD || exit 1
fi

title "Building SINGA-Auto Web Admin's image..."
docker build -t $SINGA_AUTO_IMAGE_WEB_ADMIN:$SINGA_AUTO_VERSION -f ./dockerfiles/web_admin.Dockerfile \
    --build-arg DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH $PWD || exit 1

echo "Finished building all SINGA-Auto's images successfully!"
