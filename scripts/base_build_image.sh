


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

# Docker build -t <label-of-docker-image>
# Docker build -f <path-to-dockerfile>

source $HOST_WORKDIR_PATH/scripts/base_utils.sh

# web admin is the same for both dev or prod
title "Building SINGA-Auto Web Admin's image..."
docker build -t $SINGA_AUTO_IMAGE_WEB_ADMIN:$SINGA_AUTO_VERSION -f ./dockerfiles/web_admin.Dockerfile \
    --build-arg DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH $PWD || exit 1

# in prod model, the singa-auto project will be install inside docker container.
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

# those is the log monitor images, not necessary to the system, ingore the error if there is
title "Building SINGA-Auto LogStash's image..."
docker build -t $SINGA_AUTO_IMAGE_LOGSTASH:$SINGA_AUTO_VERSION -f ./log_minitor/dockerfiles/logstash.Dockerfile \
      --build-arg LOGSTASH_DOCKER_WORKDIR_PATH=$LOGSTASH_DOCKER_WORKDIR_PATH $PWD

title "Building SINGA-Auto ES's image..."
docker build -t $SINGA_AUTO_IMAGE_ES:$SINGA_AUTO_VERSION -f ./log_minitor/dockerfiles/elasticsearch.Dockerfile \
      --build-arg ES_DOCKER_WORKDIR_PATH=$ES_DOCKER_WORKDIR_PATH $PWD

echo "Finished building all SINGA-Auto's images successfully!"
