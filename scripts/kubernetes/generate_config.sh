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

if [ $HOST_WORKDIR_PATH ];then
	echo "HOST_WORKDIR_PATH is exist, and echo to = $HOST_WORKDIR_PATH"
else
	export HOST_WORKDIR_PATH=$PWD
fi

source $HOST_WORKDIR_PATH/scripts/kubernetes/.env.sh

python3 $HOST_WORKDIR_PATH/scripts/kubernetes/create_config.py \
$POSTGRES_PASSWORD \
$SUPERADMIN_PASSWORD \
$APP_SECRET \
$KUBERNETES_NETWORK \
$KUBERNETES_ADVERTISE_ADDR \
$SINGA_AUTO_VERSION \
$SINGA_AUTO_ADDR \
$ADMIN_EXT_PORT \
$WEB_ADMIN_EXT_PORT \
$POSTGRES_EXT_PORT \
$REDIS_EXT_PORT \
$ZOOKEEPER_EXT_PORT \
$KAFKA_EXT_PORT \
$HOST_WORKDIR_PATH \
$APP_MODE \
$POSTGRES_DUMP_FILE_PATH \
$DOCKER_NODE_LABEL_AVAILABLE_GPUS \
$DOCKER_NODE_LABEL_NUM_SERVICES \
$POSTGRES_USER \
$POSTGRES_DB \
$POSTGRES_HOST \
$POSTGRES_PORT \
$ADMIN_HOST \
$ADMIN_PORT \
$REDIS_HOST \
$REDIS_PORT \
$PREDICTOR_PORT \
$WEB_ADMIN_HOST \
$ZOOKEEPER_HOST \
$ZOOKEEPER_PORT \
$KAFKA_HOST \
$KAFKA_PORT \
$DOCKER_WORKDIR_PATH \
$DATA_DIR_PATH \
$LOGS_DIR_PATH \
$PARAMS_DIR_PATH \
$CONDA_ENVIORNMENT \
$WORKDIR_PATH \
$SINGA_AUTO_IMAGE_ADMIN \
$SINGA_AUTO_IMAGE_WEB_ADMIN \
$SINGA_AUTO_IMAGE_WORKER \
$SINGA_AUTO_IMAGE_PREDICTOR \
$IMAGE_POSTGRES \
$IMAGE_REDIS \
$IMAGE_ZOOKEEPER \
$IMAGE_KAFKA \
$PYTHONPATH \
$PYTHONUNBUFFERED \
$CONTAINER_MODE \
$CLUSTER_MODE \
$DB_DIR_PATH \
$INGRESS_NAME \
$INGRESS_EXT_PORT \
$REDIS_PASSWORD \
$LOGSTASH_HOST \
$LOGSTASH_PORT \
$ES_HOST \
$ES_PORT \
$ES_NODE_PORT \
$KIBANA_HOST \
$KIBANA_PORT \
$LOGSTASH_DOCKER_WORKDIR_PATH \
$KIBANA_DOCKER_WORKDIR_PATH \
$SINGA_AUTO_IMAGE_LOGSTASH \
$IMAGE_KIBANA \
$SINGA_AUTO_IMAGE_ES \
$KIBANA_EXT_PORT \
$SINGA_AUTO_IMAGE_SPARKAPP \
$SPAEK_DOCKER_JARS_PATH \
$ES_DOCKER_WORKDIR_PATH
