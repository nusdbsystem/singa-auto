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


# Core secrets for SINGA-auto - change these in production!
export POSTGRES_PASSWORD=singa_auto
export SUPERADMIN_PASSWORD=singa_auto
export APP_SECRET=singa_auto

# Core external configuration for SINGA-auto
# SINGA_AUTO_ADDR need to be changed when do the deployments,
# this is provided by using dockerswarm env or k8s env
export SINGA_AUTO_VERSION=$2
export SINGA_AUTO_ADDR=$1

export ADMIN_EXT_PORT=3000
export WEB_ADMIN_EXT_PORT=3001
export POSTGRES_EXT_PORT=5433
export REDIS_EXT_PORT=6380
export ZOOKEEPER_EXT_PORT=2181
export KAFKA_EXT_PORT=9092

if [ $APP_MODE ];then
	echo "APP_MODE is exist, and echo to = $APP_MODE"
else
	export APP_MODE=DEV # DEV or PROD
fi

export POSTGRES_DUMP_FILE_PATH=$PWD/db_dump.sql # PostgreSQL database dump file
export DOCKER_NODE_LABEL_AVAILABLE_GPUS=available_gpus # Docker node label for no. of services currently running on the node
export DOCKER_NODE_LABEL_NUM_SERVICES=num_services # Docker node label for no. of services currently running on the node

# Internal hosts & ports and configuration for SINGA-auto's components

export POSTGRES_USER=singa_auto
export POSTGRES_DB=singa_auto

export POSTGRES_HOST=singa-auto-db
export POSTGRES_PORT=5432
export ADMIN_HOST=singa-auto-admin
export ADMIN_PORT=3000
export REDIS_HOST=singa-auto-redis
export REDIS_PORT=6379
export REDIS_PASSWORD=singa_auto
export PREDICTOR_PORT=3003
export WEB_ADMIN_HOST=singa-auto-admin-web
export ZOOKEEPER_HOST=singa-auto-zookeeper
export ZOOKEEPER_PORT=2181
export KAFKA_HOST=singa-auto-kafka
export KAFKA_PORT=9092
export LOGSTASH_HOST=singa-auto-logstash
export LOGSTASH_PORT=9600
export ES_HOST=elasticsearch
export ES_PORT=9200
export ES_NODE_PORT=9300
export KIBANA_HOST=singa-auto-kibana
export KIBANA_PORT=5601
export KIBANA_EXT_PORT=31009


export DOCKER_WORKDIR_PATH=/root
export DB_DIR_ROOT=db
export DB_DIR_PATH=db/data
export DATA_DIR_PATH=data # Shares a data folder with containers, relative to workdir
export LOGS_DIR_PATH=logs # Shares a folder with containers that stores components' logs, relative to workdir
export PARAMS_DIR_PATH=params # Shares a folder with containers that stores model parameters, relative to workdir

export CONDA_ENVIORNMENT=singa_auto
export WORKDIR_PATH=$HOST_WORKDIR_PATH # Specifying workdir if Python programs are run natively


export LOGSTASH_DOCKER_WORKDIR_PATH=/usr/share/logstash
export KIBANA_DOCKER_WORKDIR_PATH=/usr/share/kibana
export ES_DOCKER_WORKDIR_PATH=/usr/share/elasticsearch


# Docker images for SINGA-Auto's custom components
export SINGA_AUTO_IMAGE_ADMIN=singaauto/singa_auto_admin
export SINGA_AUTO_IMAGE_WEB_ADMIN=singaauto/singa_auto_admin_web
export SINGA_AUTO_IMAGE_WORKER=singaauto/singa_auto_worker
export SINGA_AUTO_IMAGE_PREDICTOR=singaauto/singa_auto_predictor
export SINGA_AUTO_IMAGE_LOGSTASH=singaauto/singa_auto_logstash
export SINGA_AUTO_IMAGE_ES=singaauto/singa_auto_es

export SINGA_AUTO_IMAGE_TEST=singaauto/singa_auto_test

# Docker images for dependent services
export IMAGE_POSTGRES=postgres:10.5-alpine
export IMAGE_REDIS=redis:5.0.3-alpine3.8
export IMAGE_ZOOKEEPER=zookeeper:3.5
export IMAGE_KAFKA=wurstmeister/kafka:2.12-2.1.1

export IMAGE_KIBANA=kibana:7.7.0
export IMAGE_ES=docker.elastic.co/elasticsearch/elasticsearch:7.7.0


# Utility configuration
export PYTHONPATH=$PWD # Ensures that `singa_auto` module can be imported at project root
export PYTHONUNBUFFERED=1 # Ensures logs from Python appear instantly
