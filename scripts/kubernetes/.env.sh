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

# Cluster Mode for SINGA-auto
export CLUSTER_MODE=SINGLE # CLUSTER or SINGLE

# Core secrets for SINGA-auto - change these in production!
export POSTGRES_PASSWORD=singa_auto
export SUPERADMIN_PASSWORD=singa_auto
export APP_SECRET=singa_auto

# Core external configuration for SINGA-auto
export KUBERNETES_NETWORK=singa_auto
export DOCKER_SWARM_ADVERTISE_ADDR=127.0.0.1
export SINGA_AUTO_VERSION=dev_1.0
export SINGA_AUTO_ADDR=127.0.0.1
export ADMIN_EXT_PORT=3000
export WEB_ADMIN_EXT_PORT=3001
export POSTGRES_EXT_PORT=5433
export REDIS_EXT_PORT=6380
export ZOOKEEPER_EXT_PORT=2181
export KAFKA_EXT_PORT=9092
export HOST_WORKDIR_PATH=$PWD
export APP_MODE=DEV # DEV or PROD
export POSTGRES_DUMP_FILE_PATH=$PWD/db_dump.sql # PostgreSQL database dump file
export DOCKER_NODE_LABEL_AVAILABLE_GPUS=available_gpus # Docker node label for no. of services currently running on the node
export DOCKER_NODE_LABEL_NUM_SERVICES=num_services # Docker node label for no. of services currently running on the node

# Internal credentials for SINGA-auto's components
export POSTGRES_USER=singa_auto
export POSTGRES_DB=singa_auto
export POSTGRES_STOLON_PASSWD=cmFmaWtpCg==  # The Passwd for stolon, base64 encode

# Internal hosts & ports and configuration for SINGA-auto's components
export POSTGRES_HOST=singa-auto-db
export POSTGRES_PORT=5432
export ADMIN_HOST=singa-auto-admin
export ADMIN_PORT=3000
export REDIS_HOST=singa-auto-redis
export REDIS_PORT=6379
export PREDICTOR_PORT=3003
export WEB_ADMIN_HOST=singa-auto-admin-web
export ZOOKEEPER_HOST=singa-auto-zookeeper
export ZOOKEEPER_PORT=2181
export KAFKA_HOST=singa-auto-kafka
export KAFKA_PORT=9092
export DOCKER_WORKDIR_PATH=/root
export DB_DIR_PATH=db/data
export DATA_DIR_PATH=data # Shares a data folder with containers, relative to workdir
export LOGS_DIR_PATH=logs # Shares a folder with containers that stores components' logs, relative to workdir
export PARAMS_DIR_PATH=params # Shares a folder with containers that stores model parameters, relative to workdir
export CONDA_ENVIORNMENT=singa_auto
export WORKDIR_PATH=$HOST_WORKDIR_PATH # Specifying workdir if Python programs are run natively

# Docker images for SINGA-Auto's custom components
export SINGA_AUTO_IMAGE_ADMIN=singa_auto/singa_auto_admin
export SINGA_AUTO_IMAGE_WEB_ADMIN=singa_auto/singa_auto_admin_web
export SINGA_AUTO_IMAGE_WORKER=singa_auto/singa_auto_worker
export SINGA_AUTO_IMAGE_PREDICTOR=singa_auto/singa_auto_predictor
export SINGA_AUTO_IMAGE_STOLON=sorintlab/stolon:master-pg10
export SINGA_AUTO_IMAGE_TEST=singa_auto/singa_auto_test

# Docker images for dependent services
export IMAGE_POSTGRES=postgres:10.5-alpine
export IMAGE_REDIS=redis:5.0.3-alpine3.8
export IMAGE_ZOOKEEPER=zookeeper:3.5
export IMAGE_KAFKA=wurstmeister/kafka:2.12-2.1.1

# Utility configuration
export PYTHONPATH=$PWD # Ensures that `singa_auto` module can be imported at project root
export PYTHONUNBUFFERED=1 # Ensures logs from Python appear instantly

export CONTAINER_MODE=K8S

if [ "$CLUSTER_MODE" = "CLUSTER" ]; then
    export POSTGRES_HOST=stolon-proxy-service
    export NFS_HOST_IP=127.0.0.1       # NFS Host IP - if used nfs as pv for database storage
    export RUN_DIR_PATH=run            # Shares a folder with containers that stores components' running info, relative to workdir
fi
