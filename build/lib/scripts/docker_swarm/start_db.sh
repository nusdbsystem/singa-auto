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
source ./scripts/docker_swarm/.env.sh
LOG_FILE_PATH=$PWD/$LOGS_DIR_PATH/start_db.log

source ./scripts/docker_swarm/utils.sh


title "Starting Singa-Auto's DB..."

PROD_MOUNT_DATA=$HOST_WORKDIR_PATH/$DATA_DIR_PATH:$DOCKER_WORKDIR_PATH/$DATA_DIR_PATH
PROD_MOUNT_PARAMS=$HOST_WORKDIR_PATH/$PARAMS_DIR_PATH:$DOCKER_WORKDIR_PATH/$PARAMS_DIR_PATH
PROD_MOUNT_LOGS=$HOST_WORKDIR_PATH/$LOGS_DIR_PATH:$DOCKER_WORKDIR_PATH/$LOGS_DIR_PATH
PROD_MOUNT_DB="$HOST_WORKDIR_PATH/$DB_DIR_PATH:/var/lib/postgresql/data"
#PROD_MOUNT_DB=HOST_WORKDIR_PATH/DB_DIR_PATH:"/var/lib/postgresql/data"

VOLUME_MOUNTS="-v $PROD_MOUNT_DB"


# docker container run flags info:
# --rm: container is removed when it exits
# (--rm will also remove anonymous volumes)
# -v == --volume: shared filesystems
# -e == --env: environment variable
# --name: name used to identify the container
# --network: default is docker bridge
# -p: expose and map port(s)

(docker run --rm --name $POSTGRES_HOST \
  --network $DOCKER_NETWORK \
  -e POSTGRES_HOST=$POSTGRES_HOST \
  $VOLUME_MOUNTS \
  -e POSTGRES_PORT=$POSTGRES_PORT \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -p $POSTGRES_EXT_PORT:$POSTGRES_PORT \
  $IMAGE_POSTGRES \
  &> $LOG_FILE_PATH) &

ensure_stable "Singa-Auto's DB" $LOG_FILE_PATH 20

echo "Creating Singa-Auto's PostgreSQL database & user..."
docker exec $POSTGRES_HOST psql -U postgres -c "CREATE DATABASE $POSTGRES_DB"
ensure_stable "Singa-Auto's DB create database" $LOG_FILE_PATH 5
docker exec $POSTGRES_HOST psql -U postgres -c "CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD'"
ensure_stable "Singa-Auto's DB create user" $LOG_FILE_PATH 2
