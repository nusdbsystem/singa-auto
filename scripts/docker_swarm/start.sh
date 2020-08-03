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

# docker container run flags info:
# --rm: container is removed when it exits
# (--rm will also remove anonymous volumes)
# -v == --volume: shared filesystems
# -e == --env: environment variable
# --name: name used to identify the container
# --network: default is docker bridge
# -p: expose and map port(s)


help()
{
    cat <<- EOF
    Desc: used to launch the services
    Usage: start all services using: bash scripts/docker_swarm/start.sh
           start redis using: bash scripts/docker_swarm/start.sh redis
           start db using: bash scripts/docker_swarm/start.sh db
           start kafka using: bash scripts/docker_swarm/start.sh kafka
           start admin using: bash scripts/docker_swarm/start.sh admin
           start web using: bash scripts/docker_swarm/start.sh web
    Author: naili
EOF
}

create_docker_swarm()
{

      title "Creating Docker swarm for SINGA-Auto..."
      # docker swarm leave
      docker swarm init --advertise-addr $DOCKER_SWARM_ADVERTISE_ADDR \
          || >&2 echo "Failed to init Docker swarm - continuing..."
      docker network create $DOCKER_NETWORK -d overlay --attachable --scope=swarm \
          || >&2 echo  "Failed to create Docker network for swarm - continuing..."
}


start_admin()
{
      title "Starting SINGA-Auto's Admin..."

      LOG_FILE_PATH=$PWD/$LOGS_DIR_PATH/start_admin.log
      PROD_MOUNT_DATA=$PWD/$DATA_DIR_PATH:$DOCKER_WORKDIR_PATH/$DATA_DIR_PATH
      PROD_MOUNT_PARAMS=$PWD/$PARAMS_DIR_PATH:$DOCKER_WORKDIR_PATH/$PARAMS_DIR_PATH
      PROD_MOUNT_LOGS=$PWD/$LOGS_DIR_PATH:$DOCKER_WORKDIR_PATH/$LOGS_DIR_PATH

      # Mount whole project folder with code for dev for shorter iterations
      if [ $APP_MODE = "DEV" ]; then
        VOLUME_MOUNTS="-v $PWD:$DOCKER_WORKDIR_PATH"
      else
        VOLUME_MOUNTS="-v $PROD_MOUNT_DATA -v $PROD_MOUNT_PARAMS -v $PROD_MOUNT_LOGS"
      fi

      (docker run --rm --name $ADMIN_HOST \
        --network $DOCKER_NETWORK \
        -e POSTGRES_HOST=$POSTGRES_HOST \
        -e POSTGRES_PORT=$POSTGRES_PORT \
        -e POSTGRES_USER=$POSTGRES_USER \
        -e POSTGRES_DB=$POSTGRES_DB \
        -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
        -e SUPERADMIN_PASSWORD=$SUPERADMIN_PASSWORD \
        -e ADMIN_HOST=$ADMIN_HOST \
        -e ADMIN_PORT=$ADMIN_PORT \
        -e REDIS_HOST=$REDIS_HOST \
        -e REDIS_PORT=$REDIS_PORT \
        -e REDIS_PASSWORD=$REDIS_PASSWORD \
        -e KAFKA_HOST=$KAFKA_HOST \
        -e KAFKA_PORT=$KAFKA_PORT \
        -e PREDICTOR_PORT=$PREDICTOR_PORT \
        -e SINGA_AUTO_ADDR=$SINGA_AUTO_ADDR \
        -e SINGA_AUTO_IMAGE_WORKER=$SINGA_AUTO_IMAGE_WORKER \
        -e SINGA_AUTO_IMAGE_PREDICTOR=$SINGA_AUTO_IMAGE_PREDICTOR \
        -e SINGA_AUTO_VERSION=$SINGA_AUTO_VERSION \
        -e DOCKER_WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
        -e WORKDIR_PATH=$DOCKER_WORKDIR_PATH \
        -e HOST_WORKDIR_PATH=$HOST_WORKDIR_PATH \
        -e DATA_DIR_PATH=$DATA_DIR_PATH \
        -e PARAMS_DIR_PATH=$PARAMS_DIR_PATH \
        -e LOGS_DIR_PATH=$LOGS_DIR_PATH \
        -e APP_MODE=$APP_MODE \
        -e CONTAINER_MODE=$CONTAINER_MODE \
        -v /var/run/docker.sock:/var/run/docker.sock \
        $VOLUME_MOUNTS \
        -p $ADMIN_EXT_PORT:$ADMIN_PORT \
        $SINGA_AUTO_IMAGE_ADMIN:$SINGA_AUTO_VERSION \
        &> $LOG_FILE_PATH) &
        ensure_stable "SINGA-Auto's Admin" $LOG_FILE_PATH 5
}

start_db()
{
        title "Starting SINGA-Auto's DB..."
        LOG_FILE_PATH=$PWD/$LOGS_DIR_PATH/start_db.log
        PROD_MOUNT_DATA=$PWD/$DATA_DIR_PATH:$DOCKER_WORKDIR_PATH/$DATA_DIR_PATH
        PROD_MOUNT_PARAMS=$PWD/$PARAMS_DIR_PATH:$DOCKER_WORKDIR_PATH/$PARAMS_DIR_PATH
        PROD_MOUNT_LOGS=$PWD/$LOGS_DIR_PATH:$DOCKER_WORKDIR_PATH/$LOGS_DIR_PATH
        PROD_MOUNT_DB="$PWD/$DB_DIR_PATH:/var/lib/postgresql/data"

        VOLUME_MOUNTS="-v $PROD_MOUNT_DB"

        (docker run --rm --name $POSTGRES_HOST \
          --network $DOCKER_NETWORK \
          -e POSTGRES_HOST=$POSTGRES_HOST \
          $VOLUME_MOUNTS \
          -e POSTGRES_PORT=$POSTGRES_PORT \
          -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
          -p $POSTGRES_EXT_PORT:$POSTGRES_PORT \
          $IMAGE_POSTGRES \
          &> $LOG_FILE_PATH) &

        echo "Creating SINGA-Auto's PostgreSQL database & user..."
        # try 6 times
        for val in {1..6}
        do
            docker exec $POSTGRES_HOST psql -U postgres -c "CREATE DATABASE $POSTGRES_DB"
            if [ $? -eq 0 ]; then
                echo "SINGA-Auto's DB create database successful"
                break
            else
                echo "retry creating database $val"
                sleep 5
            fi
        done

        for val in {1..6}
        do
            docker exec $POSTGRES_HOST psql -U postgres -c "CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD'"
            if [ $? -eq 0 ]; then
                echo "SINGA-Auto's DB create user successful"
                break
            else
                echo "retry creating user $val"
                sleep 5
            fi
        done
}

start_kafka()
{
        title "Starting SINGA-Auto's Kafka..."
        LOG_FILE_PATH=$PWD/$LOGS_DIR_PATH/start_kafka.log
        (docker run --rm --name $KAFKA_HOST \
          --network $DOCKER_NETWORK \
          -e KAFKA_ZOOKEEPER_CONNECT=$ZOOKEEPER_HOST:$ZOOKEEPER_PORT \
          -e KAFKA_ADVERTISED_HOST_NAME=$KAFKA_HOST \
          -e KAFKA_ADVERTISED_PORT=$KAFKA_PORT \
          -e KAFKA_MESSAGE_MAX_BYTES=134217728\
          -e KAFKA_FETCH_MAX_BYTES=134217728\
          -p $KAFKA_EXT_PORT:$KAFKA_PORT \
          -d $IMAGE_KAFKA \
          &> $LOG_FILE_PATH) &
        ensure_stable "SINGA-Auto's Kafka" $LOG_FILE_PATH 2

}

start_zookeeper()
{
        title "Starting SINGA-Auto's Zookeeper..."
        LOG_FILE_PATH=$PWD/$LOGS_DIR_PATH/start_zookeeper.log
        (docker run --rm --name $ZOOKEEPER_HOST \
          --network $DOCKER_NETWORK \
          -p $ZOOKEEPER_EXT_PORT:$ZOOKEEPER_PORT \
          -d $IMAGE_ZOOKEEPER \
          &> $LOG_FILE_PATH) &
        ensure_stable "SINGA-Auto's Zookeeper" $LOG_FILE_PATH 5
}

start_redis()
{
      title "Starting SINGA-Auto's Redis..."
      LOG_FILE_PATH=$PWD/$LOGS_DIR_PATH/start_redis.log
      (docker run --rm --name $REDIS_HOST \
        --network $DOCKER_NETWORK \
        -p $REDIS_EXT_PORT:$REDIS_PORT \
        $IMAGE_REDIS redis-server --appendonly yes --requirepass $REDIS_PASSWORD \
        &> $LOG_FILE_PATH) &

      ensure_stable "SINGA-Auto's Redis" $LOG_FILE_PATH 2
}

start_web_admin()
{
      title "Starting SINGA-Auto's Web Admin..."
      LOG_FILE_PATH=$PWD/$LOGS_DIR_PATH/start_web_admin.log
      (docker run --rm --name $WEB_ADMIN_HOST \
        --network $DOCKER_NETWORK \
        -e SINGA_AUTO_ADDR=$SINGA_AUTO_ADDR \
        -e ADMIN_EXT_PORT=$ADMIN_EXT_PORT \
        -p $WEB_ADMIN_EXT_PORT:3001 \
        $SINGA_AUTO_IMAGE_WEB_ADMIN:$SINGA_AUTO_VERSION \
        &> $LOG_FILE_PATH) &
      ensure_stable "SINGA-Auto's Web Admin" $LOG_FILE_PATH 5
}

# Open Docker, only if is not running
if (! docker stats --no-stream ); then
  service docker start # Ubuntu/Debian
  # Wait until Docker daemon is running and has completed initialisation
while (! docker stats --no-stream ); do
  # Docker takes a few seconds to initialize
  echo "Waiting for Docker to launch..."
  sleep 1
done
fi

# set $HOST_WORKDIR_PATH
if [ $HOST_WORKDIR_PATH ];then
	echo "HOST_WORKDIR_PATH is exist, and echo to = $HOST_WORKDIR_PATH"
else
	export HOST_WORKDIR_PATH=$PWD
fi

# Read from shell configuration file
source $HOST_WORKDIR_PATH/scripts/docker_swarm/.env.sh
source $HOST_WORKDIR_PATH/scripts/base_utils.sh

title "Guidence"
help
create_folders

# Pull images from Docker Hub
bash $HOST_WORKDIR_PATH/scripts/pull_images.sh || exit 1

if [ ! -n "$1" ] ;then
     title "Launch Model: Init cluster and start all services"
     # Create Docker swarm for SINGA-Auto
     create_docker_swarm

     start_zookeeper || exit 1
     start_kafka || exit 1
     start_redis || exit 1

     # Skip starting & loading DB if DB is already running
     if is_docker_running $POSTGRES_HOST;then
        echo "Detected that SINGA-Auto's DB is already running!"
     else
          start_db
     fi

     start_admin || exit 1
     start_web_admin || exit 1

elif [[ $1 = "redis" ]];then
     title "Launch Model: Start redis"
     start_redis || exit 1

elif [[ $1 = "db" ]];then
     title "Launch Model: Start db"
     start_db || exit 1

elif [[ $1 = "kafka" ]];then
     title "Launch Model: Start kafka"
     start_zookeeper || exit 1
     start_kafka || exit 1

elif [[ $1 = "admin" ]];then
     title "Launch Model: Start admin"
     start_admin || exit 1

elif [[ $1 = "web" ]];then
     title "Launch Model: Start admin_web"
     start_web_admin || exit 1
else
    title "Unsupport arguments, please see the help doc"
fi

echo "To use SINGA-Auto, use SINGA-Auto Client in the Python CLI"
echo "A quickstart is available at https://nginyc.github.io/rafiki/docs/latest/src/user/quickstart.html"
echo "To configure SINGA-Auto, refer to SINGA-Auto's developer docs at https://nginyc.github.io/rafiki/docs/latest/src/dev/setup.html"
