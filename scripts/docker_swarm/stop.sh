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

stop_db()
{
      LOG_FILEPATH=$PWD/$LOGS_DIR_PATH/stop.log

      title "Dumping database..."

      #-------------------------------------------
      #| saving db |
      #-------------------------------------------

      DUMP_FILE=$POSTGRES_DUMP_FILE_PATH

      # Check if dump file exists
      if [ -f $DUMP_FILE ]
      then
          if ! prompt "Database dump file exists at $DUMP_FILE. Override it?"
          then
              echo "Not dumping database!"
              exit 0
          fi
      fi

      echo "Dumping database to $DUMP_FILE..."
      docker exec $POSTGRES_HOST pg_dump -U postgres --if-exists --clean $POSTGRES_DB > $DUMP_FILE

      # If database dump previously failed, prompt whether to continue script
      #if [ $? -ne 0 ]
      #then
      #    if ! prompt "Failed to dump database. Continue?"
      #    then
      #        exit 1
      #    fi
      #fi

      title "Stopping SINGA-Auto's DB..."
      docker rm -f $POSTGRES_HOST || echo "Failed to stop SINGA-Auto's DB"
}

LOG_FILEPATH=$PWD/$LOGS_DIR_PATH/stop.log

if [ $HOST_WORKDIR_PATH ];then
	echo "HOST_WORKDIR_PATH is exist, and echo to = $HOST_WORKDIR_PATH"
else
	export HOST_WORKDIR_PATH=$PWD
fi

source $HOST_WORKDIR_PATH/scripts/docker_swarm/.env.sh

source $HOST_WORKDIR_PATH/scripts/base_utils.sh

# Read from shell configuration file

title "Stopping any existing jobs..."
echo $(which python3)
pyv="$(python3 -V 2>&1)"
echo $pyv
python3 $HOST_WORKDIR_PATH/scripts/stop_all_jobs.py

stop_db || exit 1

title "Stopping SINGA-Auto's Zookeeper..."
docker rm -f $ZOOKEEPER_HOST || echo "Failed to stop SINGA-Auto's Zookeeper"

title "Stopping SINGA-Auto's Kafka..."
docker rm -f $KAFKA_HOST || echo "Failed to stop SINGA-Auto's Kafka"

title "Stopping SINGA-Auto's Redis..."
docker rm -f $REDIS_HOST || echo "Failed to stop SINGA-Auto's Redis"

title "Stopping SINGA-Auto's Admin..."
docker rm -f $ADMIN_HOST || echo "Failed to stop SINGA-Auto's Admin"

title "Stopping SINGA-Auto's Web Admin..."
docker rm -f $WEB_ADMIN_HOST || echo "Failed to stop SINGA-Auto's Web Admin"

title "Stopping SINGA-Auto's Kibana..."
docker rm -f $KIBANA_HOST

title "Stopping SINGA-Auto's logstash..."
docker rm -f $LOGSTASH_HOST

title "Stopping SINGA-Auto's es..."
docker rm -f $ES_HOST

echo "You'll need to destroy your machine's Docker swarm manually"
