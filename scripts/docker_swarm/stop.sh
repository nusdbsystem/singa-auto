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

LOG_FILEPATH=$PWD/$LOGS_DIR_PATH/stop.log

source ./scripts/base_utils.sh

# Read from shell configuration file
source ./scripts/docker_swarm/.env.sh

title "Stopping any existing jobs..."
echo $(which python3)
pyv="$(python3 -V 2>&1)"
echo $pyv
python3 ./scripts/stop_all_jobs.py

bash scripts/docker_swarm/stop_db.sh || exit 1

# Prompt if should stop DB
#if prompt "Should stop SINGA-Auto's DB?"
#then
#    bash scripts/docker_swarm/stop_db.sh || exit 1
#else
#    echo "Not stopping SINGA-Auto's DB!"
#fi

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
