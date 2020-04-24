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

#Open Docker, only if is not running
if (! docker stats --no-stream ); then
  service docker start # Ubuntu/Debian
  #Wait until Docker daemon is running and has completed initialisation
while (! docker stats --no-stream ); do
  # Docker takes a few seconds to initialize
  echo "Waiting for Docker to launch..."
  sleep 1
done
fi

source ./scripts/docker_swarm/utils.sh

# Read from shell configuration file
source ./scripts/docker_swarm/.env.sh

# Create Docker swarm for SINGA-Auto
bash ./scripts/docker_swarm/create_docker_swarm.sh

# Pull images from Docker Hub
#bash ./scripts/docker_swarm/pull_images.sh || exit 1

# Start whole SINGA-Auto stack
# Start Zookeeper, Kafka & Redis
bash ./scripts/docker_swarm/start_zookeeper.sh || exit 1
bash ./scripts/docker_swarm/start_kafka.sh || exit 1
bash ./scripts/docker_swarm/start_redis.sh || exit 1
# Skip starting & loading DB if DB is already running
if is_running $POSTGRES_HOST
then
  echo "Detected that SINGA-Auto's DB is already running!"
else
    bash ./scripts/docker_swarm/start_db.sh

fi
bash ./scripts/docker_swarm/start_admin.sh || exit 1
bash ./scripts/docker_swarm/start_web_admin.sh || exit 1

echo "To use SINGA-Auto, use SINGA-Auto Client in the Python CLI"
echo "A quickstart is available at https://nginyc.github.io/rafiki/docs/latest/src/user/quickstart.html"
echo "To configure SINGA-Auto, refer to SINGA-Auto's developer docs at https://nginyc.github.io/rafiki/docs/latest/src/dev/setup.html"
