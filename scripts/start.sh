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

source ./scripts/utils.sh

# Read from shell configuration file
source ./.env.sh

# Create Docker swarm for Singa-Auto
bash ./scripts/create_docker_swarm.sh

# Pull images from Docker Hub
bash ./scripts/pull_images.sh || exit 1

# Start whole Singa-Auto stack
# Start Zookeeper, Kafka & Redis
bash ./scripts/start_zookeeper.sh || exit 1
bash ./scripts/start_kafka.sh || exit 1
bash ./scripts/start_redis.sh || exit 1
# Skip starting & loading DB if DB is already running
if is_running $POSTGRES_HOST
then
  echo "Detected that Singa-Auto's DB is already running!"
else
    bash ./scripts/start_db.sh || exit 1
    bash ./scripts/load_db.sh || exit 1
fi
bash ./scripts/start_admin.sh || exit 1
bash ./scripts/start_web_admin.sh || exit 1

echo "To use Singa-Auto, use Singa-Auto Client in the Python CLI"
echo "A quickstart is available at https://nginyc.github.io/rafiki/docs/latest/src/user/quickstart.html"
echo "To configure Singa-Auto, refer to Singa-Auto's developer docs at https://nginyc.github.io/rafiki/docs/latest/src/dev/setup.html"
