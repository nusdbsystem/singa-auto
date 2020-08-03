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

# those need to be changed when do the deployments
IP_ADRESS=127.0.0.1
SINGA_AUTO_VERSION=dev

if [ $HOST_WORKDIR_PATH ];then
	echo "HOST_WORKDIR_PATH is exist, and echo to = $HOST_WORKDIR_PATH"
else
	export HOST_WORKDIR_PATH=$PWD
fi

# Core external configuration for SINGA-auto
export DOCKER_NETWORK=singa_auto
export DOCKER_SWARM_ADVERTISE_ADDR=$IP_ADRESS
export CONTAINER_MODE=SWARM

source $HOST_WORKDIR_PATH/scripts/.base_env.sh $IP_ADRESS $SINGA_AUTO_VERSION
