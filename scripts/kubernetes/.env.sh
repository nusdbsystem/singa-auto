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

#ingress default configurations
export INGRESS_NAME=ingress-predictor
export INGRESS_EXT_PORT=3005

# Core external configuration for SINGA-auto
export KUBERNETES_NETWORK=singa_auto
export KUBERNETES_ADVERTISE_ADDR=$IP_ADRESS

export POSTGRES_STOLON_PASSWD=cmFmaWtpCg==  # The Passwd for stolon, base64 encode

export SPAEK_DOCKER_JARS_PATH=/opt/spark/examples
export SINGA_AUTO_IMAGE_SPARKAPP=singaauto/singa_auto_sparkapp

export SINGA_AUTO_IMAGE_STOLON=sorintlab/stolon:master-pg10

export CONTAINER_MODE=K8S

# Cluster Mode for SINGA-auto
export CLUSTER_MODE=SINGLE # CLUSTER or SINGLE

if [ "$CLUSTER_MODE" = "CLUSTER" ]; then
    export POSTGRES_HOST=stolon-proxy-service
    export NFS_HOST_IP=$IP_ADRESS      # NFS Host IP - if used nfs as pv for database storage
    export RUN_DIR_PATH=run            # Shares a folder with containers that stores components' running info, relative to workdir
fi

source scripts/.base_env.sh $IP_ADRESS $SINGA_AUTO_VERSION || exit 1

