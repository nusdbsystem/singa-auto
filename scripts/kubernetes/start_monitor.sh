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

if [ $HOST_WORKDIR_PATH ];then
	echo "HOST_WORKDIR_PATH is exist, and echo to = $HOST_WORKDIR_PATH"
else
	export HOST_WORKDIR_PATH=$PWD
fi

source $HOST_WORKDIR_PATH/scripts/kubernetes/.env.sh
source $HOST_WORKDIR_PATH/scripts/base_utils.sh

title "Starting SINGA-Auto's Monitor..."

# start logstash
LOGSTADH_LOG_FILE_PATH=$PWD/logs/start_logstash_service.log
(kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_logstash_service.json \
&> $LOGSTADH_LOG_FILE_PATH) &
ensure_stable "SINGA-Auto's logstash Service" $LOGSTADH_LOG_FILE_PATH 1

LOGSTADH_LOG_FILE_PATH=$PWD/logs/start_logstash_deployment.log
(kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_logstash_deployment.json \
&> $LOGSTADH_LOG_FILE_PATH) &
ensure_stable "SINGA-Auto's logstash Service" $LOGSTADH_LOG_FILE_PATH 1

# start es

ES_LOG_FILE_PATH=$PWD/logs/start_es_service.log
(kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_es_service.json \
&> $ES_LOG_FILE_PATH) &
ensure_stable "SINGA-Auto's es Service" $ES_LOG_FILE_PATH 1

ES_LOG_FILE_PATH=$PWD/logs/start_es_deployment.log
(kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_es_deployment.json \
&> $ES_LOG_FILE_PATH) &
ensure_stable "SINGA-Auto's es Service" $ES_LOG_FILE_PATH 1

# kibana start

KIBANA_LOG_FILE_PATH=$PWD/logs/start_kibana_service.log
(kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_kibana_service.json \
&> $KIBANA_LOG_FILE_PATH) &
ensure_stable "SINGA-Auto's kibana Service" KIBANA_LOG_FILE_PATH 1

KIBANA_LOG_FILE_PATH=$PWD/logs/start_kibana_deployment.log
(kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_kibana_deployment.json \
&> $KIBANA_LOG_FILE_PATH) &
ensure_stable "SINGA-Auto's kibana Service" KIBANA_LOG_FILE_PATH 1


# start sparkapp

helm repo add incubator http://storage.googleapis.com/kubernetes-charts-incubator
helm install incubator/sparkoperator --namespace default --set operatorVersion=v1beta2-1.1.2-2.4.5 --generate-name

title "Starting SINGA-Auto's spark streamming..."
SPARK_LOG_FILE_PATH=$PWD/logs/start_spark.log
(kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/yaml/spark-rbac.yaml \
&> $SPARK_LOG_FILE_PATH) &
(kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/spark-app.json \
&> $SPARK_LOG_FILE_PATH) &
ensure_stable "SINGA-Auto's sparkoperator" KIBANA_LOG_FILE_PATH 20
