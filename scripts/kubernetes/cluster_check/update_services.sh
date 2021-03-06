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
source $HOST_WORKDIR_PATH/scripts/kubernetes/.env.sh
source $HOST_WORKDIR_PATH/scripts/base_utils.sh

title "updating SINGA-Auto's services"

bash $HOST_WORKDIR_PATH/scripts/kubernetes/generate_config.sh

kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_admin_deployment.json --record
kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_admin_service.json --record

kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_db_deployment.json --record
kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_db_service.json --record

kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_kafka_deployment.json --record
kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_kafka_service.json --record

kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_redis_deployment.json --record
kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_redis_service.json --record

kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_web_admin_deployment.json --record
kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_web_admin_service.json --record

kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_zookeeper_deployment.json --record
kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_zookeeper_service.json --record

bash $HOST_WORKDIR_PATH/scripts/kubernetes/remove_config.sh
