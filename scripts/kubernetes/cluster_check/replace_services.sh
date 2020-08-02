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

title "replce SINGA-Auto's services, this scipt is used to update the service with only code change, while yaml file stay same"

bash $HOST_WORKDIR_PATH/scripts/kubernetes/generate_config.sh

echo "repalcing $1 "

if [[ $1 = "admin" ]]
then

    kubectl replace --force -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_admin_deployment.json
    kubectl replace --force -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_admin_service.json
fi

if [[ $1 = "db" ]]
then
    kubectl replace --force -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_db_deployment.json
    kubectl replace --force -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_db_service.json
fi

if [[ $1 = "web" ]]
then
    kubectl replace --force -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_web_admin_deployment.json
    kubectl replace --force -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_web_admin_service.json
fi


if [[ $1 = "monitor" ]]
then
    kubectl replace --force -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_logstash_deployment.json
    kubectl replace --force -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_logstash_service.json

    kubectl replace --force -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_es_deployment.json
    kubectl replace --force -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_es_service.json

    kubectl replace --force -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_kibana_deployment.json
    kubectl replace --force -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_kibana_service.json

    kubectl replace --force -f $HOST_WORKDIR_PATH/scripts/kubernetes/spark-app.json
fi
bash $HOST_WORKDIR_PATH/scripts/kubernetes/remove_config.sh
