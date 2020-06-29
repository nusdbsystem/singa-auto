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

source ./scripts/kubernetes/.env.sh

source ./scripts/kubernetes/utils.sh

echo "Stop option: $1"

if [[ $1 = "allsvc" ]]
then
      title "Only stop predictor and training services"
      kubectl get deployments | grep singa-auto-predict | awk '{print $1}' | sudo xargs -I {} kubectl delete deployment {}
      kubectl get deployments | grep singa-auto-inference | awk '{print $1}' | sudo xargs -I {} kubectl delete deployment {}
      kubectl get deployments | grep singa-auto-advisor | awk '{print $1}' | sudo xargs -I {} kubectl delete deployment {}
      kubectl get deployments | grep singa-auto-train | awk '{print $1}' | sudo xargs -I {} kubectl delete deployment {}

      kubectl get services | grep singa-auto-predict | awk '{print $1}' | sudo xargs -I {} kubectl delete services {}
elif [[ $1 = "failsvc" ]]
then
      title "Only stop failed predictor and training services"

      kubectl get deployments | grep singa-auto-predict | grep 0/ | awk '{print $1}' | sudo xargs -I {} kubectl delete deployment {} |  awk '{print $2}'| sudo xargs -I {} kubectl delete svc {}
      kubectl get deployments | grep singa-auto-inference | grep 0/ | awk '{print $1}' | sudo xargs -I {} kubectl delete deployment {}
      kubectl get deployments | grep singa-auto-advisor | grep 0/ | awk '{print $1}' | sudo xargs -I {} kubectl delete deployment {}
      kubectl get deployments | grep singa-auto-train | grep 0/ | awk '{print $1}' | sudo xargs -I {} kubectl delete deployment {}

else

#      kubectl delete -f ./scripts/kubernetes/nvidia-device-plugin.yml

      title "Stopping any existing jobs..."
      python ./scripts/stop_all_jobs.py

      title "Stopping SINGA-Auto's Web Admin Deployment..."
      kubectl delete deployment $WEB_ADMIN_HOST || echo "Failed to stop SINGA-Auto's Web Admin Deployment"

      title "Stopping SINGA-Auto's Admin Deployment..."
      kubectl delete deployment $ADMIN_HOST || echo "Failed to stop SINGA-Auto's Admin Deployment"

      title "Stopping SINGA-Auto's Redis Deployment..."
      kubectl delete deployment $REDIS_HOST || echo "Failed to stop SINGA-Auto's Redis Deployment"

      title "Stopping SINGA-Auto's Kafka Deployment..."
      kubectl delete deployment $KAFKA_HOST || echo "Failed to stop SINGA-Auto's Kafka Deployment"

      title "Stopping SINGA-Auto's Zookeeper Deployment..."
      kubectl delete deployment $ZOOKEEPER_HOST || echo "Failed to stop SINGA-Auto's Zookeeper Deployment"

      title "Stopping SINGA-Auto's LogStash Deployment..."
      kubectl delete deployment $LOGSTASH_HOST || echo "Failed to stop SINGA-Auto's LogStash deployment"

      title "Stopping SINGA-Auto's Kibana Deployment..."
      kubectl delete deployment $KIBANA_HOST || echo "Failed to stop SINGA-Auto's Kibana deployment"

      title "Stopping SINGA-Auto's ES Deployment..."
      kubectl delete deployment $ES_HOST || echo "Failed to stop SINGA-Auto's ES deployment"


      title "Stopping SINGA-Auto's Web Admin Service..."
      kubectl delete service $WEB_ADMIN_HOST || echo "Failed to stop SINGA-Auto's Web Admin Service"

      title "Stopping SINGA-Auto's Admin Service..."
      kubectl delete service $ADMIN_HOST || echo "Failed to stop SINGA-Auto's Admin Service"

      title "Stopping SINGA-Auto's Redis Service..."
      kubectl delete service $REDIS_HOST || echo "Failed to stop SINGA-Auto's Redis Service"

      title "Stopping SINGA-Auto's Kafka Service..."
      kubectl delete service $KAFKA_HOST || echo "Failed to stop SINGA-Auto's Kafka Service"

      title "Stopping SINGA-Auto's Zookeeper Service..."
      kubectl delete service $ZOOKEEPER_HOST || echo "Failed to stop SINGA-Auto's Zookeeper Service"

      title "Stopping SINGA-Auto's LogStash Service..."
      kubectl delete service $LOGSTASH_HOST || echo "Failed to stop SINGA-Auto's LogStash Service"

      title "Stopping SINGA-Auto's Kibana Service..."
      kubectl delete service $KIBANA_HOST || echo "Failed to stop SINGA-Auto's Kibana Service"

      title "Stopping SINGA-Auto's ES Service..."
      kubectl delete service $ES_HOST || echo "Failed to stop SINGA-Auto's ES Service"

      bash ./scripts/kubernetes/generate_config.sh || exit 1
      title "Stopping SINGA-Auto's ES Service..."
      kubectl delete -f scripts/kubernetes/spark-app.json
      bash ./scripts/kubernetes/remove_config.sh

      if [ "$CLUSTER_MODE" = "SINGLE" ]; then
          bash scripts/kubernetes/stop_db.sh || exit 1
      else
          bash scripts/kubernetes/stop_stolon.sh || exit 1
      fi

fi

# Prompt if should stop DB
#if prompt "Should stop SINGA-Auto's DB?"
#then
#    if [ "$CLUSTER_MODE" = "SINGLE" ]; then
#        bash scripts/kubernetes/stop_db.sh || exit 1
#    else
#        bash scripts/kubernetes/stop_stolon.sh || exit 1
#    fi
#else
#    echo "Not stopping SINGA-Auto's DB!"
#fi
