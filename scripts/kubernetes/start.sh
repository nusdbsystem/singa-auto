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


help()
{
    cat <<- EOF
    Desc: used to launch the services
    Usage: start all services using: bash scripts/kubernetes/start.sh
           start redis using: bash scripts/kubernetes/start.sh redis
           start db using: bash scripts/kubernetes/start.sh db
           start kafka using: bash scripts/kubernetes/start.sh kafka
           start admin using: bash scripts/kubernetes/start.sh admin
           start web using: bash scripts/kubernetes/start.sh web
    Author: naili
EOF
}

start_zookeeper()
{
      title "Starting SINGA-Auto's Zookeeper..."

      LOG_FILE_PATH=$PWD/logs/start_zookeeper_service.log
      (kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_zookeeper_service.json \
      &> $LOG_FILE_PATH) &
      ensure_stable "SINGA-Auto's Zookeeper Service" $LOG_FILE_PATH 5

      LOG_FILE_PATH=$PWD/logs/start_zookeeper_deployment.log
      (kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_zookeeper_deployment.json \
      &> $LOG_FILE_PATH) &
      ensure_stable "SINGA-Auto's Zookeeper Deployment" $LOG_FILE_PATH 5
}

start_kafka()
{
      title "Starting SINGA-Auto's Kafka..."
      LOG_FILE_PATH=$PWD/logs/start_kafka_service.log
      (kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_kafka_service.json \
      &> $LOG_FILE_PATH) &
      ensure_stable "SINGA-Auto's Kafka Service" $LOG_FILE_PATH 2

      LOG_FILE_PATH=$PWD/logs/start_kafka_deployment.log
      (kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_kafka_deployment.json \
      &> $LOG_FILE_PATH) &
      ensure_stable "SINGA-Auto's Kafka Deployment" $LOG_FILE_PATH 2
}

start_redis()
{
      title "Starting SINGA-Auto's Redis..."
      LOG_FILE_PATH=$PWD/logs/start_redis_service.log
      (kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_redis_service.json \
      &> $LOG_FILE_PATH) &
      ensure_stable "SINGA-Auto's Redis Service" $LOG_FILE_PATH 2
      LOG_FILE_PATH=$PWD/logs/start_redis_deployment.log
      (kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_redis_deployment.json \
      &> $LOG_FILE_PATH) &
      ensure_stable "SINGA-Auto's Redis Deployment" $LOG_FILE_PATH 2
}

start_db()
{

      title "Starting SINGA-Auto's DB..."
      LOG_FILE_PATH=$PWD/logs/start_db_service.log
      (kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_db_service.json \
      &> $LOG_FILE_PATH) &
      ensure_stable "SINGA-Auto's DB Service" $LOG_FILE_PATH 10

      LOG_FILE_PATH=$PWD/logs/start_db_deployment.log
      (kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_db_deployment.json \
      &> $LOG_FILE_PATH) &
      ensure_stable "SINGA-Auto's DB Deployment" $LOG_FILE_PATH 10

      echo "Creating SINGA-Auto's PostgreSQL database & user..."
      ensure_stable "SINGA-Auto's DB" $LOG_FILE_PATH 20

      # retry 6 times
      for val in {1..6}
      do
          DB_PODNAME=$(kubectl get pod | grep $POSTGRES_HOST)
          DB_PODNAME=${DB_PODNAME:0:30}
          kubectl exec $DB_PODNAME -c $POSTGRES_HOST -- psql -U postgres -c "CREATE DATABASE $POSTGRES_DB"
          if [ $? -eq 0 ]; then
              echo "SINGA-Auto's DB create database successful"
              break
          else
              echo "retry creating database $val"
              sleep 5
          fi
      done

      for val in {1..6}
      do
          DB_PODNAME=$(kubectl get pod | grep $POSTGRES_HOST)
          DB_PODNAME=${DB_PODNAME:0:30}
          kubectl exec $DB_PODNAME -c $POSTGRES_HOST -- psql -U postgres -c "CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD'"
          if [ $? -eq 0 ]; then
              echo "SINGA-Auto's DB create user successful"
              break
          else
              echo "retry creating user $val"
              sleep 5
          fi
      done

}

start_admin()
{

      title "Starting SINGA-Auto's Admin..."

      LOG_FILE_PATH=$PWD/logs/start_admin_service.log
      (kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_admin_service.json \
      &> $LOG_FILE_PATH) &
      ensure_stable "SINGA-Auto's Admin Service" $LOG_FILE_PATH 5

      LOG_FILE_PATH=$PWD/logs/start_admin_deployment.log
      (kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_admin_deployment.json \
      &> $LOG_FILE_PATH) &
      ensure_stable "SINGA-Auto's Admin Deployment" $LOG_FILE_PATH 5

}

start_web_admin()
{
      title "Starting SINGA-Auto's Web Admin..."

      LOG_FILE_PATH=$PWD/logs/start_web_admin_service.log
      (kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_web_admin_service.json \
        &> $LOG_FILE_PATH) &
      ensure_stable "SINGA-Auto's Web Admin Service" $LOG_FILE_PATH 5

      LOG_FILE_PATH=$PWD/logs/start_web_admin_deployment.log
      (kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/start_web_admin_deployment.json \
      &> $LOG_FILE_PATH) &
      ensure_stable "SINGA-Auto's Web Admin Deployment" $LOG_FILE_PATH 5
}


if [ $HOST_WORKDIR_PATH ];then
	echo "HOST_WORKDIR_PATH is exist, and echo to = $HOST_WORKDIR_PATH"
else
	export HOST_WORKDIR_PATH=$PWD
fi

# Read from shell configuration file
source $HOST_WORKDIR_PATH/scripts/base_utils.sh
source $HOST_WORKDIR_PATH/scripts/kubernetes/.env.sh

title "Guidence"
help
create_folders


title "Creating the cluster role binding"
# ensure python api in pod has auth to control kubernetes
kubectl create clusterrolebinding add-on-cluster-admin \
    --clusterrole=cluster-admin --serviceaccount=default:default

# Pull images from Docker Hub
#bash $HOST_WORKDIR_PATH/scripts/pull_images.sh || exit 1

# Generate config files
echo "Generate config files"
bash $HOST_WORKDIR_PATH/scripts/kubernetes/generate_config.sh || exit 1

if [ ! -n "$1" ] ;then

     start_zookeeper || exit 1
     start_kafka || exit 1
     start_redis || exit 1

     # Skip starting & loading DB if DB is already running
     if [ "$CLUSTER_MODE" = "SINGLE" ]; then
        if is_k8s_running $POSTGRES_HOST
        then
          echo "Detected that Rafiki's DB is already running!"
        else
            start_db || exit 1
        fi
     else
        # Whether stolon has started inside the script
        bash $HOST_WORKDIR_PATH/scripts/kubernetes/start_stolon.sh || exit 1
     fi

     start_admin || exit 1
     start_web_admin || exit 1

     echo "Deploy monitor plugin"
     bash $HOST_WORKDIR_PATH/scripts/kubernetes/start_monitor.sh

     echo "Deploy ingress-nginx"
     if is_running_ingress
        then
          echo "Detected that Ingress-controller is already running!"
        else
          # this is the source yaml
          # kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-0.32.0/deploy/static/provider/baremetal/deploy.yaml
          # customer yaml: add replica to 3, fix the port to 3005
          kubectl apply -f $HOST_WORKDIR_PATH/scripts/kubernetes/yaml/ingress_controller_deploy.yaml || exit 1
        fi

     echo "Deploy GPU plugin"
     kubectl create -f $HOST_WORKDIR_PATH/scripts/kubernetes/yaml/nvidia-device-plugin.yml


elif [[ $1 = "redis" ]];then
     title "Launch Model: Start redis"
     start_redis || exit 1

elif [[ $1 = "db" ]];then
     title "Launch Model: Start db"
     start_db || exit 1

elif [[ $1 = "kafka" ]];then
     title "Launch Model: Start kafka"
     start_zookeeper || exit 1
     start_kafka || exit 1

elif [[ $1 = "admin" ]];then
     title "Launch Model: Start admin"
     start_admin || exit 1

elif [[ $1 = "web" ]];then
     title "Launch Model: Start admin_web"
     start_web_admin || exit 1
else
    title "Unsupport arguments, please see the help doc"
fi

bash $HOST_WORKDIR_PATH/scripts/kubernetes/remove_config.sh || exit 1

echo "To use SINGA-Auto, use SINGA-Auto Client in the Python CLI"
echo "A quickstart is available at https://nginyc.github.io/rafiki/docs/latest/src/user/quickstart.html"
echo "To configure SINGA-Auto, refer to SINGA-Auto's developer docs at https://nginyc.github.io/rafiki/docs/latest/src/dev/setup.html"
