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

# Push Singa-Auto's images to Docker Hub

docker login

echo "Pushing all Singa-Auto's images to Docker Hub..."
docker push $SINGA_AUTO_IMAGE_ADMIN:$SINGA_AUTO_VERSION || exit 1
docker push $SINGA_AUTO_IMAGE_WORKER:$SINGA_AUTO_VERSION || exit 1
docker push $SINGA_AUTO_IMAGE_PREDICTOR:$SINGA_AUTO_VERSION || exit 1
docker push $SINGA_AUTO_IMAGE_WEB_ADMIN:$SINGA_AUTO_VERSION || exit 1
echo "Pushed all images!"
