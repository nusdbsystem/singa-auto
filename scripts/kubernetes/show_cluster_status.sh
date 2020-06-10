
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

title()
{
    title="| $1 |"
    edge=$(echo "$title" | sed 's/./-/g')
    echo "$edge"
    echo "$title"
    echo "$edge"
}

title "pods info"
sudo kubectl get pods -o wide
title "svc info"
sudo kubectl get svc -o wide
title "deployments info"
sudo kubectl get deployments -o wide
title "nodes info"
sudo kubectl get nodes -o wide

title "systemm pods info"
sudo kubectl get pods -o wide --namespace=kube-system
title "system svc info"
sudo kubectl get svc -o wide --namespace=kube-system
title "system deployments info"
sudo kubectl get deployments -o wide --namespace=kube-system

title "ingress deployments info"
sudo kubectl get deployments -o wide --namespace=ingress-nginx
title "ingress svc  info"
sudo kubectl get svc -o wide --namespace=ingress-nginx
title "ingress pods info"
sudo kubectl get pods -o wide --namespace=ingress-nginx

title "spark operator"
helm list
