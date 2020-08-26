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

import os
import time
from typing import Tuple, List
from kubernetes import client
import logging
import traceback
from functools import wraps

from .container_manager import ContainerManager, ContainerService
from singa_auto.error_code import ServiceRequestError
from singa_auto.constants import NodeLabes

RETRY_WAIT_SECS = 1
RETRY_TIMES = 5

logger = logging.getLogger(__name__)


class KubernetesContainerManager(ContainerManager):

    def __init__(self, **kwargs):
        aToken = None
        with open('/var/run/secrets/kubernetes.io/serviceaccount/token', 'r') as fToken:
            aToken = fToken.read()

        # Create a configuration object
        aConfiguration = client.Configuration()

        # Specify the endpoint of your Kube cluster
        aConfiguration.host = "https://{}:{}".format(
            os.getenv('KUBERNETES_SERVICE_HOST'),
            os.getenv('KUBERNETES_SERVICE_PORT'))

        # Security part.
        # In this simple example we are not going to verify the SSL certificate of
        # the remote cluster (for simplicity reason)
        aConfiguration.verify_ssl = False
        # Nevertheless if you want to do it you can with these 2 parameters
        # configuration.verify_ssl=True
        # ssl_ca_cert is the filepath to the file that contains the certificate.
        # configuration.ssl_ca_cert="certificate"

        aConfiguration.api_key = {"authorization": "Bearer " + aToken}

        # Create a ApiClient with our config
        aApiClient = client.ApiClient(aConfiguration)

        self._client_deployment = client.AppsV1Api(aApiClient)
        self._client_service = client.CoreV1Api(aApiClient)
        self.api_instance = client.NetworkingV1beta1Api(aApiClient)

    def update_ingress(self, ingress_name: str, ingress_body: dict):
        paths = self._update_ingress_paths(ingress_body)
        body = client.NetworkingV1beta1Ingress(
            api_version="networking.k8s.io/v1beta1",
            kind="Ingress",
            metadata=client.V1ObjectMeta(
                                        name=ingress_name,
                                        annotations={
                                            "nginx.ingress.kubernetes.io/rewrite-target": "/"
                                            }
                                        ),
            spec=client.NetworkingV1beta1IngressSpec(
                rules=[client.NetworkingV1beta1IngressRule(
                    http=client.NetworkingV1beta1HTTPIngressRuleValue(
                        paths=paths
                    )
                )
                ]
            )
        )

        # check the current ingress list
        ingress_list = self.api_instance.list_namespaced_ingress_with_http_info(namespace='default')

        if ingress_list[1] != 200:
            raise ServiceRequestError("ingress response code is not 200")

        # get the ingress name of each ingress
        ingress_names = [ele.metadata.name for ele in ingress_list[0].items]

        # check if the ingress_name in the list
        if ingress_name in ingress_names:

            # if the ingress is exist, update it
            self.api_instance.replace_namespaced_ingress_with_http_info(name=ingress_name,
                                                                        namespace='default',
                                                                        body=body)
        else:
            # otherwise, create new one
            self.api_instance.create_namespaced_ingress_with_http_info(namespace='default',
                                                                       body=body)

    def _update_ingress_paths(self, ingress_body: dict) -> list:
        paths = list()
        for path_info in ingress_body["spec"]["rules"][0]["http"]["paths"]:
            path_obj = client.NetworkingV1beta1HTTPIngressPath(
                            path=path_info["path"],
                            backend=client.NetworkingV1beta1IngressBackend(
                                service_port=path_info["backend"]["servicePort"],
                                service_name=path_info["backend"]["serviceName"])

                        )
            paths.append(path_obj)

        return paths

    def destroy_service(self, service: ContainerService):
        self._client_deployment.delete_namespaced_deployment(service.id, namespace='default')
        self._client_service.delete_namespaced_service(service.id, namespace='default')

    def create_service(self,
                       service_name,
                       docker_image,
                       replicas,
                       args,
                       environment_vars,
                       mounts=None,
                       publish_port=None,
                       gpus=0,
                       dist_workers=0) -> ContainerService:
        if mounts is None:
            mounts = {}
        hostname = service_name
        if publish_port is not None:
            service_config = self._create_service_config(service_name, docker_image, replicas,
                            args, environment_vars, mounts, publish_port,
                            gpus)
            _retry(self._client_service.create_namespaced_service)(namespace='default', body=service_config)

        # if use disributed training, create muti-pod without using deployments
        if dist_workers > 0:
            print('Using distributed training (data parallelism)')

            # those variables is the same for all processes
            # master process's port
            environment_vars["DIST_TRAIN_MODEL"] = "DIST"
            environment_vars["MASTER_PORT"] = "23456"
            # num of processes
            environment_vars["WORLD_SIZE"] = str(dist_workers)
            environment_vars["PYTHONUNBUFFERED"] = "0"

            master_name = service_name + "-master-0"

            # get the scheduler result
            node_gpuid = []
            select_gpu, select_node_name = "", ""

            if gpus > 0:
                # run the scheduler algorithm, choose the gpu and node for few pods.
                node_gpuid = self._get_top_gpus(dist_workers)

            for index in range(dist_workers):
                environment_vars["RANK"] = str(index)

                # if scheduler is successful, get node and gpu info
                if node_gpuid:
                    select_node_name, select_gpu = node_gpuid[index]["nodeName"], node_gpuid[index]["GPUID"]

                if index == 0:
                    # create master, by default, first process is the master process
                    # for master process, master name should be localhost
                    environment_vars["MASTER_ADDR"] = "localhost"

                    pod_config = self._create_pod_config(master_name, docker_image,
                                                         environment_vars, mounts, select_gpu, select_node_name)
                    print("pod_config", pod_config)
                    _retry(self._client_service.create_namespaced_pod)(namespace='default', body=pod_config)

                    # create a service for the master pod, which is used to provide communication between worker's pod
                    # and master's pod, (assign master's service name and port to worker's pod envs)
                    # above is only one option
                    # another option is to assign the master pod's ip address to worker pods' env.
                    # will do it after pod is built.
                    # currently we choose the first option. create a service for master's pod

                    service_config = self._create_clusterip_service_config(service_name=master_name,
                                                                           publish_port=environment_vars["MASTER_PORT"]
                                                                           )
                    print("SVC config", service_config)
                    _retry(self._client_service.create_namespaced_service)(namespace='default', body=service_config)

                else:
                    # create worker, by default
                    # for worker process, worker name should be master's pod name
                    environment_vars["MASTER_ADDR"] = master_name

                    worker_name = service_name + "-worker-{}".format(index-1)
                    pod_config = self._create_pod_config(worker_name, docker_image,
                                                         environment_vars, mounts, select_gpu, select_node_name)
                    print("pod_config", pod_config)
                    _retry(self._client_service.create_namespaced_pod)(namespace='default', body=pod_config)

        else:
            deployment_config = self._create_deployment_config(service_name, docker_image, replicas,
                                                               environment_vars, mounts, gpus
                                                               )

            _retry(self._client_deployment.create_namespaced_deployment)(namespace='default', body=deployment_config)

        info = {
            'node_id': 'default',
            'gpu_nos': gpus,
            'service_name': service_name,
            'replicas': replicas
        }

        service = ContainerService(
            service_name, hostname,
            publish_port[0] if publish_port is not None else None, info)
        return service

    def _create_pod_config(self,
                           service_name: str,
                           docker_image: str,
                           environment_vars: dict,
                           mounts: dict,
                           gpu_id: str,
                           select_node_name: str):

        volumeMounts = list()
        volumes = list()
        mounts_count = 0
        for (k, v) in mounts.items():
            volumeMounts.append({
                'name': 'v' + str(mounts_count),
                'mountPath': v
            })
            volumes.append({
                'name': 'v' + str(mounts_count),
                'hostPath': {
                    'path': k
                }
            })
            mounts_count += 1

        env = [{'name': k, 'value': v} for (k, v) in environment_vars.items()]

        if gpu_id and select_node_name:
            nodeSelector = {NodeLabes.NodeName: select_node_name}

            # NVIDIA_VISIBLE_DEVICES is used to expose a specific gpu to this pod
            env.append({"name": "NVIDIA_VISIBLE_DEVICES", "value": gpu_id})

            content = \
                {'apiVersion': 'v1',
                 'kind': 'Pod',
                 'metadata': {'labels': {'name': service_name},
                              'name': service_name},
                 'spec': {'containers': [{'env': env,
                                          'image': docker_image,
                                          'imagePullPolicy': 'Always',
                                          'name': service_name,
                                          'resources': {'limits': {'nvidia.com/gpu': "1"}},
                                          'volumeMounts': volumeMounts}],
                          'nodeSelector': nodeSelector,
                          'volumes': volumes,
                          "restartPolicy": "Never"
                          }
                 }
        else:
            # this configuration is for cpu
            content = \
                {'apiVersion': 'v1',
                 'kind': 'Pod',
                 'metadata': {'labels': {'name': service_name},
                              'name': service_name},
                 'spec': {'containers': [{'env': env,
                                          'image': docker_image,
                                          'imagePullPolicy': 'Always',
                                          'name': service_name,
                                          'volumeMounts': volumeMounts}],
                          'volumes': volumes,
                          "restartPolicy": "Never"

                          }
                 }
        return content

    def _create_deployment_config(self,
                                  service_name,
                                  docker_image,
                                  replicas,
                                  environment_vars,
                                  mounts,
                                  gpus=0
                                  ):
        content = {}
        content.setdefault('apiVersion', 'apps/v1')
        content.setdefault('kind', 'Deployment')
        metadata = content.setdefault('metadata', {})
        metadata.setdefault('name', service_name)
        labels = metadata.setdefault('labels', {})
        labels.setdefault('name', service_name)
        spec = content.setdefault('spec', {})
        spec.setdefault('replicas', replicas)
        spec.setdefault('selector', {'matchLabels': {'name': service_name}})
        template = spec.setdefault('template', {})
        template.setdefault('metadata', {'labels': {'name': service_name}})
        container = {}
        container.setdefault('name', service_name)
        container.setdefault('image', docker_image)
        spec.setdefault('imagePullPolicy', 'Always')
        volumeMounts = container.setdefault('volumeMounts', [])
        volumes = []
        mounts_count = 0
        for (k, v) in mounts.items():
            volumeMounts.append({
                'name': 'v' + str(mounts_count),
                'mountPath': v
            })
            volumes.append({
                'name': 'v' + str(mounts_count),
                'hostPath': {
                    'path': k
                }
            })
            mounts_count += 1
        template.setdefault('spec', {
            'containers': [container],
            'volumes': volumes
        })
        env = [{'name': k, 'value': v} for (k, v) in environment_vars.items()]

        if gpus > 0:
            node_gpuid = self._get_top_gpus(1)
            if node_gpuid and node_gpuid[0]:
                select_node_name, select_gpu = node_gpuid[0]["nodeName"], node_gpuid[0]["GPUID"]
                # nodeSelector can be used to bind a pod to a node
                nodeSelector = {NodeLabes.NodeName: select_node_name}
                template["spec"]["nodeSelector"] = nodeSelector

                # NVIDIA_VISIBLE_DEVICES is used to expose a specific gpu to this pod
                env.append({"name": "NVIDIA_VISIBLE_DEVICES", "value": select_gpu})

            container.setdefault('resources',
                                 {'limits': {
                                     'nvidia.com/gpu': gpus
                                 }})
        container.setdefault('env', env)
        return content

    def _get_top_gpus(self, n) -> List[dict]:


        """
        This method is used to find the top n gpus, the one with most free memory
        nodeInfo is format of following:

        {'api_version': 'v1',
         'items': [
           {'api_version': None,
            'kind': None,
            'metadata': {'annotations': {'flannel.alpha.coreos.com/backend-data': '{"VtepMAC":"52:6c:01:9c:4b:27"}',
                                         'flannel.alpha.coreos.com/backend-type': 'vxlan',
                                         'flannel.alpha.coreos.com/kube-subnet-manager': 'true',
                                         'flannel.alpha.coreos.com/public-ip': '10.0.0.121',
                                         'kubeadm.alpha.kubernetes.io/cri-socket': '/var/run/dockershim.sock',
                                         'node.alpha.kubernetes.io/ttl': '0',
                                         'volumes.kubernetes.io/controller-managed-attach-detach': 'true'},
                         'cluster_name': None,
                         'creation_timestamp': datetime.datetime(2020, 6, 29, 10, 41, 21, tzinfo=tzutc()),
                         'deletion_grace_period_seconds': None,
                         'deletion_timestamp': None,
                         'finalizers': None,
                         'generate_name': None,
                         'generation': None,
                         'initializers': None,
                         'labels': {},
                         'managed_fields': None,
                         'name': 'panda1',
                         'namespace': None,
                         'owner_references': None,
                         'resource_version': '16021798',
                         'self_link': '/api/v1/nodes/panda1',
                         'uid': '97ec8d34-afd2-412c-8cf6-24f7a6f6bec0'},
            'spec': {'config_source': None,
                     'external_id': None,
                     'pod_cidr': '10.244.2.0/24',
                     'provider_id': None,
                     'taints': None,
                     'unschedulable': None},
            'status': {'addresses': [{'address': '10.0.0.121',
                                      'type': 'InternalIP'},
                                     {'address': 'panda1', 'type': 'Hostname'}],
                       'allocatable': {'cpu': '12',
                                       'ephemeral-storage': '225807475134',
                                       'hugepages-1Gi': '0',
                                       'hugepages-2Mi': '0',
                                       'memory': '65784940Ki',
                                       'nvidia.com/gpu': '3',
                                       'pods': '110'},
                       'capacity': {'cpu': '12',
                                    'ephemeral-storage': '245016792Ki',
                                    'hugepages-1Gi': '0',
                                    'hugepages-2Mi': '0',
                                    'memory': '65887340Ki',
                                    'nvidia.com/gpu': '3',
                                    'pods': '110'},
                       'conditions': [{'last_heartbeat_time': datetime.datetime(2020, 7, 1, 9, 47, 52, tzinfo=tzutc()),
                                       'last_transition_time': datetime.datetime(2020, 7, 1, 9, 47, 52, tzinfo=tzutc()),
                                       'message': 'Flannel is running on this '
                                                  'node',
                                       'reason': 'FlannelIsUp',
                                       'status': 'False',
                                       'type': 'NetworkUnavailable'}
                                       ],
                       'config': None,
                       'daemon_endpoints': {'kubelet_endpoint': {'port': 10250}},
                       'images': [{'names': ['singa_auto/singa_auto_worker:dev_1.1'],
                                   'size_bytes': 3239415641}
                                   ],
                       'node_info': {'architecture': 'amd64',
                                     'boot_id': '5ebe5f49-35fd-4306-a0c0-7b41df95f1ff',
                                     'container_runtime_version': 'docker://18.9.0',
                                     'kernel_version': '4.4.0-131-generic',
                                     'kube_proxy_version': 'v1.15.12',
                                     'kubelet_version': 'v1.15.12',
                                     'machine_id': '8dc98b6bfe8ae63d8de029965bf79b4b',
                                     'operating_system': 'linux',
                                     'os_image': 'Ubuntu 16.04.6 LTS',
                                     'system_uuid': '584756D2-95BF-0440-E146-107B44B12B73'},
                       'phase': None,
                       'volumes_attached': None,
                       'volumes_in_use': None}},
            ]
        }
        return [ {'nodeName': 'c', 'GPUID': 'c2'},
                 {'nodeName': 'a', 'GPUID': 'a1'},
                 {'nodeName': 'b', 'GPUID': 'b2'}
                ]
        """

        node_infos = self._client_service.list_node()

        node_gpus = dict()

        for node_info in node_infos.items:

            # if the node doesnt have gpu label or gpu is false, skip this node
            if NodeLabes.Gpu not in node_info.metadata.labels or not node_info.metadata.labels[NodeLabes.Gpu]:
                continue

            gpu_summary = node_info.metadata.labels[NodeLabes.GpuSummary]

            node_name = node_info.metadata.labels[NodeLabes.NodeName]

            for gpu_info in gpu_summary.split("."):
                gpu_device_id = gpu_info.split("_")[0]
                free_memory = gpu_info.split("_")[1]
                node_gpus[node_name+'__'+gpu_device_id] = int(free_memory)

        top_n = sorted(node_gpus.items(), key=lambda d: d[1], reverse=True)[:n]
        print("top_n:  ", top_n)
        node_gpuid = [{"nodeName": ele[0].split('__')[0], "GPUID": ele[0].split('__')[1]} for ele in top_n]

        print("node_gpuid:  ", node_gpuid)
        return node_gpuid

    def _create_clusterip_service_config(self, service_name, publish_port):
        content = \
            {'apiVersion': 'v1',
             'kind': 'Service',
             'metadata': {'labels': {'name': service_name},
                          'name': service_name},
             'spec': {'ports': [{'port': int(publish_port), 'targetPort': int(publish_port)}],
                      'selector': {'name': service_name},
                      'type': 'ClusterIP'}}

        return content

    def _create_service_config(self,
                               service_name,
                               docker_image,
                               replicas,
                               args,
                               environment_vars,
                               mounts=None,
                               publish_port=None,
                               gpus=0):
        #admin service
        if mounts is None:
            mounts = {}
        content = {}
        content.setdefault('apiVersion', 'v1')
        content.setdefault('kind', 'Service')
        metadata = content.setdefault('metadata', {})
        metadata.setdefault('name', service_name)
        labels = metadata.setdefault('labels', {})
        labels.setdefault('name', service_name)
        spec = content.setdefault('spec', {})
        if publish_port is not None:
            spec.setdefault('type', 'NodePort')
            ports = spec.setdefault('ports', [])
            ports.append({
                'port': int(publish_port[1]),
                'targetPort': int(publish_port[1]),
                'nodePort': int(publish_port[0])
            })
        spec.setdefault('selector', {'name': service_name})
        return content


# Decorator that retries a method call a number of times
def _retry(func):
    wait_secs = RETRY_WAIT_SECS

    @wraps(func)
    def retried_func(*args, **kwargs):
        for no in range(RETRY_TIMES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f'Error when calling `{func}`:')
                logger.error(traceback.format_exc())

                # Retried so many times but still errors - raise exception
                if no == RETRY_TIMES:
                    raise e

            logger.info(f'Retrying {func} after {wait_secs}s...')
            time.sleep(wait_secs)

    return retried_func
