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
import sys


def ptitle(s):
    screen_width = 100
    txt_width = len(s) + 6
    box_width = 3
    left_margin = (screen_width - txt_width) // 2

    print()
    print(' ' * left_margin + '+' + '-' * (txt_width) + '+')
    print(' ' * left_margin + '|' + ' ' * (txt_width) + '|')
    print(' ' * left_margin + '|' + ' ' * (box_width) + s + ' ' * (box_width) + '|')
    print(' ' * left_margin + '|' + ' ' * (txt_width) + '|')
    print(' ' * left_margin + '+' + '-' * (txt_width) + '+')
    print()


def start_all():
    model_path = os.path.dirname(__file__).rsplit("/", 1)[0]
    sys.path.append(model_path)
    ptitle("Start Services")
    os.environ['HOST_WORKDIR_PATH'] = model_path
    os.environ['APP_MODE'] = "PROD"
    os.system("bash {}".format(model_path + "/scripts/docker_swarm/start.sh"))


def stop_all():
    model_path = os.path.dirname(__file__).rsplit("/", 1)[0]
    sys.path.append(model_path)
    ptitle("Stop Services")
    os.environ['HOST_WORKDIR_PATH'] = model_path
    os.environ['APP_MODE'] = "PROD"
    os.system("bash {}".format(model_path + "/scripts/docker_swarm/stop.sh"))


def clean():
    model_path = os.path.dirname(__file__).rsplit("/", 1)[0]
    sys.path.append(model_path)
    ptitle("Clean Files")
    os.environ['HOST_WORKDIR_PATH'] = model_path
    os.environ['APP_MODE'] = "PROD"
    os.system("bash {}".format(model_path + "/scripts/clean.sh"))


def start_admin():
    model_path = os.path.dirname(__file__).rsplit("/", 1)[0]
    sys.path.append(model_path)
    os.environ['APP_MODE'] = "PROD"
    os.system("python {}".format(model_path + "/scripts/start_admin.py"))


def start_predictor():
    model_path = os.path.dirname(__file__).rsplit("/", 1)[0]
    sys.path.append(model_path)
    os.environ['APP_MODE'] = "PROD"
    os.system("python {}".format(model_path + "/scripts/start_predictor.py"))


def start_worker():
    model_path = os.path.dirname(__file__).rsplit("/", 1)[0]
    sys.path.append(model_path)
    os.environ['APP_MODE'] = "PROD"
    os.system("python {}".format(model_path + "/scripts/start_worker.py"))


from singa_auto.client import Client

